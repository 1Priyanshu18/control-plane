import torch

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device available")

_device_count = torch.cuda.device_count()
_device_name = torch.cuda.get_device_name(0)
_major, _minor = torch.cuda.get_device_capability(0)
_capability = _major + _minor / 10

print(f"device count: {_device_count}")
print(f"device name: {_device_name}")
print(f"compute capability: {_major}.{_minor}")

if _capability < 7.5:
    raise SystemExit(
        f"GPU compute capability {_major}.{_minor} ({_device_name}) is below the 7.5 "
        "minimum bitsandbytes 4-bit quantization requires; aborting before weight download."
    )

import os
import random
import subprocess
import sys
import time

subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes"], check=True)

CODE_DIR = None
for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
    if "config.py" in filenames:
        CODE_DIR = dirpath
        break
if CODE_DIR is None:
    raise RuntimeError(f"could not find config.py under /kaggle/input; tree: {list(os.walk('/kaggle/input'))}")
sys.path.insert(0, CODE_DIR)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from baseline import length_baseline_auroc
from config import load_config
from data import flatten_dialogue, load_dialogue
from split import response_level_split
from tokenize_align import align_response_span, get_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BEST_LAYER = 17


def stratified_subset(rows: list[dict], n: int) -> list[dict]:
    return random.sample(rows, min(n, len(rows)))


def collate(tokenizer, batch_records: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    aligned = [align_response_span(r) for r in batch_records]
    max_len = max(len(a["input_ids"]) for a in aligned)
    pad_id = tokenizer.pad_token_id

    input_ids = torch.full((len(aligned), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(aligned), max_len), dtype=torch.long)
    for i, a in enumerate(aligned):
        ids = a["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask, aligned


def extract_single_layer(
    model, tokenizer, rows: list[dict], layer: int, batch_size: int = 4
) -> list[tuple[str, torch.Tensor, int]]:
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention_mask, aligned = collate(tokenizer, batch)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                output_hidden_states=True,
            )
        hidden_states = out.hidden_states
        for i, (record, a) in enumerate(zip(batch, aligned)):
            token_start, token_end = a["token_span"]
            span_feats = hidden_states[layer][i, token_start:token_end, :].detach().to("cpu", dtype=torch.float32)
            results.append((record["response_id"], span_feats, record["label"]))
        del out, hidden_states
        torch.cuda.empty_cache()
    return results


def _pad_stack(feats: list[torch.Tensor], max_len: int, hidden_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(feats)
    padded = torch.zeros(n, max_len, hidden_dim)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, f in enumerate(feats):
        L = f.shape[0]
        padded[i, :L] = f
        mask[i, :L] = True
    return padded, mask


def train_probe(
    train_feats: list[torch.Tensor],
    train_labels: list[int],
    val_feats: list[torch.Tensor],
    val_labels: list[int],
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
) -> tuple[nn.Linear, float]:
    hidden_dim = train_feats[0].shape[-1]
    max_len = max(f.shape[0] for f in train_feats + val_feats)

    train_padded, train_mask = _pad_stack(train_feats, max_len, hidden_dim)
    val_padded, val_mask = _pad_stack(val_feats, max_len, hidden_dim)
    train_labels_t = torch.tensor([float(lbl) for lbl in train_labels])

    probe = nn.Linear(hidden_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    n_train = train_padded.shape[0]
    for _epoch in range(epochs):
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            logits = probe(train_padded[idx]).squeeze(-1)
            logits = logits.masked_fill(~train_mask[idx], float("-inf"))
            max_logits = logits.max(dim=1).values
            loss = F.binary_cross_entropy_with_logits(max_logits, train_labels_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_padded).squeeze(-1).masked_fill(~val_mask, float("-inf"))
        val_scores = torch.sigmoid(val_logits.max(dim=1).values).numpy()
    auroc = roc_auc_score(val_labels, val_scores)
    return probe, auroc


def main(config: dict, sample_size: int) -> None:
    records = load_dialogue()
    flat = flatten_dialogue(records)
    splits = response_level_split(flat)

    n_train = int(sample_size * 0.8)
    n_val = sample_size - n_train
    subset_train = stratified_subset(splits["train"], n_train)
    subset_val = stratified_subset(splits["val"], n_val)
    subset = subset_train + subset_val
    print(f"subset size: {len(subset)} ({n_train} train, {n_val} val)")

    length_auroc = length_baseline_auroc(subset_train, subset_val)
    print(f"length-only baseline val AUROC (standing control): {length_auroc:.4f}")

    tokenizer = get_tokenizer()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")
    print(f"model load time: {time.time() - t0:.1f}s")

    t0 = time.time()
    extracted = extract_single_layer(model, tokenizer, subset, BEST_LAYER)
    print(f"layer {BEST_LAYER} extraction time for {len(subset)} responses: {time.time() - t0:.1f}s")

    train_ids = {r["response_id"] for r in subset_train}
    train_feats = [f for rid, f, lbl in extracted if rid in train_ids]
    train_labels = [lbl for rid, f, lbl in extracted if rid in train_ids]
    val_feats = [f for rid, f, lbl in extracted if rid not in train_ids]
    val_labels = [lbl for rid, f, lbl in extracted if rid not in train_ids]

    t0 = time.time()
    probe, probe_auroc = train_probe(train_feats, train_labels, val_feats, val_labels)
    print(f"probe (layer {BEST_LAYER}) val AUROC: {probe_auroc:.4f} (length baseline {length_auroc:.4f}), "
          f"trained in {time.time() - t0:.1f}s")

    w = probe.weight.detach().numpy().reshape(-1)
    b = probe.bias.detach().numpy().reshape(-1)
    np.savez("/kaggle/working/probe_weights.npz", w=w, b=b, layer=BEST_LAYER, val_auroc=probe_auroc)
    print(f"saved probe_weights.npz: w shape {w.shape}, b {b}, layer {BEST_LAYER}, val_auroc {probe_auroc:.4f}")


if __name__ == "__main__":
    cfg = load_config(os.path.join(CODE_DIR, "config.yaml"))
    main(cfg, sample_size=1000)
