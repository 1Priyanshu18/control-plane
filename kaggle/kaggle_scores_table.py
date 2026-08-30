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

import csv
import os
import random
import subprocess
import sys
import time

subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes"], check=True)

SRC_DIR = None
for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
    if "config.py" in filenames and os.path.basename(dirpath) == "src":
        SRC_DIR = dirpath
        break
if SRC_DIR is None:
    raise RuntimeError(f"could not find src/config.py under /kaggle/input; tree: {list(os.walk('/kaggle/input'))}")
REPO_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from baseline import length_baseline_auroc
from config import load_config
from data.data import flatten_dialogue, load_dialogue
from data.split import response_level_split
from data.tokenize_align import align_response_span, get_tokenizer
from detectors.p_true import VARIANTS

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
    """Returns [(response_id, response_span_features, label), ...] for one layer."""
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


def score_with_probe(probe: nn.Linear, feats: torch.Tensor) -> float:
    with torch.no_grad():
        return torch.sigmoid(probe(feats)).max().item()


def _to_prompt_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_p_true_batch(model, tokenizer, messages_list: list[list[dict]], batch_size: int = 8) -> list[float]:
    from detectors.p_true import FALSE_TOKEN_IDS, TRUE_TOKEN_IDS

    prompts = [_to_prompt_text(tokenizer, m) for m in messages_list]
    scores = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = [tokenizer(p, return_tensors=None, add_special_tokens=False)["input_ids"] for p in batch]
        lengths = [len(ids) for ids in encoded]
        max_len = max(lengths)
        pad_id = tokenizer.pad_token_id

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attention_mask[i, : len(ids)] = 1

        with torch.no_grad():
            out = model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

        for i, L in enumerate(lengths):
            logits = out.logits[i, L - 1, :].float()
            true_mass = torch.logsumexp(logits[TRUE_TOKEN_IDS], dim=0)
            false_mass = torch.logsumexp(logits[FALSE_TOKEN_IDS], dim=0)
            p_true = torch.softmax(torch.stack([true_mass, false_mass]), dim=0)[0].item()
            scores.append(p_true)

        del out
        torch.cuda.empty_cache()
    return scores


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

    split_of_response = {}
    for r in subset_train:
        split_of_response[r["response_id"]] = "train"
    for r in subset_val:
        split_of_response[r["response_id"]] = "val"
    train_ids = {r["response_id"] for r in subset_train}

    train_feats = [f for rid, f, lbl in extracted if rid in train_ids]
    train_labels = [lbl for rid, f, lbl in extracted if rid in train_ids]
    val_feats = [f for rid, f, lbl in extracted if rid not in train_ids]
    val_labels = [lbl for rid, f, lbl in extracted if rid not in train_ids]

    t0 = time.time()
    probe, probe_auroc = train_probe(train_feats, train_labels, val_feats, val_labels)
    print(f"probe (layer {BEST_LAYER}) val AUROC: {probe_auroc:.4f} (length baseline {length_auroc:.4f}), "
          f"trained in {time.time() - t0:.1f}s")

    probe_scores = {rid: score_with_probe(probe, feats) for rid, feats, lbl in extracted}

    t0 = time.time()
    ptrue_scores: dict[str, dict[str, float]] = {name: {} for name in VARIANTS}
    for variant_name, builder in VARIANTS.items():
        messages_list = [builder(r) for r in subset]
        scores = read_p_true_batch(model, tokenizer, messages_list)
        for r, s in zip(subset, scores):
            ptrue_scores[variant_name][r["response_id"]] = s
        auroc = roc_auc_score([r["label"] for r in subset_val], [ptrue_scores[variant_name][r["response_id"]] for r in subset_val])
        print(f"{variant_name}: val AUROC {auroc:.4f}")
    print(f"P(True) all variants time: {time.time() - t0:.1f}s")

    out_path = "/kaggle/working/scores_partial.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["response_id", "split", "label", "knowledge", "context", "response",
             "probe", "p_input_contradict", "p_self_contradict", "p_fact_contradict"]
        )
        for r in subset:
            rid = r["response_id"]
            writer.writerow([
                rid,
                split_of_response[rid],
                r["label"],
                r["knowledge"],
                r["context"],
                r["response"],
                probe_scores[rid],
                ptrue_scores["p_input_contradict"][rid],
                ptrue_scores["p_self_contradict"][rid],
                ptrue_scores["p_fact_contradict"][rid],
            ])
    print(f"wrote {out_path} with {len(subset)} rows")


if __name__ == "__main__":
    cfg = load_config(os.path.join(REPO_ROOT, "config.yaml"))
    main(cfg, sample_size=1000)
