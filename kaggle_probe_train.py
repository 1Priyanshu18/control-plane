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

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from config import load_config
from data import flatten_responses, load_qa
from split import response_level_split
from tokenize_align import align_response_span, get_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_LAYERS = 28


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


def extract_layer_features(
    model, tokenizer, rows: list[dict], split_of_response: dict, batch_size: int = 4
) -> dict[int, list[tuple[torch.Tensor, int, str]]]:
    """Returns {layer_idx: [(response_span_features, label, split_name), ...]}."""
    layer_data: dict[int, list[tuple[torch.Tensor, int, str]]] = {L: [] for L in range(1, NUM_LAYERS + 1)}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        input_ids, attention_mask, aligned = collate(tokenizer, batch)

        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                output_hidden_states=True,
            )
        hidden_states = out.hidden_states  # tuple of NUM_LAYERS+1 tensors, (batch, seq_len, hidden_dim)

        for i, (record, a) in enumerate(zip(batch, aligned)):
            token_start, token_end = a["token_span"]
            split_name = split_of_response[record["response_id"]]
            for L in range(1, NUM_LAYERS + 1):
                span_feats = hidden_states[L][i, token_start:token_end, :].detach().to("cpu", dtype=torch.float32)
                layer_data[L].append((span_feats, record["label"], split_name))

        del out, hidden_states
        torch.cuda.empty_cache()

    return layer_data


def train_probe(
    layer_data: list[tuple[torch.Tensor, int, str]], epochs: int = 100, lr: float = 0.01
) -> tuple[nn.Linear, float]:
    train_feats = [f for f, lbl, sp in layer_data if sp == "train"]
    train_labels = [lbl for f, lbl, sp in layer_data if sp == "train"]
    val_feats = [f for f, lbl, sp in layer_data if sp == "val"]
    val_labels = [lbl for f, lbl, sp in layer_data if sp == "val"]

    hidden_dim = train_feats[0].shape[-1]
    probe = nn.Linear(hidden_dim, 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for _epoch in range(epochs):
        optimizer.zero_grad()
        losses = []
        for feat, label in zip(train_feats, train_labels):
            scores = torch.sigmoid(probe(feat)).squeeze(-1)
            max_score = scores.max()
            loss = F.binary_cross_entropy(max_score.unsqueeze(0), torch.tensor([float(label)]))
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_scores = [torch.sigmoid(probe(feat)).max().item() for feat in val_feats]
    auroc = roc_auc_score(val_labels, val_scores)
    return probe, auroc


def main(config: dict, subset_size: int) -> None:
    records = load_qa()
    flat = flatten_responses(records)
    splits = response_level_split(flat)
    split_of_response = {}
    for name, rows in splits.items():
        for r in rows:
            split_of_response[r["response_id"]] = name

    n_train = int(subset_size * 0.8)
    n_val = subset_size - n_train
    subset = stratified_subset(splits["train"], n_train) + stratified_subset(splits["val"], n_val)
    print(f"subset size: {len(subset)} ({n_train} train, {n_val} val)")

    tokenizer = get_tokenizer()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    load_elapsed = time.time() - t0
    print(f"model load time: {load_elapsed:.1f}s")

    t0 = time.time()
    layer_data = extract_layer_features(model, tokenizer, subset, split_of_response)
    extract_elapsed = time.time() - t0
    print(f"extraction time for {len(subset)} responses: {extract_elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

    results = {}
    t0 = time.time()
    for L in range(1, NUM_LAYERS + 1):
        _probe, auroc = train_probe(layer_data[L])
        results[L] = auroc
        print(f"layer {L}: val AUROC {auroc:.4f}")
    train_elapsed = time.time() - t0
    print(f"probe training+eval time for all layers: {train_elapsed:.1f}s")

    best_layer = max(results, key=results.get)
    print(f"best layer: {best_layer} (val AUROC {results[best_layer]:.4f})")

    try:
        import matplotlib.pyplot as plt

        layers = sorted(results.keys())
        aurocs = [results[L] for L in layers]
        plt.figure(figsize=(8, 4))
        plt.plot(layers, aurocs, marker="o")
        plt.xlabel("layer")
        plt.ylabel("val AUROC")
        plt.title("Probe layer sweep")
        plt.grid(True)
        plt.savefig("/kaggle/working/layer_sweep.png")
        print("saved layer sweep plot to /kaggle/working/layer_sweep.png")
    except Exception as e:
        print(f"plotting failed (non-fatal): {e}")

    fixture = __import__("fixtures").MARIE_CURIE_FIXTURE
    fixture_aligned = align_response_span(fixture)
    input_ids = torch.tensor([fixture_aligned["input_ids"]])
    attention_mask = torch.ones_like(input_ids)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            output_hidden_states=True,
        )
    token_start, token_end = fixture_aligned["token_span"]
    best_probe, _ = train_probe(layer_data[best_layer])
    span_feats = out.hidden_states[best_layer][0, token_start:token_end, :].detach().to("cpu", dtype=torch.float32)
    with torch.no_grad():
        fixture_score = torch.sigmoid(best_probe(span_feats)).max().item()
    print(f"marie curie fixture probe score at layer {best_layer}: {fixture_score:.4f} (expected: low/clean, since all entities are correct)")


if __name__ == "__main__":
    cfg = load_config(os.path.join(CODE_DIR, "config.yaml"))
    main(cfg, subset_size=200)
