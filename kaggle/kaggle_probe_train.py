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
from data.data import flatten_dialogue, flatten_general, load_dialogue, load_general
from data.split import response_level_split
from data.tokenize_align import align_response_span, align_sub_span, get_tokenizer
from fixtures import MARIE_CURIE_FIXTURE

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
        hidden_states = out.hidden_states

        for i, (record, a) in enumerate(zip(batch, aligned)):
            token_start, token_end = a["token_span"]
            split_name = split_of_response[record["response_id"]]
            for L in range(1, NUM_LAYERS + 1):
                span_feats = hidden_states[L][i, token_start:token_end, :].detach().to("cpu", dtype=torch.float32)
                layer_data[L].append((span_feats, record["label"], split_name))

        del out, hidden_states
        torch.cuda.empty_cache()

    return layer_data


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
    layer_data: list[tuple[torch.Tensor, int, str]], epochs: int = 100, lr: float = 0.01, batch_size: int = 32
) -> tuple[nn.Linear, float]:
    """Span-max MIL probe: per-token sigmoid(w.h+b), response score = max over
    its token span, trained via BCE against the response-level label. Batched
    with masking -- mathematically identical to per-example max(sigmoid(x))
    since sigmoid is monotonic (max(sigmoid(x)) == sigmoid(max(x))), just
    vectorized instead of a Python loop over individual examples."""
    train_feats = [f for f, lbl, sp in layer_data if sp == "train"]
    train_labels = torch.tensor([float(lbl) for f, lbl, sp in layer_data if sp == "train"])
    val_feats = [f for f, lbl, sp in layer_data if sp == "val"]
    val_labels = [lbl for f, lbl, sp in layer_data if sp == "val"]

    hidden_dim = train_feats[0].shape[-1]
    max_len = max(f.shape[0] for f in train_feats + val_feats)

    train_padded, train_mask = _pad_stack(train_feats, max_len, hidden_dim)
    val_padded, val_mask = _pad_stack(val_feats, max_len, hidden_dim)

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
            loss = F.binary_cross_entropy_with_logits(max_logits, train_labels[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_padded).squeeze(-1).masked_fill(~val_mask, float("-inf"))
        val_scores = torch.sigmoid(val_logits.max(dim=1).values).numpy()
    auroc = roc_auc_score(val_labels, val_scores)
    return probe, auroc


def main(config: dict, subset_size: int) -> None:
    records = load_dialogue()
    flat = flatten_dialogue(records)
    splits = response_level_split(flat)
    split_of_response = {}
    for name, rows in splits.items():
        for r in rows:
            split_of_response[r["response_id"]] = name

    n_train = int(subset_size * 0.8)
    n_val = subset_size - n_train
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

    results = {}
    t0 = time.time()
    trained_probes = {}
    for L in range(1, NUM_LAYERS + 1):
        probe, auroc = train_probe(layer_data[L])
        results[L] = auroc
        trained_probes[L] = probe
        delta = auroc - length_auroc
        flag = "-- beats length baseline" if delta > 0.02 else "-- DOES NOT clearly beat length baseline"
        print(f"layer {L}: val AUROC {auroc:.4f} (length baseline {length_auroc:.4f}, delta {delta:+.4f}) {flag}")
    train_elapsed = time.time() - t0
    print(f"probe training+eval time for all layers: {train_elapsed:.1f}s")

    best_layer = max(results, key=results.get)
    print(f"best layer: {best_layer} (val AUROC {results[best_layer]:.4f}, length baseline {length_auroc:.4f})")

    try:
        import matplotlib.pyplot as plt

        layers = sorted(results.keys())
        aurocs = [results[L] for L in layers]
        plt.figure(figsize=(8, 4))
        plt.plot(layers, aurocs, marker="o", label="probe")
        plt.axhline(length_auroc, color="red", linestyle="--", label="length-only baseline")
        plt.xlabel("layer")
        plt.ylabel("val AUROC")
        plt.title("Probe layer sweep vs length-only baseline")
        plt.legend()
        plt.grid(True)
        plt.savefig("/kaggle/working/layer_sweep.png")
        print("saved layer sweep plot to /kaggle/working/layer_sweep.png")
    except Exception as e:
        print(f"plotting failed (non-fatal): {e}")

    best_probe = trained_probes[best_layer]

    # Marie Curie fixture: rebuilt in dialogue style so its length isn't itself
    # a confound; probe should score it low/clean (misses the cross-clause
    # contradiction) since every entity in it is factually correct.
    fixture_aligned = align_response_span(MARIE_CURIE_FIXTURE)
    input_ids = torch.tensor([fixture_aligned["input_ids"]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            output_hidden_states=True,
        )
    token_start, token_end = fixture_aligned["token_span"]
    span_feats = out.hidden_states[best_layer][0, token_start:token_end, :].detach().to("cpu", dtype=torch.float32)
    with torch.no_grad():
        fixture_score = torch.sigmoid(best_probe(span_feats)).max().item()
    fixture_len = len(MARIE_CURIE_FIXTURE["response"])
    print(
        f"marie curie fixture probe score at layer {best_layer}: {fixture_score:.4f} "
        f"(response length {fixture_len} chars, dialogue hallucinated mean ~113; expect LOW score -- "
        "probe should miss the cross-clause contradiction since every entity is correct)"
    )
    del out
    torch.cuda.empty_cache()

    # Qualitative check on general_data.json's real hallucination_spans -- held
    # out entirely from training. Does the best-layer probe's peak token score
    # actually land inside the annotated fabrication span?
    general_records = load_general()
    general_flat = flatten_general(general_records)
    candidates = [r for r in general_flat if r["label"] == 1 and r["hallucination_spans"]]
    # keep only records whose span text is a literal, findable substring of the
    # response -- some HaluEval annotations join non-contiguous excerpts or use
    # a placeholder like "Incomplete answer" instead of a quoted span
    span_examples = [r for r in candidates if r["hallucination_spans"][0] in r["response"]][:5]
    print(f"general_data.json qualitative span check: {len(span_examples)} examples (of {len(candidates)} candidates)")

    for r in span_examples:
        resp_aligned = align_response_span(r)
        resp_ts, resp_te = resp_aligned["token_span"]
        if resp_ts is None or resp_te is None or resp_te <= resp_ts:
            print("  skip (no valid response token span)")
            continue

        input_ids = torch.tensor([resp_aligned["input_ids"]])
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                output_hidden_states=True,
            )
        span_feats = out.hidden_states[best_layer][0, resp_ts:resp_te, :].detach().to("cpu", dtype=torch.float32)
        with torch.no_grad():
            token_scores = torch.sigmoid(best_probe(span_feats)).squeeze(-1)
        peak_offset = int(token_scores.argmax().item())
        peak_token_idx = resp_ts + peak_offset

        span_text = r["hallucination_spans"][0]
        sub_aligned = align_sub_span(r, span_text)
        del out
        torch.cuda.empty_cache()

        if sub_aligned is None or sub_aligned["token_span"][0] is None:
            print(f"  span text not found verbatim in response, skipping: {span_text[:60]!r}")
            continue

        sub_ts, sub_te = sub_aligned["token_span"]
        hit = sub_ts <= peak_token_idx < sub_te
        print(
            f"  peak token idx {peak_token_idx} (response span [{resp_ts},{resp_te})), "
            f"annotated fabrication span tokens [{sub_ts},{sub_te}) -- peak in span: {hit}"
        )


if __name__ == "__main__":
    cfg = load_config(os.path.join(REPO_ROOT, "config.yaml"))
    main(cfg, subset_size=1500)
