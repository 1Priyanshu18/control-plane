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

subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes"], check=True)

CODE_DIR = None
for dirpath, _dirnames, filenames in os.walk("/kaggle/input"):
    if "config.py" in filenames:
        CODE_DIR = dirpath
        break
if CODE_DIR is None:
    raise RuntimeError(f"could not find config.py under /kaggle/input; tree: {list(os.walk('/kaggle/input'))}")
sys.path.insert(0, CODE_DIR)

from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from baseline import length_baseline_auroc
from config import load_config
from data import flatten_dialogue, load_dialogue
from fixtures import MARIE_CURIE_FIXTURE, OBVIOUSLY_FALSE_FIXTURE, OBVIOUSLY_TRUE_FIXTURE
from p_true import FALSE_TOKEN_IDS, TRUE_TOKEN_IDS, VARIANTS
from split import response_level_split

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def stratified_subset(rows: list[dict], n: int) -> list[dict]:
    return random.sample(rows, min(n, len(rows)))


def _to_prompt_text(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_p_true_batch(model, tokenizer, messages_list: list[list[dict]], batch_size: int = 8) -> list[float]:
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
            out = model(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
            )

        for i, L in enumerate(lengths):
            logits = out.logits[i, L - 1, :].float()
            true_mass = torch.logsumexp(logits[TRUE_TOKEN_IDS], dim=0)
            false_mass = torch.logsumexp(logits[FALSE_TOKEN_IDS], dim=0)
            p_true = torch.softmax(torch.stack([true_mass, false_mass]), dim=0)[0].item()
            scores.append(p_true)

        del out
        torch.cuda.empty_cache()
    return scores


def debug_top_tokens(model, tokenizer, messages: list[dict], k: int = 8) -> None:
    prompt = _to_prompt_text(tokenizer, messages)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=0)
    top_probs, top_ids = probs.topk(k)
    tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    print("    top tokens:", list(zip(tokens, [round(p, 4) for p in top_probs.tolist()])))


def main(config: dict, sample_size: int) -> None:
    records = load_dialogue()
    flat = flatten_dialogue(records)
    splits = response_level_split(flat)

    val_sample = stratified_subset(splits["val"], sample_size)
    train_for_baseline = stratified_subset(splits["train"], 1000)
    length_auroc = length_baseline_auroc(train_for_baseline, val_sample)
    print(f"length-only baseline val AUROC (standing control): {length_auroc:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quant_config, device_map="auto")

    print()
    print("sanity pair (obviously true vs obviously false statement):")
    for name, builder in VARIANTS.items():
        p_true = read_p_true_batch(model, tokenizer, [builder(OBVIOUSLY_TRUE_FIXTURE)])[0]
        p_false = read_p_true_batch(model, tokenizer, [builder(OBVIOUSLY_FALSE_FIXTURE)])[0]
        print(f"  {name}: obviously-true response P(True)={p_true:.4f}, obviously-false response P(True)={p_false:.4f}")
        if name == "p_input_contradict":
            print("  debug top tokens for obviously-false case:")
            debug_top_tokens(model, tokenizer, builder(OBVIOUSLY_FALSE_FIXTURE))

    print()
    print("marie curie fixture (all entities correct, self-contradicts across clauses):")
    for name, builder in VARIANTS.items():
        score = read_p_true_batch(model, tokenizer, [builder(MARIE_CURIE_FIXTURE)])[0]
        print(f"  {name}: P(True)={score:.4f}")

    print()
    print(f"sample AUROC on {len(val_sample)} val responses:")
    labels = [r["label"] for r in val_sample]
    for variant_name, builder in VARIANTS.items():
        prompts = [builder(r) for r in val_sample]
        scores = read_p_true_batch(model, tokenizer, prompts)
        auroc = roc_auc_score(labels, scores)
        delta = auroc - length_auroc
        flag = "-- beats length baseline" if delta > 0.02 else "-- DOES NOT clearly beat length baseline"
        print(f"  {variant_name}: val AUROC {auroc:.4f} (length baseline {length_auroc:.4f}, delta {delta:+.4f}) {flag}")


if __name__ == "__main__":
    cfg = load_config(os.path.join(CODE_DIR, "config.yaml"))
    main(cfg, sample_size=200)
