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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import load_config

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

_SYNTHETIC_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What is the boiling point of water at sea level?", "Water boils at 100 degrees Celsius at sea level."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the largest planet in the solar system?", "Jupiter is the largest planet in the solar system."),
    ("What year did World War II end?", "World War II ended in 1945."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
]


def main(config: dict, limit: int) -> None:
    pairs = _SYNTHETIC_PAIRS[:limit]
    prompts = [f"Question: {q}\nAnswer: {a}" for q, a in pairs]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
    forward_elapsed = time.time() - t0

    hidden_states = out.hidden_states
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    per_layer_bytes = hidden_states[0].numel() * hidden_states[0].element_size()

    print(f"batch size (num prompt+response pairs): {len(prompts)}")
    print(f"model load time: {load_elapsed:.1f}s")
    print(f"forward pass time: {forward_elapsed:.1f}s")
    print(f"num hidden_states tensors: {len(hidden_states)}")
    print(f"per-tensor shape: {tuple(hidden_states[0].shape)}")
    print(f"per-tensor dtype: {hidden_states[0].dtype}")
    print(f"peak GPU memory: {peak_mem_gb:.2f} GB")
    print(f"per-layer activation size for this batch: {per_layer_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    cfg = load_config(os.path.join(CODE_DIR, "config.yaml"))
    main(cfg, limit=8)
