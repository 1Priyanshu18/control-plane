# control_planeAI

Hybrid state-aware agent control system: a control layer around an LLM agent that
classifies risk, runs only the relevant checks, and intervenes (verify/retry/
regenerate/redact/restrict/block) via a deterministic policy engine.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

PII detection (`presidio-analyzer`) needs a spaCy model, pulled separately from
pip -- the last line above fetches it.

## LangSmith tracing (optional)

Observability only -- the agent never reads from or blocks on it. To enable,
set these before running (never commit a key):

```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_API_KEY = "<your key>"
$env:LANGCHAIN_PROJECT = "control-planeai"
python eval/tracing_demo.py
```
