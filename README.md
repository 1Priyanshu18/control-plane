# control_planeAI

Hybrid state-aware agent control system: a control layer around an LLM agent that
classifies risk, runs only the relevant checks, and intervenes (verify/retry/
regenerate/redact/restrict/block) via a deterministic policy engine.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
