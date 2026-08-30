from typing import Callable

from detectors.nli import nli_contradiction_score

_PII_ANALYZER = None
_TOXICITY_CLF = None
_INJECTION_CLF = None

# LangKit (the brief's originally named PII/toxicity tool) doesn't install on
# this machine: its dependency whylogs-sketching has no prebuilt Windows
# wheel for Python 3.12 (only 3.10/3.11), and building from source needs a
# full MSVC C++ toolchain not installed here. Swapped for Presidio (PII) and
# a HuggingFace toxicity classifier -- both pure-Python/PyTorch, functionally
# equivalent, confirmed fast enough for the cheap/every-request tier.
#
# PERSON and LOCATION are deliberately excluded: they flag nearly any
# encyclopedic fact ("Gustave Eiffel", "Paris"), which isn't a meaningful
# privacy-risk signal on its own -- unlike genuinely identifying/contactable
# info. (Found this the hard way: it was masking real factuality-risk
# triggers behind a false PII redact decision in the P8 recovery demo.)
SENSITIVE_ENTITY_TYPES = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
}


def get_pii_analyzer():
    global _PII_ANALYZER
    if _PII_ANALYZER is None:
        from presidio_analyzer import AnalyzerEngine

        _PII_ANALYZER = AnalyzerEngine()
    return _PII_ANALYZER


def get_toxicity_classifier():
    global _TOXICITY_CLF
    if _TOXICITY_CLF is None:
        from transformers import pipeline

        _TOXICITY_CLF = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    return _TOXICITY_CLF


def get_injection_classifier():
    global _INJECTION_CLF
    if _INJECTION_CLF is None:
        from transformers import pipeline

        # Swapped from deepset/deberta-v3-base-injection after measuring a
        # real false-positive problem on HaluEval's knowledge-graph-style
        # tool output (see memory: injection-classifier-fp-measurement).
        # protectai-v2 cleared the same false-positive fixtures at >0.9999
        # confidence while still catching genuine injections -- verified,
        # not assumed.
        _INJECTION_CLF = pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2")
    return _INJECTION_CLF


def pii_check(text: str) -> dict:
    """Cheap: Presidio NER+regex PII scan. Keeps typed spans/offsets rather
    than collapsing to a scalar -- Phase 8's redact intervention needs to
    know exactly which span to redact, not just that something was found."""
    analyzer = get_pii_analyzer()
    results = analyzer.analyze(text=text, language="en")
    entities = [
        {"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results
    ]
    sensitive_scores = [e["score"] for e in entities if e["entity_type"] in SENSITIVE_ENTITY_TYPES]
    score = max(sensitive_scores) if sensitive_scores else 0.0
    return {"score": score, "entities": entities}


def toxicity_check(text: str) -> float:
    """Cheap: max toxicity-family label score from unitary/toxic-bert."""
    clf = get_toxicity_classifier()
    scores = clf(text)[0]
    return max(s["score"] for s in scores)


def injection_check(text: str) -> float:
    """Cheap: prompt-injection classifier. Must run on tool outputs and
    retrieved documents, not just user input -- a poisoned search result or
    webpage is the realistic attack surface, not just what the user typed."""
    clf = get_injection_classifier()
    result = clf(text)[0]
    return result["score"] if result["label"] == "INJECTION" else 1.0 - result["score"]


def factuality_check(response: str) -> float:
    """Cheap: NLI self-contradiction over the response's own clauses."""
    return nli_contradiction_score({"knowledge": "", "response": response})


def self_consistency_check(generate: Callable[..., str], user_input: str) -> float:
    """Expensive: samples the generator twice (costs an extra llm_call) and
    checks whether the two generations agree. Mirrors SelfCheckGPT-style
    sampling -- only worth the extra call when the cheap signal is unclear."""
    first = generate(user_input, attempt=0)
    second = generate(user_input, attempt=1)
    return 0.0 if first == second else 1.0
