MARIE_CURIE_FIXTURE = {
    "knowledge": (
        "Marie Curie was a physicist and chemist who conducted pioneering research on "
        "radioactivity. She won two Nobel Prizes. Later in life she struggled to fund "
        "her research and, lacking money, sold some of her Nobel Prize medals."
    ),
    "context": (
        "[Human]: Tell me about Marie Curie's research funding. "
        "[Assistant]: She conducted groundbreaking work on radioactivity and won two Nobel Prizes. "
        "[Human]: Was her research well funded?"
    ),
    "response": (
        "Her research was generously funded by the French government, though she later sold "
        "her Nobel Prize medals for lack of funding."
    ),
    "label": 1,
}

OBVIOUSLY_TRUE_FIXTURE = {
    "knowledge": "Paris is the capital city of France.",
    "context": "[Human]: What is the capital of France?",
    "response": "Paris is the capital of France.",
    "label": 0,
}

OBVIOUSLY_FALSE_FIXTURE = {
    "knowledge": "Paris is the capital city of France.",
    "context": "[Human]: What is the capital of France?",
    "response": "Berlin is the capital of France.",
    "label": 1,
}
