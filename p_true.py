TRUE_TOKEN_IDS = [2514, 3007]  # "True", " True"
FALSE_TOKEN_IDS = [4049, 3557]  # "False", " False"


def _instruction(question: str) -> str:
    return f"{question}\nAnswer with exactly one word: True or False."


def build_input_contradict_messages(record: dict) -> list[dict]:
    content = (
        f"Knowledge: {record['knowledge']}\n"
        f"Response: {record['response']}\n\n"
        + _instruction("Does the response contradict the given knowledge above?")
    )
    return [{"role": "user", "content": content}]


def build_self_contradict_messages(record: dict) -> list[dict]:
    content = f"Response: {record['response']}\n\n" + _instruction(
        "Does the response contradict itself -- does it contain two parts that assert incompatible things?"
    )
    return [{"role": "user", "content": content}]


def build_fact_contradict_messages(record: dict) -> list[dict]:
    content = f"Response: {record['response']}\n\n" + _instruction(
        "Is the response factually incorrect, based on general world knowledge?"
    )
    return [{"role": "user", "content": content}]


VARIANTS = {
    "p_input_contradict": build_input_contradict_messages,
    "p_self_contradict": build_self_contradict_messages,
    "p_fact_contradict": build_fact_contradict_messages,
}
