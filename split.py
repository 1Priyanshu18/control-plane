import random


def response_level_split(
    flat_records: list[dict], train: float = 0.7, val: float = 0.15
) -> dict[str, list[dict]]:
    """Splits by group_id (shared question) so a response and its paired
    right/hallucinated counterpart never land in different splits."""
    group_ids = sorted({r["group_id"] for r in flat_records})
    random.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * train)
    n_val = int(n * val)

    split_of_group = {}
    for i, gid in enumerate(group_ids):
        if i < n_train:
            split_of_group[gid] = "train"
        elif i < n_train + n_val:
            split_of_group[gid] = "val"
        else:
            split_of_group[gid] = "test"

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in flat_records:
        splits[split_of_group[r["group_id"]]].append(r)
    return splits
