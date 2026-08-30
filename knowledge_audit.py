# Manual audit of the n=100 test-split sample's `knowledge` field, done by
# Claude checking HaluEval's claims against its own training knowledge --
# NOT independently verified ground truth. See memory / Phase 10 writeup for
# the caveat this implies: an LLM's self-assessment of another dataset's
# factual claims carries the same reliability limits this whole project is
# about detecting in the first place.
#
# Tiers: "confirmed" (high confidence factual error), "probable" (moderate-
# good confidence, likely error), "uncertain" (genuine doubt, NOT counted as
# bad -- excluded from neither the clean nor bad set's exclusion criteria,
# stays in the "clean" set since it wasn't confirmed wrong).
#
# 1-indexed to match the printed eval log / audit transcript.

CONFIRMED_BAD = {3, 7, 12, 21, 23, 41, 46, 52, 58, 74, 86}
PROBABLE_BAD = {14, 29, 31, 34, 37, 81, 100}
UNCERTAIN = {20, 26, 72, 92}

BAD_KNOWLEDGE_INDICES = CONFIRMED_BAD | PROBABLE_BAD


def is_clean(index_1based: int) -> bool:
    return index_1based not in BAD_KNOWLEDGE_INDICES
