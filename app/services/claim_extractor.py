import re
from typing import List

from app.core.config import settings


# Regex for sentence splitting
SENTENCE_SPLITTER_RE = re.compile(r'(?<=[.!?])\s+')


def extract_claims(text: str, max_claims: int = None, min_len: int = 30) -> List[str]:
    """
    Extract factual claims from text using heuristic-based approach.

    Args:
        text: Input text to extract claims from
        max_claims: Maximum number of claims to return. If None, uses settings.max_claims
        min_len: Minimum length of a claim in characters

    Returns:
        List of extracted claim strings
    """
    if max_claims is None:
        max_claims = settings.max_claims

    # Split into sentences
    sents = SENTENCE_SPLITTER_RE.split(text)
    candidates = []

    for s in sents:
        s = s.strip()
        if len(s) < min_len:
            continue

        # Heuristic filter: contains factual indicators
        # - Verbs like is/was/won/died/born/founded (factual statements)
        # - Contains digits (dates, numbers, statistics)
        has_factual_verb = bool(re.search(
            r'\b(is|was|are|were|won|died|born|founded|established|announced|reported|has|have|had)\b',
            s,
            re.IGNORECASE
        ))
        has_digit = bool(re.search(r'\d', s))

        if has_factual_verb or has_digit:
            candidates.append(s)

    # Fallback: if no candidates found, take longest sentences
    if not candidates:
        candidates = sorted(
            [s for s in sents if len(s) >= min_len],
            key=len,
            reverse=True
        )[:max_claims]

    # Return up to max_claims
    return candidates[:max_claims]
