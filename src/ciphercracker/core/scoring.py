from __future__ import annotations

import re
from collections import Counter
from importlib import resources
from typing import Iterable

from ciphercracker.core.ngrams import QuadgramScorer
from .utils import normalize_keep_spaces, normalize_az, printable_ratio, shannon_entropy

# ----------------------------
# Quadgram scorer (cached)
# ----------------------------

_QUAD_SCORER: QuadgramScorer | None = None
_QUAD_ERROR: str | None = None

def get_quadgram_scorer() -> QuadgramScorer | None:
    """Load cached quadgram scorer from ciphercracker.data/english_quadgrams.txt."""
    global _QUAD_SCORER, _QUAD_ERROR
    if _QUAD_SCORER is not None:
        return _QUAD_SCORER

    try:
        # Touch the file so missing-package-data fails early with a helpful error
        _ = resources.files("ciphercracker.data").joinpath("english_quadgrams.txt").read_text(
            encoding="utf-8"
        )[:10]
        _QUAD_SCORER = QuadgramScorer.from_package_data("english_quadgrams.txt")
        _QUAD_ERROR = None
        return _QUAD_SCORER
    except Exception as e:
        _QUAD_ERROR = repr(e)
        _QUAD_SCORER = None
        return None

def get_quadgram_error() -> str | None:
    return _QUAD_ERROR

def quadgram_score(text: str) -> float:
    """Raw quadgram log score. Higher is better (less negative)."""
    scorer = get_quadgram_scorer()
    if scorer is None:
        return float("-inf")
    return scorer.score(text)


# ----------------------------
# English heuristics
# ----------------------------

_ENGLISH_FREQ = {
    "E": 0.1270, "T": 0.0906, "A": 0.0817, "O": 0.0751, "I": 0.0697, "N": 0.0675,
    "S": 0.0633, "H": 0.0609, "R": 0.0599, "D": 0.0425, "L": 0.0403, "C": 0.0278,
    "U": 0.0276, "M": 0.0241, "W": 0.0236, "F": 0.0223, "G": 0.0202, "Y": 0.0197,
    "P": 0.0193, "B": 0.0149, "V": 0.0098, "K": 0.0077, "J": 0.0015, "X": 0.0015,
    "Q": 0.0010, "Z": 0.0007,
}

# Small fallback list (good enough to start). You can later load a bigger file.
_COMMON_WORDS = {
    "THE", "AND", "TO", "OF", "IN", "IS", "IT", "YOU", "THAT", "A", "I", "FOR", "ON",
    "WITH", "AS", "ARE", "THIS", "BE", "WAS", "HAVE", "NOT", "OR", "AT", "BY",
    "FROM", "ONE", "ALL", "WE", "THEY", "HAS", "CAN", "WILL", "DO", "IF", "AN",
}

_WORD_RE = re.compile(r"[A-Z]{2,}")

def chi_squared_english(az_text: str) -> float:
    """Lower is better."""
    s = normalize_az(az_text)
    n = len(s)
    if n == 0:
        return float("inf")

    counts = Counter(s)
    chi2 = 0.0
    for ch, expected_freq in _ENGLISH_FREQ.items():
        observed = counts.get(ch, 0)
        expected = expected_freq * n
        if expected > 0:
            chi2 += (observed - expected) ** 2 / expected
    return chi2

def _extract_words(text: str) -> list[str]:
    cleaned = normalize_keep_spaces(text).upper()
    return _WORD_RE.findall(cleaned)

def word_hit_rate(text: str) -> float:
    words = _extract_words(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _COMMON_WORDS)
    return hits / len(words)

def word_bonus(text: str) -> float:
    """
    Bounded bonus to push 'almost English' over the line.
    Returns a small positive value for real-word hits, and a small penalty for lots of misses.
    Designed to be added to quadgram_score (which is usually large negative).
    """
    words = _extract_words(text)
    if not words:
        return 0.0

    hits = sum(1 for w in words if w in _COMMON_WORDS)
    misses = len(words) - hits

    # Tuned to be "helpful but not dominant"
    bonus = 2.0 * hits - 0.5 * misses

    # Bound the effect so it can't swamp quadgrams
    if bonus > 20.0:
        bonus = 20.0
    if bonus < -10.0:
        bonus = -10.0
    return bonus

def plaintext_fitness(text: str) -> float:
    """
    The score you should use for ranking/hillclimbing.
    Higher is better.
    """
    # Use A-Z normalization for quadgrams (generally improves stability)
    az = normalize_az(text)
    return quadgram_score(az) + word_bonus(text)

def english_likeness_score(text: str) -> float:
    """
    Returns ~0..100, higher is more English-like.
    Good for confidence (not necessarily for hillclimb).
    """
    if not text:
        return 0.0

    pr = printable_ratio(text)
    if pr < 0.90:
        return 0.0

    cleaned = normalize_keep_spaces(text)
    az = normalize_az(cleaned)

    chi2 = chi_squared_english(az)
    whr = word_hit_rate(text)
    ent = shannon_entropy(cleaned)

    chi_component = 1.0 / (1.0 + (chi2 / 150.0))

    ent_component = 1.0
    if ent > 5.5:
        ent_component = max(0.0, 1.0 - (ent - 5.5) / 2.0)

    # Optional quadgram component if available
    q = quadgram_score(az)
    # Map typical quadgram ranges into 0..1 loosely (heuristic)
    # (More negative = worse; less negative = better)
    quad_component = 0.0
    if q != float("-inf"):
        quad_component = max(0.0, min(1.0, (q + 500.0) / 300.0))

    score_0_1 = (
        0.35 * chi_component +
        0.30 * whr +
        0.20 * ent_component +
        0.15 * quad_component
    )

    return max(0.0, min(100.0, 100.0 * score_0_1 * pr))

def gibberish_probability(text: str) -> float:
    s = english_likeness_score(text)
    if s >= 70:
        return 0.05
    if s >= 50:
        return 0.20
    if s >= 30:
        return 0.45
    if s >= 15:
        return 0.70
    return 0.90
