from __future__ import annotations

import math
import string
import re
from collections import Counter
from typing import Iterable


_AZ_ONLY_RE = re.compile(r"[^A-Z]+")

ASCII_PRINTABLE = set(string.printable)


def is_printable(s: str) -> bool:
    return all(ch in ASCII_PRINTABLE for ch in s)


def printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if ch in ASCII_PRINTABLE) / len(s)


def normalize_keep_spaces(s: str) -> str:
    """Uppercase and keep spaces; strip characters that are clearly noise for classical ciphers."""
    # Keep letters, digits, spaces, and common punctuation (useful later for transposition)
    allowed = set(string.ascii_letters + string.digits + " \t\n\r" + ".,;:'\"!?()-[]")
    cleaned = "".join(ch for ch in s if ch in allowed)
    return cleaned.upper()



def normalize_az(s: str) -> str:
    """Keep only A-Z, uppercase. Hardened against odd shadowing bugs."""
    if s is None:
        return ""
    # ensure we are working with a real string
    s = f"{s}".upper()
    # strip non A-Z
    return _AZ_ONLY_RE.sub("", s)


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits/char."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def index_of_coincidence_az(s: str) -> float:
    """IoC for A-Z only; returns 0.0 if too short."""
    s = normalize_az(s)
    n = len(s)
    if n < 2:
        return 0.0
    counts = Counter(s)
    num = sum(c * (c - 1) for c in counts.values())
    den = n * (n - 1)
    return num / den if den else 0.0


def chunked(seq: Iterable, size: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf
