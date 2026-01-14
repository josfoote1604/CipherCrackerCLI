from __future__ import annotations

import base64
import binascii
import string
from collections import Counter
from .utils import normalize_az

from .results import TextFeatures
from .utils import (
    index_of_coincidence_az,
    normalize_keep_spaces,
    printable_ratio,
    shannon_entropy,
)

_BASE64_CHARS = set(string.ascii_letters + string.digits + "+/=")
_HEX_CHARS = set("0123456789abcdefABCDEF")
_BIN_CHARS = set("01")


def _guess_charset(s: str) -> str:
    if not s:
        return "empty"

    stripped = s.strip()

    # Binary-ish
    if stripped and set(stripped) <= _BIN_CHARS and len(stripped) >= 16:
        return "binary-ish"

    # Hex-ish
    hex_str = stripped.replace(" ", "").replace("\n", "")
    if hex_str and set(hex_str) <= _HEX_CHARS and len(hex_str) % 2 == 0 and len(hex_str) >= 8:
        return "hex-ish"

    # Base64-ish
    b64_str = stripped.replace("\n", "")
    if b64_str and set(b64_str) <= _BASE64_CHARS and len(b64_str) % 4 == 0 and len(b64_str) >= 8:
        return "base64-ish"

    # A-Z only
    upper = normalize_keep_spaces(s).replace(" ", "")
    if upper and all("A" <= ch <= "Z" for ch in upper):
        return "A-Z"

    # A-Z0-9 only
    if upper and all(("A" <= ch <= "Z") or ("0" <= ch <= "9") for ch in upper):
        return "A-Z0-9"

    return "mixed"


def analyze_text(text: str) -> dict:
    """
    Returns a dict of features used for:
      - cipher identification heuristics
      - gibberish vs plausible plaintext detection
    """
    cleaned = normalize_keep_spaces(text)
    n = len(cleaned)
    if n == 0:
        feats = TextFeatures(
            length=0,
            unique_chars=0,
            charset="empty",
            alpha_ratio=0.0,
            digit_ratio=0.0,
            space_ratio=0.0,
            printable_ratio=0.0,
            entropy=0.0,
            ioc=0.0,
        )
        return feats.to_dict()

    counts = Counter(cleaned)
    alpha = sum(c for ch, c in counts.items() if "A" <= ch <= "Z")
    digit = sum(c for ch, c in counts.items() if "0" <= ch <= "9")
    space = sum(c for ch, c in counts.items() if ch.isspace())

    feats = TextFeatures(
        length=n,
        unique_chars=len(counts),
        charset=_guess_charset(text),
        alpha_ratio=alpha / n,
        digit_ratio=digit / n,
        space_ratio=space / n,
        printable_ratio=printable_ratio(text),
        entropy=shannon_entropy(cleaned),
        ioc=index_of_coincidence_az(cleaned),
    )
    return feats.to_dict()


def try_decode_hex(text: str) -> str | None:
    s = text.strip().replace(" ", "").replace("\n", "")
    if not s or any(ch not in _HEX_CHARS for ch in s) or len(s) % 2 != 0:
        return None
    try:
        raw = bytes.fromhex(s)
        return raw.decode("utf-8", errors="replace")
    except ValueError:
        return None


def try_decode_base64(text: str) -> str | None:
    s = text.strip().replace("\n", "")
    if not s or any(ch not in _BASE64_CHARS for ch in s) or len(s) % 4 != 0:
        return None
    try:
        raw = base64.b64decode(s, validate=True)
        return raw.decode("utf-8", errors="replace")
    except (binascii.Error, ValueError):
        return None

def ioc_scan(text: str, max_len: int = 20) -> list[tuple[int, float]]:
    az = normalize_az(text)
    if len(az) < 2:
        return []

    def ioc(s: str) -> float:
        n = len(s)
        if n < 2:
            return 0.0
        c = Counter(s)
        return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))

    scores = []
    for k in range(1, max_len + 1):
        cols = [az[i::k] for i in range(k)]
        avg = sum(ioc(col) for col in cols) / k
        scores.append((k, avg))

    return sorted(scores, key=lambda x: x[1], reverse=True)