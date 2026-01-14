from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from .results import SolveResult
from .scoring import english_likeness_score, get_quadgram_scorer, quadgram_score, plaintext_fitness
from ciphercracker.core.utils import normalize_az

def _cipher_preference(cipher_name: str, key: str | None) -> int:
    """
    Lower is better. Prefer more specific ciphers when plaintext is identical.
    Also demote affine when it is effectively caesar or atbash.
    """
    name = (cipher_name or "").lower()

    # Detect affine special cases
    if name == "affine" and key:
        k = key.replace(" ", "")
        if k.startswith("1,"):   # a=1 => Caesar
            return 3
        if k == "25,25":         # Atbash
            return 2

    pref = {
        "atbash": 1,
        "caesar": 2,
        "affine": 4,
        "substitution": 5,
        "periodic_substitution": 6,
        "vigenere": 7,
    }
    return pref.get(name, 50)


def _dedupe_by_plaintext(results: list[SolveResult]) -> list[SolveResult]:
    """
    Deduplicate candidates that decrypt to the same plaintext (A–Z normalized).
    Keep best scoring, then highest confidence, then preferred cipher type.
    """
    best: dict[str, SolveResult] = {}

    for r in results:
        fp = normalize_az(r.plaintext or "")
        if not fp:
            continue

        cur = best.get(fp)
        if cur is None:
            best[fp] = r
            continue

        if r.score > cur.score:
            best[fp] = r
            continue
        if r.score < cur.score:
            continue

        if r.confidence > cur.confidence:
            best[fp] = r
            continue
        if r.confidence < cur.confidence:
            continue

        if _cipher_preference(r.cipher_name, r.key) < _cipher_preference(cur.cipher_name, cur.key):
            best[fp] = r

    return list(best.values())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def rank_score(text: str) -> float:
    """
    Use quadgram+word bonus when available, else fallback.
    """
    if get_quadgram_scorer() is not None:
        return plaintext_fitness(text)
    return english_likeness_score(text)



def confidence_from_score(text: str, score: float) -> float:
    """
    Confidence heuristic:
    - In quadgram mode, use average log10 prob per quadgram (score / (n-3)).
      English plaintext tends to have a noticeably better (less negative) average than gibberish.
    - In fallback mode, keep the old scaling.
    """
    if get_quadgram_scorer() is not None:
        t = normalize_az(text)
        q = max(1, len(t) - 3)
        avg = score / q  # log10 prob per quadgram (negative)

        # Map avg roughly into 0..1. Tune as needed.
        # Typical: English maybe around -3.x to -4.x; gibberish worse (more negative).
        conf = _sigmoid(4.0 * (avg + 4.6))   # tune 4.6 and slope 4.0  # avg=-6 -> 0, avg=-3 -> 1
        return max(0.0, min(1.0, conf))

    return min(1.0, max(0.0, score / 100.0))

def confidence_score(text: str) -> float:
    """
    A score whose meaning matches confidence_from_score():
    - If quadgrams available: pure quadgram score on normalize_az(text)
    - Else: english_likeness_score(text)
    """
    if get_quadgram_scorer() is not None:
        return quadgram_score(normalize_az(text))
    return english_likeness_score(text)

class CipherPlugin(Protocol):
    name: str

    def decrypt(self, ciphertext: str, key: str) -> str:
        ...

    def crack(self, ciphertext: str) -> list[SolveResult]:
        ...

    def fingerprint(self, ciphertext: str) -> dict:
        ...


@dataclass
class _PluginEntry:
    plugin: CipherPlugin
    should_try: Optional[Callable[[str], bool]] = None


_PLUGINS: dict[str, _PluginEntry] = {}


def register_plugin(plugin: CipherPlugin, *, should_try: Optional[Callable[[str], bool]] = None) -> None:
    key = plugin.name.lower().strip()
    if not key:
        raise ValueError("Plugin must have a non-empty name.")
    _PLUGINS[key] = _PluginEntry(plugin=plugin, should_try=should_try)


def list_plugins() -> list[str]:
    return sorted(_PLUGINS.keys())


def decrypt_known(cipher_name: str, ciphertext: str, key: Optional[str]) -> str:
    if key is None:
        raise ValueError("This decrypt operation requires --key.")
    name = cipher_name.lower().strip()
    if name not in _PLUGINS:
        raise ValueError(f"Unknown cipher '{cipher_name}'. Available: {', '.join(list_plugins())}")
    return _PLUGINS[name].plugin.decrypt(ciphertext, key)


def _score_from_result(r: SolveResult) -> float:
    """
    Prefer plugin-provided score if present (e.g., periodic_substitution stores 'best_full').
    Otherwise compute a generic score from plaintext.
    """
    if r.meta and isinstance(r.meta, dict) and "best_full" in r.meta:
        try:
            return float(r.meta["best_full"])
        except (TypeError, ValueError):
            pass
    return rank_score(r.plaintext)


def crack_unknown(ciphertext: str, *, top_n: int = 10, include: set[str] | None = None) -> list[SolveResult]:
    """
    Ask registered plugins to attempt cracking.
    Rank results by:
      - plugin-provided meta['best_full'] when available
      - otherwise rank_score(plaintext)
    """
    results: list[SolveResult] = []

    for name, entry in _PLUGINS.items():
        if include is not None and name not in include:
            continue

        # If the user explicitly requested a cipher (-c), do NOT skip it via should_try
        if include is None and entry.should_try is not None and not entry.should_try(ciphertext):
            continue

        try:
            results.extend(entry.plugin.crack(ciphertext))
        except Exception:
            # If user targeted a plugin, show the real crash instead of silently ignoring it
            if include is not None:
                raise
            continue

    scored: list[SolveResult] = []
    for r in results:
        # Prune ultra-low-confidence noise unless user explicitly targeted ciphers
        if include is None:
            scored = [x for x in scored if x.confidence >= 0.05]

        # use plugin score if provided (ranking/display)
        s = _score_from_result(r)

        # BUT compute confidence from a "raw" score that matches the confidence model
        conf_score = confidence_score(r.plaintext)
        conf = confidence_from_score(r.plaintext, conf_score)

        scored.append(SolveResult(
            cipher_name=r.cipher_name,
            plaintext=r.plaintext,
            key=r.key,
            score=s,
            confidence=conf,
            notes=r.notes,
            meta=r.meta,
        ))

    scored = _dedupe_by_plaintext(scored)
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_n]
