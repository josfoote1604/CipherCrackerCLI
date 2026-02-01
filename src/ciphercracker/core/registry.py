from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Protocol

from .results import SolveResult
from .scoring import (
    english_likeness_score,
    get_quadgram_scorer,
    quadgram_score,
    plaintext_fitness,
    gibberish_probability,
)
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


def _length_confidence_scale(n_letters: int) -> float:
    """
    Downscale confidence for short texts where n-gram scoring is noisy.

    New behavior reaches ~1.0 around ~100 letters, but clamps low
    for very short strings.
    """
    if n_letters <= 0:
        return 0.25

    scale = n_letters / 100.0  # 100 letters -> ~1.0
    if scale < 0.25:
        scale = 0.25
    if scale > 1.0:
        scale = 1.0
    return scale


def confidence_from_score(text: str, score: float) -> float:
    """
    Confidence heuristic:
    - In quadgram mode, use avg log10 prob per quadgram (score / (n-3)).
    - Apply length downscale so short texts don't look overly certain.
    - In fallback mode, map english_likeness_score into 0..1.
    """
    n_letters = len(normalize_az(text))
    length_scale = _length_confidence_scale(n_letters)

    if get_quadgram_scorer() is not None:
        t = normalize_az(text)
        q = max(1, len(t) - 3)
        avg = score / q  # log10 prob per quadgram (negative)

        # Center at -4.8; slope ~5.0
        conf = _sigmoid(5.0 * (avg + 4.8))
        conf = max(0.0, min(1.0, conf))
        conf *= length_scale
        return max(0.0, min(1.0, conf))

    conf = min(1.0, max(0.0, score / 100.0))
    conf *= length_scale
    return max(0.0, min(1.0, conf))


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
    Prefer plugin-provided, comparable scores if present.
    Priority:
      1) meta['report_fitness'] (fulltext plaintext_fitness; best for cross-cipher ranking)
      2) meta['best_full'] (legacy hook; if you used this earlier)
      3) fallback to rank_score(plaintext)
    """
    if r.meta and isinstance(r.meta, dict):
        if "report_fitness" in r.meta:
            try:
                return float(r.meta["report_fitness"])
            except (TypeError, ValueError):
                pass
        if "best_full" in r.meta:
            try:
                return float(r.meta["best_full"])
            except (TypeError, ValueError):
                pass

    return rank_score(r.plaintext)


def crack_unknown(ciphertext: str, *, top_n: int = 10, include: set[str] | None = None) -> list[SolveResult]:
    """
    Ask registered plugins to attempt cracking.
    Rank results by:
      - plugin-provided meta['report_fitness'] when available
      - otherwise rank_score(plaintext)

    If include is None (auto mode), prune obvious noise, but never prune to empty
    if we actually produced candidates.
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
            if include is not None:
                raise
            continue

    scored: list[SolveResult] = []
    for r in results:
        s = _score_from_result(r)

        conf_score = confidence_score(r.plaintext)
        conf = confidence_from_score(r.plaintext, conf_score)

        scored.append(
            SolveResult(
                cipher_name=r.cipher_name,
                plaintext=r.plaintext,
                key=r.key,
                score=s,
                confidence=conf,
                notes=r.notes,
                meta=r.meta,
            )
        )

    scored = _dedupe_by_plaintext(scored)

    # If you added SolveResult(order=True) + sort_index, you can just sort directly:
    scored.sort()

    # --- Optional: relative confidence adjustment (winner gap) ---
    # IMPORTANT: SolveResult is frozen -> use dataclasses.replace(), do NOT mutate in place.
    if scored and get_quadgram_scorer() is not None and len(scored) >= 2:

        def avgq(res: SolveResult) -> float:
            t = normalize_az(res.plaintext or "")
            q = max(1, len(t) - 3)
            return quadgram_score(t) / q

        a0 = avgq(scored[0])
        a1 = avgq(scored[1])

        gap = a0 - a1  # positive means winner better
        if gap > 0.10:
            boost = min(0.20, 0.8 * gap)

            def clamp01(x: float) -> float:
                return max(0.0, min(1.0, x))

            scored[0] = replace(scored[0], confidence=clamp01(scored[0].confidence + boost))
            for i in range(1, len(scored)):
                scored[i] = replace(scored[i], confidence=clamp01(scored[i].confidence * (1.0 - 0.25 * boost)))

    # Auto-mode pruning AFTER confidence is computed, with length-adaptive thresholds
    if include is None and scored:
        n_in = len(normalize_az(ciphertext))

        # For short inputs, confidence is intentionally low; don't throw everything away.
        conf_cut = 0.01 if n_in < 180 else 0.05
        gib_cut = 0.92 if n_in < 180 else 0.85

        pruned = [
            x for x in scored
            if x.confidence >= conf_cut and gibberish_probability(x.plaintext) <= gib_cut
        ]

        # Never prune to empty if we had candidates; just return best-ranked
        if pruned:
            scored = pruned

    return scored[:top_n]
