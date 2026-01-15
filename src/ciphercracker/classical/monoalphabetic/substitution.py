from __future__ import annotations

import math
import random
import re
import time
from importlib import resources
from typing import Optional, Tuple

from ciphercracker.core.features import analyze_text
from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import (
    get_quadgram_scorer,
    quadgram_score,
    english_likeness_score,
    word_bonus as core_word_bonus,
)
from ciphercracker.core.utils import normalize_az
from ciphercracker.classical.common import parse_substitution_key, ALPHABET

_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
_WORD_RE = re.compile(r"[A-Z]+")

# ----------------------------
# 20k dictionary (ranked list)
# ----------------------------

_DICT_READY = False
_WORD_SCORE: dict[str, float] = {}
_MAX_WORD_LEN = 20


def _ensure_dictionary_loaded() -> None:
    """
    Load ciphercracker.data/common_words_20k.txt (1 word per line, most common first).
    Build a rank-based reward:
        score(word) = log((N+1)/(rank+1))
    """
    global _DICT_READY, _WORD_SCORE
    if _DICT_READY:
        return

    words: list[str] = []
    try:
        txt = resources.files("ciphercracker.data").joinpath("common_words_20k.txt").read_text(encoding="utf-8")
        for ln in txt.splitlines():
            w = ln.strip().upper()
            if not w or w.startswith("#"):
                continue
            if w.isalpha():
                words.append(w)
    except Exception:
        # Minimal fallback so solver still runs if packaging is wrong
        words = ["THE", "AND", "TO", "OF", "IN", "IS", "YOU", "THAT", "FOR", "WITH", "THIS", "RIGHT", "WRONG"]

    n = max(1, len(words))
    for rank, w in enumerate(words):
        _WORD_SCORE[w] = math.log((n + 1.0) / (rank + 1.0))

    _DICT_READY = True


def _extract_letter_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.upper())


def _wordbreak_score_token(token: str) -> float:
    """
    Word-break DP score for one token.
    Used ONLY as a tie-breaker near the end, never as the main objective.
    """
    _ensure_dictionary_loaded()

    L = len(token)
    if L == 0:
        return 0.0

    # Tuned to avoid making "random gibberish segmentation" attractive
    unk_char_pen = -2.8
    boundary_pen = -1.1

    dp = [-1e18] * (L + 1)
    dp[0] = 0.0

    for i in range(1, L + 1):
        # consume one unknown character
        dp[i] = dp[i - 1] + unk_char_pen

        j0 = max(0, i - _MAX_WORD_LEN)
        for j in range(j0, i):
            w = token[j:i]
            s = _WORD_SCORE.get(w)
            if s is None:
                continue
            cand = dp[j] + s + boundary_pen
            if cand > dp[i]:
                dp[i] = cand

    return dp[L]


def _wordbreak_score_text(fulltext: str) -> float:
    """
    Aggregate wordbreak score across tokens.
    Tie-break only. We clamp it so it can't explode.
    """
    tokens = _extract_letter_tokens(fulltext)
    if not tokens:
        return 0.0

    raw = 0.0
    for t in tokens:
        if 4 <= len(t) <= 28:
            raw += _wordbreak_score_token(t)

    # Clamp tie-break magnitude hard
    if raw > 300.0:
        raw = 300.0
    if raw < -300.0:
        raw = -300.0
    return raw


# ----------------------------
# Core substitution helpers
# ----------------------------

def _should_try(text: str) -> bool:
    feats = analyze_text(text)
    return feats["alpha_ratio"] >= 0.80 and feats["length"] >= 60


def _mapping_to_keystring(mapping: list[str]) -> str:
    return "".join(mapping)


def _initial_mapping_freq(ciphertext: str) -> list[str]:
    counts = {c: 0 for c in ALPHABET}
    for ch in ciphertext.upper():
        if ch in counts:
            counts[ch] += 1
    cipher_order = "".join(sorted(ALPHABET, key=lambda c: counts[c], reverse=True))

    mapping = ["?"] * 26
    for i, ciph in enumerate(cipher_order):
        mapping[ord(ciph) - 65] = _ENGLISH_FREQ_ORDER[i]
    return mapping


def _random_mapping(rng: random.Random) -> list[str]:
    letters = list(ALPHABET)
    rng.shuffle(letters)
    return letters


def _letters_meta(text: str) -> tuple[list[tuple[int, int]], list[int], list[bool]]:
    letters_meta: list[tuple[int, int]] = []
    positions: list[int] = []
    cases: list[bool] = []

    li = 0
    for pos, ch in enumerate(text):
        up = ch.upper()
        if "A" <= up <= "Z":
            letters_meta.append((li, ord(up) - 65))
            positions.append(pos)
            cases.append(ch.isupper())
            li += 1
    return letters_meta, positions, cases


def _build_occ(letters_meta: list[tuple[int, int]]) -> list[list[int]]:
    occ: list[list[int]] = [[] for _ in range(26)]
    for idx, (_, cidx) in enumerate(letters_meta):
        occ[cidx].append(idx)
    return occ


def _decrypt_letters_only(letters_meta: list[tuple[int, int]], mapping: list[str]) -> list[str]:
    return [mapping[cidx] for (_, cidx) in letters_meta]


def _apply_to_fulltext(
    ciphertext: str,
    positions: list[int],
    cases: list[bool],
    mapping: list[str],
    letters_meta: list[tuple[int, int]],
) -> list[str]:
    out = list(ciphertext)
    for (_, cidx), pos, was_upper in zip(letters_meta, positions, cases):
        p = mapping[cidx]
        out[pos] = p if was_upper else p.lower()
    return out


# ----------------------------
# Objective: base + tie-break
# ----------------------------

def _objective_components(
    ciphertext: str,
    letters_meta,
    positions,
    cases,
    mapping: list[str],
    *,
    use_dict_tiebreak: bool,
) -> Tuple[float, float]:
    """
    Returns (base, tie):
      base: MUST dominate. (quadgrams + core word bonus)
      tie : dictionary wordbreak, ONLY to break ties between near-equal base scores.

    We compare lexicographically: higher base wins; if base ties (within eps),
    higher tie wins.
    """
    pt_full = "".join(_apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta))

    if get_quadgram_scorer() is not None:
        base = quadgram_score(normalize_az(pt_full)) + core_word_bonus(pt_full)
    else:
        # fallback (rare for you)
        base = english_likeness_score(pt_full) * 10.0

    tie = 0.0
    if use_dict_tiebreak:
        tie = _wordbreak_score_text(pt_full)

    return base, tie


def _better(a: Tuple[float, float], b: Tuple[float, float], *, eps: float = 1e-6) -> bool:
    """
    True if a is better than b under lexicographic compare with epsilon on base.
    """
    if a[0] > b[0] + eps:
        return True
    if b[0] > a[0] + eps:
        return False
    return a[1] > b[1] + eps


# ----------------------------
# k-opt polish (swap + 3-cycles) with SAFE restore
# ----------------------------

def _kopt_polish(
    ciphertext: str,
    letters_meta,
    positions,
    cases,
    base_map: list[str],
    *,
    time_up_fn,
    max_moves: int,
    scan_3cycles: bool = True,
    use_dict_tiebreak: bool = True,
) -> list[str]:
    """
    Steepest-ascent polish using:
      - best improving SWAP
      - best improving 3-CYCLE (both directions)
    Safe restore: always put mapping back exactly after evaluation.
    """
    m = base_map[:]
    cur = _objective_components(ciphertext, letters_meta, positions, cases, m, use_dict_tiebreak=use_dict_tiebreak)

    moves = 0
    while moves < max_moves and not time_up_fn():
        best_val = cur
        best_move = None  # ("swap", i, j) or ("cyc", i, j, k, dir)

        # swaps
        for i in range(26):
            if time_up_fn():
                break
            for j in range(i + 1, 26):
                m[i], m[j] = m[j], m[i]
                v = _objective_components(ciphertext, letters_meta, positions, cases, m, use_dict_tiebreak=use_dict_tiebreak)
                m[i], m[j] = m[j], m[i]

                if _better(v, best_val):
                    best_val = v
                    best_move = ("swap", i, j)

        # 3-cycles
        if scan_3cycles and not time_up_fn():
            for i in range(24):
                if time_up_fn():
                    break
                for j in range(i + 1, 25):
                    for k in range(j + 1, 26):
                        a, b, c = m[i], m[j], m[k]

                        # dir 0: i<-j, j<-k, k<-i
                        m[i], m[j], m[k] = b, c, a
                        v0 = _objective_components(ciphertext, letters_meta, positions, cases, m, use_dict_tiebreak=use_dict_tiebreak)
                        m[i], m[j], m[k] = a, b, c  # restore

                        if _better(v0, best_val):
                            best_val = v0
                            best_move = ("cyc", i, j, k, 0)

                        # dir 1: i<-k, k<-j, j<-i
                        m[i], m[j], m[k] = c, a, b
                        v1 = _objective_components(ciphertext, letters_meta, positions, cases, m, use_dict_tiebreak=use_dict_tiebreak)
                        m[i], m[j], m[k] = a, b, c  # restore

                        if _better(v1, best_val):
                            best_val = v1
                            best_move = ("cyc", i, j, k, 1)

        if best_move is None or not _better(best_val, cur):
            break

        # apply best move
        if best_move[0] == "swap":
            _, i, j = best_move
            m[i], m[j] = m[j], m[i]
        else:
            _, i, j, k, d = best_move
            a, b, c = m[i], m[j], m[k]
            if d == 0:
                m[i], m[j], m[k] = b, c, a
            else:
                m[i], m[j], m[k] = c, a, b

        cur = best_val
        moves += 1

    return m


# ----------------------------
# Cracker
# ----------------------------

def crack_substitution_anneal(
    ciphertext: str,
    *,
    restarts: int = 24,
    steps: int = 21000,
    temp_start: float = 18.0,
    temp_end: float = 0.25,
    seed: int | None = None,
    max_seconds: float | None = 30.0,
    full_score_every: int = 260,
) -> list[tuple[float, list[str], str]]:
    """
    General-purpose monoalphabetic substitution solver.

    Acceptance: fast quadgrams on letters-only plaintext.
    Best tracking: base objective = quadgrams(fulltext normalized) + core word bonus.
    Final polish: swap + 3-cycle steepest-ascent, with dictionary used ONLY as tie-break.
    """
    start = time.perf_counter()
    rng = random.Random(seed)

    letters_meta, positions, cases = _letters_meta(ciphertext)
    n_letters = len(letters_meta)
    if n_letters < 60:
        return []

    occ = _build_occ(letters_meta)

    def time_up() -> bool:
        return max_seconds is not None and (time.perf_counter() - start) >= max_seconds

    use_quad = get_quadgram_scorer() is not None

    def fast_score(pt_letters: list[str]) -> float:
        if use_quad:
            return quadgram_score("".join(pt_letters))
        return english_likeness_score("".join(pt_letters)) * 10.0

    def base_score_from_map(m: list[str]) -> float:
        pt_full = "".join(_apply_to_fulltext(ciphertext, positions, cases, m, letters_meta))
        if use_quad:
            return quadgram_score(normalize_az(pt_full)) + core_word_bonus(pt_full)
        return english_likeness_score(pt_full) * 10.0

    best_across: list[tuple[float, list[str], str]] = []

    for r in range(restarts):
        if time_up():
            break

        # init
        if r < 7:
            mapping = _initial_mapping_freq(ciphertext)
            swaps = 12 + 16 * r
            for _ in range(swaps):
                i, j = rng.randrange(26), rng.randrange(26)
                if i != j:
                    mapping[i], mapping[j] = mapping[j], mapping[i]
        else:
            mapping = _random_mapping(rng)

        pt_letters = _decrypt_letters_only(letters_meta, mapping)
        cur_fast = fast_score(pt_letters)

        best_map = mapping[:]
        best_base = base_score_from_map(best_map)

        for step in range(steps):
            if time_up():
                break

            t = temp_start * ((temp_end / temp_start) ** (step / max(1, steps - 1)))

            i, j = rng.randrange(26), rng.randrange(26)
            if i == j:
                continue

            mapping[i], mapping[j] = mapping[j], mapping[i]

            for idx in occ[i]:
                pt_letters[idx] = mapping[i]
            for idx in occ[j]:
                pt_letters[idx] = mapping[j]

            new_fast = fast_score(pt_letters)

            accept = False
            if new_fast >= cur_fast:
                accept = True
            elif t > 0:
                accept = (rng.random() < math.exp((new_fast - cur_fast) / t))

            if accept:
                cur_fast = new_fast
                if (step % full_score_every) == 0:
                    b = base_score_from_map(mapping)
                    if b > best_base:
                        best_base = b
                        best_map = mapping[:]
            else:
                # revert
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

        # end-of-restart check
        b = base_score_from_map(mapping)
        if b > best_base:
            best_base = b
            best_map = mapping[:]

        # polish: only if quadgrams available
        if use_quad and not time_up():
            # short texts: allow a few moves; still fast at 162 chars
            max_moves = 10 if n_letters < 240 else 7

            # Use dict tie-break only in polish, never in anneal
            best_map = _kopt_polish(
                ciphertext,
                letters_meta,
                positions,
                cases,
                best_map,
                time_up_fn=time_up,
                max_moves=max_moves,
                scan_3cycles=True,
                use_dict_tiebreak=True,
            )
            best_base = base_score_from_map(best_map)

        final_full = "".join(_apply_to_fulltext(ciphertext, positions, cases, best_map, letters_meta))
        best_across.append((best_base, best_map[:], final_full))

    best_across.sort(key=lambda x: x[0], reverse=True)
    return best_across[:5]


class SubstitutionCipher:
    name = "substitution"

    def decrypt(self, ciphertext: str, key: str) -> str:
        mapping_dict = parse_substitution_key(key)
        mapping = [mapping_dict[c] for c in ALPHABET]

        out = []
        for ch in ciphertext:
            if ch.isalpha():
                up = ch.upper()
                if "A" <= up <= "Z":
                    plain = mapping[ord(up) - 65]
                    out.append(plain if ch.isupper() else plain.lower())
                else:
                    out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        az_len = len(normalize_az(ciphertext))
        max_seconds = 28.0 if az_len < 220 else 42.0

        found = crack_substitution_anneal(
            ciphertext,
            restarts=24 if az_len < 260 else 30,
            steps=21000 if az_len < 260 else 26000,
            max_seconds=max_seconds,
            full_score_every=260 if az_len < 260 else 320,
        )

        results: list[SolveResult] = []
        for s, mapping, pt in found:
            results.append(
                SolveResult(
                    cipher_name=self.name,
                    plaintext=pt,
                    key=_mapping_to_keystring(mapping),
                    score=float(s),
                    confidence=0.0,  # registry recomputes confidence
                    notes="Simulated annealing mono-sub (quadgram-driven + safe k-opt polish (swap+3cycle) + dict tie-break only)",
                    meta={"fitness": "quadgram" if get_quadgram_scorer() else "fallback"},
                )
            )
        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(SubstitutionCipher(), should_try=_should_try)
