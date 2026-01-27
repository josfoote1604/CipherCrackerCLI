from __future__ import annotations

import math
import random
import re
import time
from importlib import resources
from typing import Tuple

from ciphercracker.core.features import analyze_text
from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import (
    get_quadgram_scorer,
    quadgram_score,
    english_likeness_score,
    word_bonus,
)
from ciphercracker.core.utils import normalize_az
from ciphercracker.classical.common import parse_substitution_key, ALPHABET

_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
_WORD_RE = re.compile(r"[A-Z]+")

# Only consult tie-break inside a "near" band of primary scores (quadgrams are noisy on short text)
_PRIMARY_NEAR = 20.0  # tune: start 15–30 depending on your quadgram scale/logs


def _sort_near_band(
    items: list[tuple[float, float, list[str], str]],
    *,
    near: float,
) -> list[tuple[float, float, list[str], str]]:
    """
    Sort primarily by primary score descending.
    Only use tie-break ordering within blocks whose primary scores are within +/- near of the block leader.
    """
    items = sorted(items, key=lambda x: x[0], reverse=True)
    out: list[tuple[float, float, list[str], str]] = []

    i = 0
    while i < len(items):
        base = items[i][0]
        j = i + 1
        while j < len(items) and abs(items[j][0] - base) <= near:
            j += 1
        block = items[i:j]
        block.sort(key=lambda x: x[1], reverse=True)  # tie only inside near band
        out.extend(block)
        i = j

    return out


# ----------------------------
# Optional 20k dictionary tie-break
# (NEVER used as primary objective)
# ----------------------------

_DICT_READY = False
_WORD_SCORE: dict[str, float] = {}
_MAX_WORD_LEN = 20


def _ensure_dictionary_loaded() -> None:
    global _DICT_READY, _WORD_SCORE
    if _DICT_READY:
        return

    words: list[str] = []
    try:
        txt = (
            resources.files("ciphercracker.data")
            .joinpath("common_words_20k.txt")
            .read_text(encoding="utf-8")
        )
        for ln in txt.splitlines():
            w = ln.strip().upper()
            if not w or w.startswith("#"):
                continue
            if w.isalpha():
                words.append(w)
    except Exception:
        # Minimal fallback if file missing; still safe since it's tie-break only
        words = ["THE", "AND", "TO", "OF", "IN", "IS", "YOU", "THAT", "FOR", "WITH", "THIS"]

    n = max(1, len(words))
    for rank, w in enumerate(words):
        _WORD_SCORE[w] = math.log((n + 1.0) / (rank + 1.0))

    _DICT_READY = True


def _extract_letter_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.upper())


def _wordbreak_score_token(token: str) -> float:
    """
    Word-break DP score for one token.
    Only used as tie-break for near-equal quadgram solutions.
    """
    _ensure_dictionary_loaded()
    L = len(token)
    if L == 0:
        return 0.0

    unk_char_pen = -2.8
    boundary_pen = -1.1

    dp = [-1e18] * (L + 1)
    dp[0] = 0.0

    for i in range(1, L + 1):
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
    Hard-clamped (tie-break only).
    """
    tokens = _extract_letter_tokens(fulltext)
    if not tokens:
        return 0.0

    raw = 0.0
    for t in tokens:
        if 4 <= len(t) <= 28:
            raw += _wordbreak_score_token(t)

    raw = max(-250.0, min(250.0, raw))
    return raw


# ----------------------------
# Core helpers
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
) -> str:
    out = list(ciphertext)
    for (_, cidx), pos, was_upper in zip(letters_meta, positions, cases):
        p = mapping[cidx]
        out[pos] = p if was_upper else p.lower()
    return "".join(out)


# ----------------------------
# Scoring
# ----------------------------

def _score_letters_only(pt_letters: list[str]) -> float:
    """
    Primary objective for mono-sub: quadgrams on letters-only plaintext.
    """
    if get_quadgram_scorer() is not None:
        return quadgram_score("".join(pt_letters))
    return english_likeness_score("".join(pt_letters)) * 10.0


def _tie_break(ciphertext: str, letters_meta, positions, cases, mapping: list[str]) -> float:
    """
    Tie-break only (dictionary wordbreak on spaced plaintext).
    """
    pt = _apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta)
    return _wordbreak_score_text(pt)


def _better_pair(a: Tuple[float, float], b: Tuple[float, float], eps: float = 1e-6) -> bool:
    """
    Compare (primary, tie). Higher is better.
    NOTE: Sorting uses near-band blocks; this is still used in polish search.
    """
    if a[0] > b[0] + eps:
        return True
    if b[0] > a[0] + eps:
        return False
    return a[1] > b[1] + eps


# ----------------------------
# Targeted polish helpers
# ----------------------------

def _top_cipher_letters_by_freq(occ: list[list[int]], top_k: int) -> list[int]:
    order = sorted(range(26), key=lambda c: len(occ[c]), reverse=True)
    return order[:top_k]


# ----------------------------
# Final short-text refinement (GENERIC, no hardcoding)
# ----------------------------

def _shorttext_final_refine(
    ciphertext: str,
    letters_meta,
    positions,
    cases,
    pt_letters: list[str],
    mapping: list[str],
    occ: list[list[int]],
    *,
    time_up_fn,
    top_k_letters: int = 12,
    max_moves: int = 18,
    word_w: float = 0.8,
    wb_w: float = 0.01,
    allow_primary_drop: float = 25.0,
) -> tuple[float, list[str], list[str]]:
    """
    Generic final refinement for short texts:
      - explore swaps among the most frequent cipher letters
      - choose move that maximizes:
            eval = primary_quad + word_w * word_bonus(fulltext) + wb_w * wordbreak(fulltext)
      - keep reporting primary (quadgrams) as the solver score

    This is exactly what fixes 'REAP/CRACZER/POING' style near-misses without any fixed key.
    """
    if get_quadgram_scorer() is None:
        return _score_letters_only(pt_letters), mapping, pt_letters

    cand = _top_cipher_letters_by_freq(occ, top_k_letters)

    def eval_state(primary: float) -> float:
        pt_full = _apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta)
        wb = _wordbreak_score_text(pt_full)
        return primary + word_w * word_bonus(pt_full) + wb_w * wb

    cur_primary = _score_letters_only(pt_letters)
    cur_eval = eval_state(cur_primary)

    moves = 0
    while moves < max_moves and not time_up_fn():
        best_eval = cur_eval
        best_primary = cur_primary
        best_swap = None

        # try all swaps in candidate set
        for ai in range(len(cand)):
            if time_up_fn():
                break
            i = cand[ai]
            for aj in range(ai + 1, len(cand)):
                j = cand[aj]

                # apply swap
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

                p = _score_letters_only(pt_letters)

                # avoid spending time on clearly-worse basins
                if p >= cur_primary - allow_primary_drop:
                    e = eval_state(p)
                    if e > best_eval + 1e-9:
                        best_eval = e
                        best_primary = p
                        best_swap = (i, j)

                # revert
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

        if best_swap is None:
            break

        # commit best swap
        i, j = best_swap
        mapping[i], mapping[j] = mapping[j], mapping[i]
        for idx in occ[i]:
            pt_letters[idx] = mapping[i]
        for idx in occ[j]:
            pt_letters[idx] = mapping[j]

        cur_eval = best_eval
        cur_primary = best_primary
        moves += 1

    return cur_primary, mapping, pt_letters


# ----------------------------
# Greedy swap polish
# ----------------------------

def _greedy_swap_polish(
    pt_letters: list[str],
    mapping: list[str],
    occ: list[list[int]],
    *,
    time_up_fn,
    max_passes: int = 4,
) -> Tuple[float, list[str], list[str]]:
    """
    Cheap steepest-ascent SWAP polish using primary score only.
    """
    cur_s = _score_letters_only(pt_letters)
    improved = True
    passes = 0

    while improved and passes < max_passes and not time_up_fn():
        improved = False
        passes += 1

        best_pair = None
        best_s = cur_s

        for i in range(26):
            if time_up_fn():
                break
            for j in range(i + 1, 26):
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

                s = _score_letters_only(pt_letters)

                # revert
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

                if s > best_s:
                    best_s = s
                    best_pair = (i, j)

        if best_pair is not None and best_s > cur_s:
            i, j = best_pair
            mapping[i], mapping[j] = mapping[j], mapping[i]
            for idx in occ[i]:
                pt_letters[idx] = mapping[i]
            for idx in occ[j]:
                pt_letters[idx] = mapping[j]
            cur_s = best_s
            improved = True

    return cur_s, mapping, pt_letters


# ----------------------------
# Targeted k-opt polish (swap + 3-cycles)
# ----------------------------

def _targeted_kopt_polish(
    ciphertext: str,
    letters_meta,
    positions,
    cases,
    pt_letters: list[str],
    mapping: list[str],
    occ: list[list[int]],
    *,
    time_up_fn,
    top_k_letters: int = 10,
    max_moves: int = 8,
) -> Tuple[float, list[str], list[str]]:
    cand = _top_cipher_letters_by_freq(occ, top_k_letters)

    def cur_pair() -> Tuple[float, float]:
        return (_score_letters_only(pt_letters), _tie_break(ciphertext, letters_meta, positions, cases, mapping))

    cur = cur_pair()
    moves = 0

    while moves < max_moves and not time_up_fn():
        best = cur
        best_move = None

        for ai in range(len(cand)):
            if time_up_fn():
                break
            i = cand[ai]
            for aj in range(ai + 1, len(cand)):
                j = cand[aj]

                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

                v = cur_pair()

                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

                if _better_pair(v, best):
                    best = v
                    best_move = ("swap", i, j)

        if not time_up_fn():
            L = len(cand)
            top = min(L, 10)
            for a in range(top):
                if time_up_fn():
                    break
                for b in range(a + 1, top):
                    for c in range(b + 1, top):
                        i, j, k = cand[a], cand[b], cand[c]
                        A, B, C = mapping[i], mapping[j], mapping[k]

                        mapping[i], mapping[j], mapping[k] = B, C, A
                        for idx in occ[i]:
                            pt_letters[idx] = mapping[i]
                        for idx in occ[j]:
                            pt_letters[idx] = mapping[j]
                        for idx in occ[k]:
                            pt_letters[idx] = mapping[k]
                        v0 = cur_pair()

                        mapping[i], mapping[j], mapping[k] = A, B, C
                        for idx in occ[i]:
                            pt_letters[idx] = mapping[i]
                        for idx in occ[j]:
                            pt_letters[idx] = mapping[j]
                        for idx in occ[k]:
                            pt_letters[idx] = mapping[k]

                        if _better_pair(v0, best):
                            best = v0
                            best_move = ("cyc", i, j, k, 0)

                        mapping[i], mapping[j], mapping[k] = C, A, B
                        for idx in occ[i]:
                            pt_letters[idx] = mapping[i]
                        for idx in occ[j]:
                            pt_letters[idx] = mapping[j]
                        for idx in occ[k]:
                            pt_letters[idx] = mapping[k]
                        v1 = cur_pair()

                        mapping[i], mapping[j], mapping[k] = A, B, C
                        for idx in occ[i]:
                            pt_letters[idx] = mapping[i]
                        for idx in occ[j]:
                            pt_letters[idx] = mapping[j]
                        for idx in occ[k]:
                            pt_letters[idx] = mapping[k]

                        if _better_pair(v1, best):
                            best = v1
                            best_move = ("cyc", i, j, k, 1)

        if best_move is None or not _better_pair(best, cur):
            break

        if best_move[0] == "swap":
            _, i, j = best_move
            mapping[i], mapping[j] = mapping[j], mapping[i]
            for idx in occ[i]:
                pt_letters[idx] = mapping[i]
            for idx in occ[j]:
                pt_letters[idx] = mapping[j]
        else:
            _, i, j, k, d = best_move
            A, B, C = mapping[i], mapping[j], mapping[k]
            if d == 0:
                mapping[i], mapping[j], mapping[k] = B, C, A
            else:
                mapping[i], mapping[j], mapping[k] = C, A, B
            for idx in occ[i]:
                pt_letters[idx] = mapping[i]
            for idx in occ[j]:
                pt_letters[idx] = mapping[j]
            for idx in occ[k]:
                pt_letters[idx] = mapping[k]

        cur = best
        moves += 1

    return cur[0], mapping, pt_letters


# ----------------------------
# Targeted 4-opt polish (double swap + 4-cycles)
# ----------------------------

def _targeted_4opt_polish(
    ciphertext: str,
    letters_meta,
    positions,
    cases,
    pt_letters: list[str],
    mapping: list[str],
    occ: list[list[int]],
    *,
    time_up_fn,
    top_k_letters: int = 9,
    max_moves: int = 5,
    near_eps: float = 0.35,
) -> tuple[float, list[str], list[str]]:
    cand = _top_cipher_letters_by_freq(occ, top_k_letters)

    def cur_pair() -> tuple[float, float]:
        return (_score_letters_only(pt_letters), _tie_break(ciphertext, letters_meta, positions, cases, mapping))

    def apply_vals(idxs: tuple[int, ...], new_vals: tuple[str, ...]) -> None:
        for cidx, v in zip(idxs, new_vals):
            mapping[cidx] = v
            for pos in occ[cidx]:
                pt_letters[pos] = v

    def best_pair_update(best: tuple[float, float], v_primary: float) -> tuple[float, float]:
        if v_primary > best[0] + 1e-9:
            return (v_primary, best[1])
        if abs(v_primary - best[0]) <= near_eps:
            v_tie = _tie_break(ciphertext, letters_meta, positions, cases, mapping)
            return (v_primary, v_tie)
        return best

    cur = cur_pair()
    moves = 0

    while moves < max_moves and not time_up_fn():
        best = cur
        best_move = None

        m = len(cand)

        for a in range(m):
            if time_up_fn():
                break
            for b in range(a + 1, m):
                i = cand[a]
                j = cand[b]
                Ai, Aj = mapping[i], mapping[j]

                mapping[i], mapping[j] = Aj, Ai
                for pos in occ[i]:
                    pt_letters[pos] = mapping[i]
                for pos in occ[j]:
                    pt_letters[pos] = mapping[j]

                for c in range(m):
                    if time_up_fn():
                        break
                    if c == a or c == b:
                        continue
                    for d in range(c + 1, m):
                        if d == a or d == b:
                            continue
                        k = cand[c]
                        l = cand[d]
                        Ak, Al = mapping[k], mapping[l]

                        mapping[k], mapping[l] = Al, Ak
                        for pos in occ[k]:
                            pt_letters[pos] = mapping[k]
                        for pos in occ[l]:
                            pt_letters[pos] = mapping[l]

                        s = _score_letters_only(pt_letters)
                        v = best_pair_update(best, s)
                        if _better_pair(v, best):
                            best = v
                            best_move = ("dswap", i, j, k, l)

                        mapping[k], mapping[l] = Ak, Al
                        for pos in occ[k]:
                            pt_letters[pos] = mapping[k]
                        for pos in occ[l]:
                            pt_letters[pos] = mapping[l]

                mapping[i], mapping[j] = Ai, Aj
                for pos in occ[i]:
                    pt_letters[pos] = mapping[i]
                for pos in occ[j]:
                    pt_letters[pos] = mapping[j]

        cyc_cand = cand[: min(len(cand), 8)]
        m2 = len(cyc_cand)

        for a in range(m2):
            if time_up_fn():
                break
            for b in range(a + 1, m2):
                for c in range(b + 1, m2):
                    for d in range(c + 1, m2):
                        i, j, k, l = cyc_cand[a], cyc_cand[b], cyc_cand[c], cyc_cand[d]
                        A, B, C, D = mapping[i], mapping[j], mapping[k], mapping[l]

                        apply_vals((i, j, k, l), (B, C, D, A))
                        s0 = _score_letters_only(pt_letters)
                        v0 = best_pair_update(best, s0)
                        if _better_pair(v0, best):
                            best = v0
                            best_move = ("cyc", i, j, k, l, 0)
                        apply_vals((i, j, k, l), (A, B, C, D))

                        apply_vals((i, j, k, l), (D, A, B, C))
                        s1 = _score_letters_only(pt_letters)
                        v1 = best_pair_update(best, s1)
                        if _better_pair(v1, best):
                            best = v1
                            best_move = ("cyc", i, j, k, l, 1)
                        apply_vals((i, j, k, l), (A, B, C, D))

        if best_move is None or not _better_pair(best, cur):
            break

        if best_move[0] == "dswap":
            _, i, j, k, l = best_move
            Ai, Aj, Ak, Al = mapping[i], mapping[j], mapping[k], mapping[l]
            apply_vals((i, j, k, l), (Aj, Ai, Al, Ak))
        else:
            _, i, j, k, l, d = best_move
            A, B, C, D = mapping[i], mapping[j], mapping[k], mapping[l]
            if d == 0:
                apply_vals((i, j, k, l), (B, C, D, A))
            else:
                apply_vals((i, j, k, l), (D, A, B, C))

        cur = best
        moves += 1

    return cur[0], mapping, pt_letters


# ----------------------------
# Cracker
# ----------------------------

def crack_substitution_anneal(
    ciphertext: str,
    *,
    restarts: int = 28,
    steps: int = 24000,
    temp_start: float = 18.0,
    temp_end: float = 0.25,
    seed: int | None = None,
    max_seconds: float | None = 35.0,
    elite_k: int = 3,
) -> list[tuple[float, list[str], str]]:
    start = time.perf_counter()
    rng = random.Random(seed)

    letters_meta, positions, cases = _letters_meta(ciphertext)
    n_letters = len(letters_meta)
    if n_letters < 60:
        return []

    occ = _build_occ(letters_meta)
    use_quad = get_quadgram_scorer() is not None

    use_word_guidance = (n_letters < 180 and use_quad)
    word_w = 0.6 if use_word_guidance else 0.0

    def time_up() -> bool:
        return max_seconds is not None and (time.perf_counter() - start) >= max_seconds

    best_across: list[tuple[float, float, list[str], str]] = []

    # Global beam: store (eval_score, primary_score, mapping, pt_letters)
    beam: list[tuple[float, float, list[str], list[str]]] = []
    BEAM_K = 60 if n_letters < 180 else 40

    def beam_push(eval_s: float, primary_s: float, mapping: list[str], pt_letters: list[str]) -> None:
        beam.append((eval_s, primary_s, mapping[:], pt_letters[:]))

    for r in range(restarts):
        if time_up():
            break

        if r < 8:
            mapping = _initial_mapping_freq(ciphertext)
            swaps = 12 + 16 * r
            for _ in range(swaps):
                i, j = rng.randrange(26), rng.randrange(26)
                if i != j:
                    mapping[i], mapping[j] = mapping[j], mapping[i]
        else:
            mapping = _random_mapping(rng)

        pt_letters = _decrypt_letters_only(letters_meta, mapping)
        cur_primary = _score_letters_only(pt_letters)

        if word_w:
            cur_full = _apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta)
            cur_eval = cur_primary + word_w * word_bonus(cur_full)
        else:
            cur_eval = cur_primary

        elites: list[tuple[float, float, list[str], list[str]]] = [(cur_eval, cur_primary, mapping[:], pt_letters[:])]

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

            new_primary = _score_letters_only(pt_letters)
            if word_w:
                new_full = _apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta)
                new_eval = new_primary + word_w * word_bonus(new_full)
            else:
                new_eval = new_primary

            accept = False
            if new_eval >= cur_eval:
                accept = True
            elif t > 0:
                accept = (rng.random() < math.exp((new_eval - cur_eval) / t))

            if accept:
                cur_eval = new_eval
                cur_primary = new_primary

                inserted = False
                for ei, (es, _, _, _) in enumerate(elites):
                    if new_eval > es:
                        elites.insert(ei, (new_eval, new_primary, mapping[:], pt_letters[:]))
                        inserted = True
                        break
                if not inserted and len(elites) < elite_k:
                    elites.append((new_eval, new_primary, mapping[:], pt_letters[:]))
                if len(elites) > elite_k:
                    elites = elites[:elite_k]
            else:
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

        for (es, ps, m, pl) in elites:
            beam_push(es, ps, m, pl)

        best_restart = None  # (primary, mapping, pt_letters)

        for (_es, _ps, m0, pl0) in elites:
            if time_up():
                break

            s1, m1, pl1 = _greedy_swap_polish(pl0[:], m0[:], occ, time_up_fn=time_up, max_passes=3)

            if use_quad and not time_up():
                s2, m2, pl2 = _targeted_kopt_polish(
                    ciphertext, letters_meta, positions, cases,
                    pl1, m1, occ,
                    time_up_fn=time_up,
                    top_k_letters=10 if n_letters < 220 else 12,
                    max_moves=7 if n_letters < 220 else 6,
                )
            else:
                s2, m2, pl2 = s1, m1, pl1

            if use_quad and not time_up():
                s3, m3, pl3 = _targeted_4opt_polish(
                    ciphertext, letters_meta, positions, cases,
                    pl2, m2, occ,
                    time_up_fn=time_up,
                    top_k_letters=9 if n_letters < 220 else 11,
                    max_moves=5 if n_letters < 220 else 4,
                )
            else:
                s3, m3, pl3 = s2, m2, pl2

            # NEW: short-text final refinement (generic)
            if use_quad and n_letters < 220 and not time_up():
                s4, m4, pl4 = _shorttext_final_refine(
                    ciphertext, letters_meta, positions, cases,
                    pl3, m3, occ,
                    time_up_fn=time_up,
                    top_k_letters=12,
                    max_moves=18,
                    word_w=0.8,
                    wb_w=0.01,
                    allow_primary_drop=25.0,
                )
            else:
                s4, m4, pl4 = s3, m3, pl3

            if best_restart is None or s4 > best_restart[0]:
                best_restart = (s4, m4, pl4)

        if best_restart is not None:
            s_best, m_best, _pl_best = best_restart
            pt_full = _apply_to_fulltext(ciphertext, positions, cases, m_best, letters_meta)
            tie = _wordbreak_score_text(pt_full)
            best_across.append((s_best, tie, m_best[:], pt_full))

    # ---- polish global beam too ----
    if beam:
        beam.sort(key=lambda x: x[0], reverse=True)
        beam = beam[:BEAM_K]

        for (_es, _ps, m0, pl0) in beam:
            if time_up():
                break

            s1, m1, pl1 = _greedy_swap_polish(pl0[:], m0[:], occ, time_up_fn=time_up, max_passes=3)

            if use_quad and not time_up():
                s2, m2, pl2 = _targeted_kopt_polish(
                    ciphertext, letters_meta, positions, cases,
                    pl1, m1, occ,
                    time_up_fn=time_up,
                    top_k_letters=10 if n_letters < 220 else 12,
                    max_moves=7 if n_letters < 220 else 6,
                )
            else:
                s2, m2, pl2 = s1, m1, pl1

            if use_quad and not time_up():
                s3, m3, pl3 = _targeted_4opt_polish(
                    ciphertext, letters_meta, positions, cases,
                    pl2, m2, occ,
                    time_up_fn=time_up,
                    top_k_letters=9 if n_letters < 220 else 11,
                    max_moves=5 if n_letters < 220 else 4,
                )
            else:
                s3, m3, pl3 = s2, m2, pl2

            # NEW: short-text final refinement (generic)
            if use_quad and n_letters < 220 and not time_up():
                s4, m4, pl4 = _shorttext_final_refine(
                    ciphertext, letters_meta, positions, cases,
                    pl3, m3, occ,
                    time_up_fn=time_up,
                    top_k_letters=12,
                    max_moves=18,
                    word_w=0.8,
                    wb_w=0.01,
                    allow_primary_drop=25.0,
                )
            else:
                s4, m4, pl4 = s3, m3, pl3

            pt_full = _apply_to_fulltext(ciphertext, positions, cases, m4, letters_meta)
            tie = _wordbreak_score_text(pt_full)
            best_across.append((s4, tie, m4[:], pt_full))

    best_across = _sort_near_band(best_across, near=_PRIMARY_NEAR)
    return [(s, m, pt) for (s, _tie, m, pt) in best_across[:5]]


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

        if az_len < 180:
            restarts = 160
            steps = 8000
            temp_start = 22.0
            temp_end = 0.75
            max_seconds = 40.0
        elif az_len < 260:
            restarts = 28
            steps = 24000
            temp_start = 18.0
            temp_end = 0.25
            max_seconds = 32.0
        else:
            restarts = 34
            steps = 28000
            temp_start = 18.0
            temp_end = 0.25
            max_seconds = 45.0

        found = crack_substitution_anneal(
            ciphertext,
            restarts=restarts,
            steps=steps,
            temp_start=temp_start,
            temp_end=temp_end,
            max_seconds=max_seconds,
            elite_k=3,
        )

        results: list[SolveResult] = []
        short_note = " (short text: low reliability)" if az_len < 180 else ""

        for s, mapping, pt in found:
            results.append(
                SolveResult(
                    cipher_name=self.name,
                    plaintext=pt,
                    key=_mapping_to_keystring(mapping),
                    score=float(s),
                    confidence=0.0,  # registry recomputes confidence
                    notes=(
                        "Simulated annealing mono-sub (quadgram primary + elite states + targeted k-opt polish; "
                        "dict tie-break only)" + short_note
                    ),
                    meta={"fitness": "quadgram" if get_quadgram_scorer() else "fallback"},
                )
            )
        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(SubstitutionCipher(), should_try=_should_try)
