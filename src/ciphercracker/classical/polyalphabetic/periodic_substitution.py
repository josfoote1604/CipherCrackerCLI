from __future__ import annotations

import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Optional

from ciphercracker.core.features import analyze_text, ioc_scan
from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import (
    chi_squared_english,
    english_likeness_score,
    get_common_words,
    get_quadgram_scorer,
    plaintext_fitness,
    quadgram_score,
    word_bonus,
)
from ciphercracker.core.utils import normalize_az

# ============================================================
# Periodic Substitution (poly-periodic mono-sub)
# Focus: periods 2..6 with Kasiski+IoC ranking
# Strategy:
#   1) Period ranking: IoC bump + Kasiski-style repeat-distance factor votes
#   2) Stream initialization: frequency map per stream
#   3) Global refinement: full-alphabet simulated annealing with restarts
#   4) Polish: greedy polish + word-repair + guided swaps
# ============================================================

_PRIMARY_NEAR_BAND = 20.0     # within this band, word_bonus may tie-break
_WORDCHECK_EVERY = 120
_WORDCHECK_PROB = 0.06

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

# Typical English letter frequencies (A..Z). Used for lightweight statistics / sanity.
# (Not used as the main objective.)
_ENGLISH_FREQ = [
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966,
    0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507, 0.01929, 0.00095, 0.05987,
    0.06327, 0.09056, 0.02758, 0.00978, 0.02360, 0.00150, 0.01974, 0.00074,
]


def _reliability_label(*, english_like: float, wb: float, alpha_len: int) -> str:
    # For longer texts, demand a bit more word evidence.
    wb_hi = 18.0 if alpha_len >= 220 else 12.0
    wb_med = 6.0 if alpha_len >= 220 else 4.0

    if english_like >= 85.0 and wb >= wb_hi:
        return "high"
    if english_like >= 65.0 and wb >= wb_med:
        return "medium"
    return "low"


def _reliability_notice(label: str) -> str:
    if label == "high":
        return "Heuristic solve; likely correct, but minor typos are still possible."
    if label == "medium":
        return "Heuristic solve; readable, but may contain a few incorrect letters/typos."
    return "Heuristic solve; output may be partially incorrect - treat as a strong hint, not a guarantee."




# ---------------------------
# Period detection / ranking
# ---------------------------

def _kasiski_factor_votes(
    az: str,
    *,
    min_ngram: int = 3,
    max_ngram: int = 5,
    max_period: int = 6,
    max_hits_per_ngram: int = 12,
) -> dict[int, float]:
    """
    Lightweight Kasiski-ish voting:
      - find repeated n-grams (len 3..5) in letters-only ciphertext
      - take distances between consecutive occurrences
      - vote for factors (2..max_period) of those distances

    Returns: factor->vote_score
    """
    votes: dict[int, float] = defaultdict(float)
    n = len(az)
    if n < 40:
        return {}

    for L in range(min_ngram, max_ngram + 1):
        pos_map: dict[str, list[int]] = defaultdict(list)
        for i in range(0, n - L + 1):
            ng = az[i : i + L]
            pos_map[ng].append(i)

        # only process repeats
        for positions in pos_map.values():
            if len(positions) < 2:
                continue

            # cap work for very common ngrams
            positions = positions[: max_hits_per_ngram]

            # consecutive distances are enough for a stable signal
            for a, b in zip(positions, positions[1:]):
                d = b - a
                if d <= 0:
                    continue
                # factor votes (2..max_period)
                for f in range(2, max_period + 1):
                    if d % f == 0:
                        # mild preference for shorter distances (more informative)
                        votes[f] += 1.0 / max(1.0, math.log(2.0 + d))
    return dict(votes)


def _rank_periods(ct: str, *, max_period: int = 6, top_n: int = 4) -> list[int]:
    """
    Combine signals to rank periods 2..max_period.
      - IoC bump vs k=1
      - Kasiski factor votes
      - Penalize periods that create too-short columns
    """
    az = normalize_az(ct)
    n = len(az)
    if n < 70:
        # still return a safe order
        base = [3, 2, 4, 5, 6]
        return [p for p in base if 2 <= p <= max_period]

    scan = ioc_scan(ct, max_len=max_period)
    scan_map = {k: v for k, v in scan} if scan else {}

    ioc1 = scan_map.get(1, 0.0)
    kas = _kasiski_factor_votes(az, max_period=max_period)

    # base preference order for classical exercises
    base_pref = [3, 2, 4, 5, 6]
    base_rank = {p: i for i, p in enumerate(base_pref)}

    def score_period(p: int) -> float:
        iocp = scan_map.get(p, 0.0)
        bump = iocp - ioc1

        # columns too short => unreliable/overfit
        col_len = n // p
        short_pen = 0.0
        if col_len < 20:
            short_pen += (20 - col_len) * 2.5
        if col_len < 15:
            short_pen += 30.0

        # weight IoC bump strongly, kasiski as secondary
        # scale bump to human-sized numbers
        s = 12000.0 * bump + 18.0 * kas.get(p, 0.0) - short_pen

        # tiny tie-break preference for common periods
        s += 2.0 * (-(base_rank.get(p, 99)))
        return s

    periods = [p for p in range(2, max_period + 1)]
    periods.sort(key=score_period, reverse=True)
    return periods[: max(1, top_n)]


def _should_try(ct: str) -> bool:
    """
    Only try periodic substitution when there is a credible periodicity signal.
    Now uses BOTH:
      - IoC bump among periods 2..6
      - Kasiski vote strength among periods 2..6
    """
    az = normalize_az(ct)
    n = len(az)

    if n < 70:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.70:
        return False

    scan = ioc_scan(ct, max_len=6)
    if not scan:
        return False
    scan_map = {k: v for k, v in scan}
    ioc1 = scan_map.get(1, 0.0)

    # IoC bump check (periods 2..6)
    best_k = None
    best_val = float("-inf")
    min_col_len = 15

    for k in range(2, 7):
        v = scan_map.get(k)
        if v is None:
            continue
        if (n // k) < min_col_len:
            continue
        if v > best_val:
            best_val = v
            best_k = k

    ioc_bump = (best_val - ioc1) if best_k is not None else 0.0

    # Kasiski votes
    kas = _kasiski_factor_votes(az, max_period=6)
    kas_best = max(kas.values()) if kas else 0.0

    # Thresholds tuned for 2..6 focus
    # (allow either signal to justify trying)
    ioc_ok = (best_val >= 0.058) and (ioc_bump >= 0.007)
    kas_ok = kas_best >= 0.55  # “some” repeats voting consistently

    # for very short-ish ciphertexts, be more conservative
    if n < 120:
        ioc_ok = (best_val >= 0.058) and (ioc_bump >= 0.010)
        kas_ok = kas_best >= 0.75

    return bool(ioc_ok or kas_ok)


# ---------------------------
# Text plumbing / maps
# ---------------------------

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


def _apply_to_fulltext(
    ciphertext: str,
    letters_meta: list[tuple[int, int]],
    positions: list[int],
    cases: list[bool],
    maps: list[list[str]],
    period: int,
) -> str:
    out = list(ciphertext)
    for (li, cidx), pos, was_upper in zip(letters_meta, positions, cases):
        a = li % period
        p = maps[a][cidx]
        out[pos] = p if was_upper else p.lower()
    return "".join(out)


def _decrypt_letters_only(letters_meta: list[tuple[int, int]], maps: list[list[str]], period: int) -> str:
    out: list[str] = []
    for li, cidx in letters_meta:
        out.append(maps[li % period][cidx])
    return "".join(out)


def _streams_from_letters_meta(ciphertext: str, letters_meta: list[tuple[int, int]], period: int) -> list[str]:
    streams = [""] * period
    for li, cidx in letters_meta:
        streams[li % period] += chr(65 + cidx)
    return streams


def _init_maps_by_freq_streams(streams: list[str]) -> list[list[str]]:
    """
    Textbook-ish initializer: for each stream, map by frequency order.
    """
    maps: list[list[str]] = []
    for s in streams:
        counts = Counter(s)
        cipher_order = "".join(sorted(ALPHABET, key=lambda c: counts[c], reverse=True))

        mapping = ["?"] * 26
        for i, ciph in enumerate(cipher_order):
            mapping[ord(ciph) - 65] = _ENGLISH_FREQ_ORDER[i]
        maps.append(mapping)

        # fill unused
        used = set(mapping)
        unused = [c for c in ALPHABET if c not in used]
        it = iter(unused)
        for i in range(26):
            if mapping[i] == "?":
                mapping[i] = next(it)
    return maps


def _random_maps(rng: random.Random, period: int) -> list[list[str]]:
    maps: list[list[str]] = []
    for _ in range(period):
        letters = list(ALPHABET)
        rng.shuffle(letters)
        maps.append(letters)
    return maps


# ---------------------------
# Local objectives and polish
# ---------------------------

def _greedy_polish(
    occ: list[list[list[int]]],
    maps: list[list[str]],
    pt_letters: list[str],
    period: int,
    score_letters_fn,
    max_passes: int = 6,
) -> tuple[float, list[list[str]], list[str]]:
    cur_s = score_letters_fn(pt_letters)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for a in range(period):
            for i in range(26):
                for j in range(i + 1, 26):
                    maps[a][i], maps[a][j] = maps[a][j], maps[a][i]

                    for idx in occ[a][i]:
                        pt_letters[idx] = maps[a][i]
                    for idx in occ[a][j]:
                        pt_letters[idx] = maps[a][j]

                    new_s = score_letters_fn(pt_letters)
                    if new_s > cur_s:
                        cur_s = new_s
                        improved = True
                    else:
                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                        for idx in occ[a][i]:
                            pt_letters[idx] = maps[a][i]
                        for idx in occ[a][j]:
                            pt_letters[idx] = maps[a][j]

    return cur_s, maps, pt_letters


def _calibrate_temps(
    rng: random.Random,
    period: int,
    maps: list[list[str]],
    occ: list[list[list[int]]],
    pt_letters: list[str],
    objective_fn,
    samples: int = 220,
    quantile: float = 0.85,
) -> tuple[float, float]:
    base = objective_fn(pt_letters)
    neg_deltas: list[float] = []

    for _ in range(samples):
        a = rng.randrange(period)
        i = rng.randrange(26)
        j = rng.randrange(26)
        if i == j:
            continue

        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        for idx in occ[a][i]:
            pt_letters[idx] = maps[a][i]
        for idx in occ[a][j]:
            pt_letters[idx] = maps[a][j]

        s2 = objective_fn(pt_letters)
        d = s2 - base
        if d < 0:
            neg_deltas.append(d)

        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        for idx in occ[a][i]:
            pt_letters[idx] = maps[a][i]
        for idx in occ[a][j]:
            pt_letters[idx] = maps[a][j]

    if not neg_deltas:
        return 10.0, 0.8

    neg_deltas.sort()
    k = max(0, min(len(neg_deltas) - 1, int(quantile * (len(neg_deltas) - 1))))
    typical = neg_deltas[k]  # negative

    target = 0.35
    t_start = float(typical / math.log(target))
    t_start = max(4.5, min(240.0, t_start))
    t_end = max(0.8, min(16.0, 0.06 * t_start))
    return t_start, t_end


def _stream_local_solve(
    rng: random.Random,
    stream_ct: str,
    init_map: list[str],
    *,
    restarts: int = 2,
    steps: int = 4000,
) -> list[str]:
    """
    Quick mono-sub improvement for ONE stream (initializer).
    Objective: maximize monogram fit (negative chi-squared vs English).
    NOTE: Stream text is every k-th letter, so n-gram/word scores are misleading.
    """
    # precompute occ for fast updates
    occ = [[] for _ in range(26)]
    for idx, ch in enumerate(stream_ct):
        occ[ord(ch) - 65].append(idx)

    def score(pt_list: list[str]) -> float:
        s = "".join(pt_list)
        return -chi_squared_english(s)

    best_map = init_map[:]
    # initial plaintext list
    pt = ["?"] * len(stream_ct)
    for c in range(26):
        for idx in occ[c]:
            pt[idx] = best_map[c]
    best_s = score(pt)

    # SA params (fixed small)
    t0, t1 = 12.0, 0.8

    for r in range(restarts):
        cur_map = best_map[:]
        # shake
        for _ in range(18 + 8 * r):
            i, j = rng.randrange(26), rng.randrange(26)
            if i != j:
                cur_map[i], cur_map[j] = cur_map[j], cur_map[i]

        cur_pt = pt[:]  # will be rebuilt quickly below
        for c in range(26):
            for idx in occ[c]:
                cur_pt[idx] = cur_map[c]
        cur_s = score(cur_pt)

        for step in range(steps):
            t = t0 * ((t1 / t0) ** (step / max(1, steps - 1)))
            i, j = rng.randrange(26), rng.randrange(26)
            if i == j:
                continue

            cur_map[i], cur_map[j] = cur_map[j], cur_map[i]
            for idx in occ[i]:
                cur_pt[idx] = cur_map[i]
            for idx in occ[j]:
                cur_pt[idx] = cur_map[j]

            s2 = score(cur_pt)
            accept = s2 >= cur_s or (t > 0 and rng.random() < math.exp((s2 - cur_s) / t))
            if accept:
                cur_s = s2
                if cur_s > best_s:
                    best_s = cur_s
                    best_map = cur_map[:]
            else:
                # revert
                cur_map[i], cur_map[j] = cur_map[j], cur_map[i]
                for idx in occ[i]:
                    cur_pt[idx] = cur_map[i]
                for idx in occ[j]:
                    cur_pt[idx] = cur_map[j]

    return best_map


# ---------------------------
# Global crack (restructured)
# ---------------------------

def crack_periodic_substitution_global(
    ciphertext: str,
    *,
    period: int = 3,
    restarts: int = 60,
    steps: int = 30000,
    temp_start: float = 12.0,
    temp_end: float = 0.2,
    auto_calibrate_temps: bool = True,
    max_seconds: float = 35.0,
    seed: int | None = None,
    verbose: bool = True,
    enable_word_repair: bool = True,
    word_repair_lam: float = 0.35,
    word_repair_passes: int = 3,
    word_repair_swaps_per_alpha: int = 400,
    # logging controls
    log_level: int = 1,
    log_best_stride_steps: int = 2500,
    log_best_min_delta: float = 0.75,
    log_best_max_per_restart: int = 12,
) -> Optional[tuple[float, list[list[str]], str, float]]:
    """
    Returns:
      (best_objective, best_maps, plaintext_full, report_fitness)
    """
    rng = random.Random(seed)
    use_quad = get_quadgram_scorer() is not None

    if not verbose:
        log_level = 0

    def _log(msg: str, *, level: int = 1) -> None:
        if log_level >= level:
            print(msg, file=sys.stderr)

    letters_meta, positions, cases = _letters_meta(ciphertext)
    min_letters = max(60, period * 25)
    if len(letters_meta) < min_letters:
        return None

    # occ[a][c] -> indices in pt_letters where alphabet a decrypts cipher letter c
    occ: list[list[list[int]]] = [[[] for _ in range(26)] for _ in range(period)]
    for idx, (li, cidx) in enumerate(letters_meta):
        occ[li % period][cidx].append(idx)

    def objective_from_letters(pt_letters_list: list[str]) -> float:
        s = "".join(pt_letters_list)
        return quadgram_score(s) if use_quad else english_likeness_score(s)

    aff_words = get_common_words()
    aff_re = re.compile(r"[A-Z]{4,}")

    def word_affinity(text: str, miss_weight: float = 0.6) -> float:
        words = aff_re.findall(text.upper())
        if not words:
            return 0.0
        total = 0.0
        score = 0.0
        for w in words:
            wlen = min(12.0, float(len(w)))
            total += wlen
            if w in aff_words:
                score += wlen
            else:
                score -= miss_weight * wlen
        return score / total if total > 0 else 0.0

    best_obj = float("-inf")
    best_word = float("-inf")
    best_maps: Optional[list[list[str]]] = None
    best_letters: Optional[list[str]] = None

    def maybe_update_best_with_word_tiebreak(
        cur_obj: float,
        maps_: list[list[str]],
        pt_letters_: list[str],
        step: int,
    ) -> None:
        nonlocal best_obj, best_maps, best_letters, best_word

        if cur_obj > best_obj:
            best_obj = cur_obj
            best_maps = [m[:] for m in maps_]
            best_letters = pt_letters_[:]
            if step % _WORDCHECK_EVERY == 0:
                pt_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                best_word = word_bonus(pt_full)
            return

        if cur_obj < (best_obj - _PRIMARY_NEAR_BAND):
            return

        if step % _WORDCHECK_EVERY != 0 and rng.random() > _WORDCHECK_PROB:
            return

        pt_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        wb = word_bonus(pt_full)
        if wb > best_word:
            best_obj = cur_obj
            best_maps = [m[:] for m in maps_]
            best_letters = pt_letters_[:]
            best_word = wb

    def final_word_repair(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        max_passes: int = 3,
        swaps_per_alpha: int = 400,
        lam: float = 0.35,
        near_band: float = 8.0,
    ) -> tuple[list[list[str]], list[str]]:
        maps_ = [m[:] for m in maps_in]
        pt_letters_ = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters_)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        best_wb = word_bonus(best_full)
        best_mix = best_obj_local + lam * best_wb

        for _pass in range(max_passes):
            improved = False
            for a in range(period):
                for _ in range(swaps_per_alpha):
                    i, j = rng.randrange(26), rng.randrange(26)
                    if i == j:
                        continue

                    maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                    for idx in occ[a][i]:
                        pt_letters_[idx] = maps_[a][i]
                    for idx in occ[a][j]:
                        pt_letters_[idx] = maps_[a][j]

                    obj = objective_from_letters(pt_letters_)
                    if obj < best_obj_local - near_band:
                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]
                        continue

                    full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                    wb = word_bonus(full)
                    mix = obj + lam * wb

                    if mix > best_mix:
                        best_mix = mix
                        best_obj_local = obj
                        improved = True
                    else:
                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]
            if not improved:
                break

        return maps_, pt_letters_

    def final_guided_polish(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        passes: int = 6,
        lam: float = 1.25,
        max_obj_drop: float = 6.0,
        candidate_letters: str = "BDVXZJQK",
    ) -> tuple[list[list[str]], list[str]]:
        maps_ = [m[:] for m in maps_in]
        pt_letters_ = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters_)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        best_wb = word_bonus(best_full)
        best_mix = best_obj_local + lam * best_wb

        cand = [c for c in candidate_letters if c in ALPHABET]

        for _ in range(passes):
            improved = False
            for a in range(period):
                inv = {maps_[a][i]: i for i in range(26)}
                for x_i in range(len(cand)):
                    for y_i in range(x_i + 1, len(cand)):
                        x, y = cand[x_i], cand[y_i]
                        if x not in inv or y not in inv:
                            continue
                        i, j = inv[x], inv[y]

                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]

                        obj = objective_from_letters(pt_letters_)
                        if obj < best_obj_local - max_obj_drop:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
                            continue

                        full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                        wb = word_bonus(full)
                        mix = obj + lam * wb

                        if mix > best_mix:
                            best_mix = mix
                            best_obj_local = obj
                            improved = True
                        else:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
            if not improved:
                break

        return maps_, pt_letters_

    def final_word_polish_all(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        passes: int = 3,
        lam: float = 12.0,
        near_band: float = 25.0,
    ) -> tuple[list[list[str]], list[str]]:
        maps_ = [m[:] for m in maps_in]
        pt_letters_ = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters_)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        best_wb = word_bonus(best_full)
        best_mix = best_obj_local + lam * best_wb

        for _pass in range(passes):
            improved = False
            for a in range(period):
                for i in range(26):
                    for j in range(i + 1, 26):
                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]

                        obj = objective_from_letters(pt_letters_)
                        if obj < best_obj_local - near_band:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
                            continue

                        full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                        wb = word_bonus(full)
                        mix = obj + lam * wb

                        if mix > best_mix:
                            best_mix = mix
                            best_obj_local = obj
                            improved = True
                        else:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
            if not improved:
                break

        return maps_, pt_letters_

    def final_word_kopt(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        top_k_high: int = 10,
        bottom_k_low: int = 8,
        max_moves: int = 6,
        lam_wb: float = 6.0,
        lam_aff: float = 60.0,
        near_band: float = 40.0,
        miss_weight: float = 0.6,
    ) -> tuple[list[list[str]], list[str]]:
        maps_ = [m[:] for m in maps_in]
        pt_letters_ = pt_letters_in[:]

        cur_obj = objective_from_letters(pt_letters_)
        cur_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        cur_wb = word_bonus(cur_full)
        cur_aff = word_affinity(cur_full, miss_weight=miss_weight)
        cur_mix = cur_obj + lam_wb * cur_wb + lam_aff * cur_aff

        def candidate_letters(a: int) -> list[int]:
            counts = [(i, len(occ[a][i])) for i in range(26)]
            counts.sort(key=lambda x: x[1], reverse=True)
            top = [i for i, _ in counts[:top_k_high]]
            bottom = [i for i, _ in counts[-bottom_k_low:]] if bottom_k_low > 0 else []
            return list(dict.fromkeys(top + bottom))

        def apply_map(a: int, idx: int, val: str) -> None:
            maps_[a][idx] = val
            for pos in occ[a][idx]:
                pt_letters_[pos] = val

        for _move in range(max_moves):
            best_mix = cur_mix
            best_obj = cur_obj
            best_move = None

            for a in range(period):
                cand = candidate_letters(a)
                L = len(cand)

                for xi in range(L):
                    i = cand[xi]
                    for xj in range(xi + 1, L):
                        j = cand[xj]

                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]

                        obj = objective_from_letters(pt_letters_)
                        if obj >= best_obj - near_band:
                            full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                            wb = word_bonus(full)
                            aff = word_affinity(full, miss_weight=miss_weight)
                            mix = obj + lam_wb * wb + lam_aff * aff
                            if mix > best_mix:
                                best_mix = mix
                                best_obj = obj
                                best_move = ("swap", a, i, j)

                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]

                top = min(L, 8)
                for ai in range(top):
                    for aj in range(ai + 1, top):
                        for ak in range(aj + 1, top):
                            i, j, k = cand[ai], cand[aj], cand[ak]
                            A, B, C = maps_[a][i], maps_[a][j], maps_[a][k]

                            apply_map(a, i, B)
                            apply_map(a, j, C)
                            apply_map(a, k, A)
                            obj0 = objective_from_letters(pt_letters_)
                            if obj0 >= best_obj - near_band:
                                full0 = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                                wb0 = word_bonus(full0)
                                aff0 = word_affinity(full0, miss_weight=miss_weight)
                                mix0 = obj0 + lam_wb * wb0 + lam_aff * aff0
                                if mix0 > best_mix:
                                    best_mix = mix0
                                    best_obj = obj0
                                    best_move = ("cyc", a, i, j, k, 0)

                            apply_map(a, i, C)
                            apply_map(a, j, A)
                            apply_map(a, k, B)
                            obj1 = objective_from_letters(pt_letters_)
                            if obj1 >= best_obj - near_band:
                                full1 = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                                wb1 = word_bonus(full1)
                                aff1 = word_affinity(full1, miss_weight=miss_weight)
                                mix1 = obj1 + lam_wb * wb1 + lam_aff * aff1
                                if mix1 > best_mix:
                                    best_mix = mix1
                                    best_obj = obj1
                                    best_move = ("cyc", a, i, j, k, 1)

                            apply_map(a, i, A)
                            apply_map(a, j, B)
                            apply_map(a, k, C)

                top4 = min(L, 9)
                for ai in range(top4):
                    for aj in range(ai + 1, top4):
                        for ak in range(aj + 1, top4):
                            for al in range(ak + 1, top4):
                                i, j, k, l = cand[ai], cand[aj], cand[ak], cand[al]
                                A, B, C, D = maps_[a][i], maps_[a][j], maps_[a][k], maps_[a][l]

                                cycles = [
                                    (B, C, D, A),
                                    (B, D, A, C),
                                    (C, A, D, B),
                                    (C, D, B, A),
                                    (D, A, B, C),
                                    (D, C, A, B),
                                ]
                                for mode, (i_val, j_val, k_val, l_val) in enumerate(cycles):
                                    apply_map(a, i, i_val)
                                    apply_map(a, j, j_val)
                                    apply_map(a, k, k_val)
                                    apply_map(a, l, l_val)

                                    obj = objective_from_letters(pt_letters_)
                                    if obj >= best_obj - near_band:
                                        full = _apply_to_fulltext(
                                            ciphertext, letters_meta, positions, cases, maps_, period
                                        )
                                        wb = word_bonus(full)
                                        aff = word_affinity(full, miss_weight=miss_weight)
                                        mix = obj + lam_wb * wb + lam_aff * aff
                                        if mix > best_mix:
                                            best_mix = mix
                                            best_obj = obj
                                            best_move = ("cyc4", a, i, j, k, l, mode)

                                    apply_map(a, i, A)
                                    apply_map(a, j, B)
                                    apply_map(a, k, C)
                                    apply_map(a, l, D)

            if best_move is None or best_mix <= cur_mix:
                break

            if best_move[0] == "swap":
                _, a, i, j = best_move
                maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                for idx in occ[a][i]:
                    pt_letters_[idx] = maps_[a][i]
                for idx in occ[a][j]:
                    pt_letters_[idx] = maps_[a][j]
            elif best_move[0] == "cyc":
                _, a, i, j, k, mode = best_move
                A, B, C = maps_[a][i], maps_[a][j], maps_[a][k]
                if mode == 0:
                    maps_[a][i], maps_[a][j], maps_[a][k] = B, C, A
                else:
                    maps_[a][i], maps_[a][j], maps_[a][k] = C, A, B
                for idx in occ[a][i]:
                    pt_letters_[idx] = maps_[a][i]
                for idx in occ[a][j]:
                    pt_letters_[idx] = maps_[a][j]
                for idx in occ[a][k]:
                    pt_letters_[idx] = maps_[a][k]
            elif best_move[0] == "cyc4":
                _, a, i, j, k, l, mode = best_move
                A, B, C, D = maps_[a][i], maps_[a][j], maps_[a][k], maps_[a][l]
                cycles = [
                    (B, C, D, A),
                    (B, D, A, C),
                    (C, A, D, B),
                    (C, D, B, A),
                    (D, A, B, C),
                    (D, C, A, B),
                ]
                i_val, j_val, k_val, l_val = cycles[mode]
                apply_map(a, i, i_val)
                apply_map(a, j, j_val)
                apply_map(a, k, k_val)
                apply_map(a, l, l_val)
            else:
                pass

            cur_obj = best_obj
            cur_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
            cur_wb = word_bonus(cur_full)
            cur_aff = word_affinity(cur_full, miss_weight=miss_weight)
            cur_mix = cur_obj + lam_wb * cur_wb + lam_aff * cur_aff

        return maps_, pt_letters_

    def final_affinity_polish(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        passes: int = 2,
        lam_wb: float = 6.0,
        lam_aff: float = 60.0,
        near_band: float = 40.0,
        miss_weight: float = 0.6,
    ) -> tuple[list[list[str]], list[str]]:
        maps_ = [m[:] for m in maps_in]
        pt_letters_ = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters_)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        best_wb = word_bonus(best_full)
        best_aff = word_affinity(best_full, miss_weight=miss_weight)
        best_mix = best_obj_local + lam_wb * best_wb + lam_aff * best_aff

        for _pass in range(passes):
            improved = False
            for a in range(period):
                for i in range(26):
                    for j in range(i + 1, 26):
                        maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                        for idx in occ[a][i]:
                            pt_letters_[idx] = maps_[a][i]
                        for idx in occ[a][j]:
                            pt_letters_[idx] = maps_[a][j]

                        obj = objective_from_letters(pt_letters_)
                        if obj < best_obj_local - near_band:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
                            continue

                        full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                        wb = word_bonus(full)
                        aff = word_affinity(full, miss_weight=miss_weight)
                        mix = obj + lam_wb * wb + lam_aff * aff

                        if mix > best_mix:
                            best_mix = mix
                            best_obj_local = obj
                            improved = True
                        else:
                            maps_[a][i], maps_[a][j] = maps_[a][j], maps_[a][i]
                            for idx in occ[a][i]:
                                pt_letters_[idx] = maps_[a][i]
                            for idx in occ[a][j]:
                                pt_letters_[idx] = maps_[a][j]
            if not improved:
                break

        return maps_, pt_letters_

    def final_affinity_anneal(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        restarts: int = 4,
        steps: int = 6000,
        lam_wb: float = 6.0,
        lam_aff: float = 60.0,
        miss_weight: float = 0.6,
        t0: float = 5.0,
        t1: float = 0.4,
        double_swap_prob: float = 0.45,
    ) -> tuple[list[list[str]], list[str]]:
        best_maps: list[list[str]] | None = None
        best_letters: list[str] | None = None
        best_mix = float("-inf")

        for _ in range(max(1, restarts)):
            maps_ = [m[:] for m in maps_in]
            pt_letters_ = pt_letters_in[:]

            cur_obj = objective_from_letters(pt_letters_)
            cur_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
            cur_wb = word_bonus(cur_full)
            cur_aff = word_affinity(cur_full, miss_weight=miss_weight)
            cur_mix = cur_obj + lam_wb * cur_wb + lam_aff * cur_aff

            for step in range(steps):
                t = t0 * ((t1 / t0) ** (step / max(1, steps - 1)))
                swaps: list[tuple[int, int, int]] = []

                a = rng.randrange(period)
                i = rng.randrange(26)
                j = rng.randrange(26)
                if i == j:
                    continue
                swaps.append((a, i, j))

                if rng.random() < double_swap_prob:
                    a2 = rng.randrange(period)
                    i2 = rng.randrange(26)
                    j2 = rng.randrange(26)
                    if i2 != j2:
                        swaps.append((a2, i2, j2))

                for a_s, i_s, j_s in swaps:
                    maps_[a_s][i_s], maps_[a_s][j_s] = maps_[a_s][j_s], maps_[a_s][i_s]
                    for idx in occ[a_s][i_s]:
                        pt_letters_[idx] = maps_[a_s][i_s]
                    for idx in occ[a_s][j_s]:
                        pt_letters_[idx] = maps_[a_s][j_s]

                new_obj = objective_from_letters(pt_letters_)
                new_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
                new_wb = word_bonus(new_full)
                new_aff = word_affinity(new_full, miss_weight=miss_weight)
                new_mix = new_obj + lam_wb * new_wb + lam_aff * new_aff

                accept = new_mix >= cur_mix or (t > 0 and rng.random() < math.exp((new_mix - cur_mix) / t))
                if accept:
                    cur_obj = new_obj
                    cur_wb = new_wb
                    cur_aff = new_aff
                    cur_mix = new_mix
                else:
                    for a_s, i_s, j_s in reversed(swaps):
                        maps_[a_s][i_s], maps_[a_s][j_s] = maps_[a_s][j_s], maps_[a_s][i_s]
                        for idx in occ[a_s][i_s]:
                            pt_letters_[idx] = maps_[a_s][i_s]
                        for idx in occ[a_s][j_s]:
                            pt_letters_[idx] = maps_[a_s][j_s]

            if cur_mix > best_mix:
                best_mix = cur_mix
                best_maps = [m[:] for m in maps_]
                best_letters = pt_letters_[:]

        if best_maps is None or best_letters is None:
            return maps_in, pt_letters_in

        return best_maps, best_letters
    start = time.perf_counter()
    calibrated = False
    cal_temp_start = temp_start
    cal_temp_end = temp_end

    streams = _streams_from_letters_meta(ciphertext, letters_meta, period)

    for r in range(max(1, restarts)):
        if r > 0 and (time.perf_counter() - start) > max_seconds:
            break

        if best_maps is not None and r > 0 and rng.random() < 0.60:
            maps = [m[:] for m in best_maps]
            for a in range(period):
                for _ in range(40):
                    i, j = rng.randrange(26), rng.randrange(26)
                    if i != j:
                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        elif r < 12:
            maps = _init_maps_by_freq_streams(streams)
            if r > 0:
                swaps = 10 + 10 * r
                for a in range(period):
                    for _ in range(swaps):
                        i, j = rng.randrange(26), rng.randrange(26)
                        if i != j:
                            maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        else:
            maps = _random_maps(rng, period)

        pt_letters = list(_decrypt_letters_only(letters_meta, maps, period))
        cur_obj = objective_from_letters(pt_letters)
        maybe_update_best_with_word_tiebreak(cur_obj, maps, pt_letters, step=0)

        if auto_calibrate_temps and not calibrated:
            maps_tmp = [m[:] for m in maps]
            pt_tmp = pt_letters[:]
            cal_temp_start, cal_temp_end = _calibrate_temps(
                rng=rng,
                period=period,
                maps=maps_tmp,
                occ=occ,
                pt_letters=pt_tmp,
                objective_fn=objective_from_letters,
            )
            calibrated = True
            _log(f"[periodic_sub p={period}] calibrated temps: start={cal_temp_start:.2f} end={cal_temp_end:.2f}", level=1)

        if r == 0 and log_level >= 1:
            pt0 = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
            rep0 = plaintext_fitness(pt0)
            _log(f"[periodic_sub p={period}] init report_fit={rep0:.2f} obj={cur_obj:.2f}", level=1)

        last_best_log_step = -10**9
        last_best_log_obj = best_obj
        best_logs = 0

        for step in range(steps):
            if (time.perf_counter() - start) > max_seconds:
                break

            t = cal_temp_start * ((cal_temp_end / cal_temp_start) ** (step / max(1, steps - 1)))

            a = rng.randrange(period)
            i = rng.randrange(26)
            j = rng.randrange(26)
            if i == j:
                continue

            maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
            for idx in occ[a][i]:
                pt_letters[idx] = maps[a][i]
            for idx in occ[a][j]:
                pt_letters[idx] = maps[a][j]

            new_obj = objective_from_letters(pt_letters)
            accept = new_obj >= cur_obj or (t > 0 and rng.random() < math.exp((new_obj - cur_obj) / t))

            if accept:
                cur_obj = new_obj
                prev_best = best_obj
                maybe_update_best_with_word_tiebreak(cur_obj, maps, pt_letters, step)

                if log_level >= 1 and best_obj > prev_best:
                    improved_by = best_obj - prev_best
                    should_print = (
                        (best_logs < log_best_max_per_restart)
                        and (
                            (step - last_best_log_step) >= log_best_stride_steps
                            or improved_by >= log_best_min_delta
                        )
                    )
                    if should_print:
                        _log(
                            f"[periodic_sub p={period}] best obj {best_obj:.2f} "
                            f"(restart {r}, step {step}, +{improved_by:.2f})",
                            level=1,
                        )
                        last_best_log_step = step
                        last_best_log_obj = best_obj
                        best_logs += 1
            else:
                maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                for idx in occ[a][i]:
                    pt_letters[idx] = maps[a][i]
                for idx in occ[a][j]:
                    pt_letters[idx] = maps[a][j]

        if log_level >= 2:
            _log(f"[periodic_sub p={period}] restart {r} done; current best_obj={best_obj:.2f}", level=2)

    if best_maps is None or best_letters is None:
        return None

    # greedy polish on the same objective (letters-only)
    best_maps = [m[:] for m in best_maps]
    best_letters = best_letters[:]
    best_obj, best_maps, best_letters = _greedy_polish(
        occ=occ,
        maps=best_maps,
        pt_letters=best_letters,
        period=period,
        score_letters_fn=objective_from_letters,
        max_passes=6,
    )

    if enable_word_repair:
        best_maps, best_letters = final_word_repair(
            best_maps,
            best_letters,
            max_passes=word_repair_passes,
            swaps_per_alpha=word_repair_swaps_per_alpha,
            lam=word_repair_lam,
        )

    guided_passes = 6
    guided_lam = 1.25
    guided_max_obj_drop = 6.0
    full_after_repair = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, best_maps, period)
    el_after_repair = english_likeness_score(full_after_repair)
    wb_after_repair = word_bonus(full_after_repair)
    if el_after_repair >= 65.0 and wb_after_repair >= 4.0:
        guided_passes = 8
        guided_lam = 10.0
        guided_max_obj_drop = 12.0

    best_maps, best_letters = final_guided_polish(
        best_maps,
        best_letters,
        passes=guided_passes,
        lam=guided_lam,
        max_obj_drop=guided_max_obj_drop,
        candidate_letters="BDVXZJQK",
    )

    best_maps, best_letters = final_word_polish_all(
        best_maps,
        best_letters,
        passes=3,
        lam=12.0,
        near_band=25.0,
    )

    best_maps, best_letters = final_word_kopt(
        best_maps,
        best_letters,
        top_k_high=10,
        bottom_k_low=8,
        max_moves=6,
        lam_wb=6.0,
        lam_aff=60.0,
        near_band=40.0,
        miss_weight=0.6,
    )

    full_after_kopt = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, best_maps, period)
    if english_likeness_score(full_after_kopt) >= 65.0 and word_bonus(full_after_kopt) >= 4.0:
        best_maps, best_letters = final_affinity_polish(
            best_maps,
            best_letters,
            passes=2,
            lam_wb=6.0,
            lam_aff=60.0,
            near_band=40.0,
            miss_weight=0.6,
        )
        best_maps, best_letters = final_affinity_anneal(
            best_maps,
            best_letters,
            restarts=4,
            steps=6000,
            lam_wb=6.0,
            lam_aff=60.0,
            miss_weight=0.6,
            t0=5.0,
            t1=0.4,
            double_swap_prob=0.45,
        )

    plaintext = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, best_maps, period)
    report_fit = plaintext_fitness(plaintext)
    return best_obj, best_maps, plaintext, report_fit


# ---------------------------
# Plugin class
# ---------------------------

class PeriodicSubstitutionCipher:
    name = "periodic_substitution"

    def decrypt(self, ciphertext: str, key: str) -> str:
        raise ValueError("Use crack() for periodic substitution. (Known-key mode not implemented yet.)")

    def crack(self, ciphertext: str) -> list[SolveResult]:
        az_len = len(normalize_az(ciphertext))

        # Focus: periods 2..6 only (master these first)
        periods = _rank_periods(ciphertext, max_period=6, top_n=4)

        # time budgets (tuned for your CLI use)
        total_seconds = 180.0
        if az_len < 160:
            total_seconds = 35.0
        elif az_len < 220:
            total_seconds = 70.0

        # prefer p=3 early if present (common)
        if 3 in periods:
            periods = [3] + [p for p in periods if p != 3]

        # For shorter texts, keep candidate count small to avoid overfitting selection.
        if az_len < 220:
            periods = [p for p in periods if 2 <= p <= 6][:3]  # often [3,2,4]

        if not periods:
            periods = [3]

        # budget split: more to first
        if len(periods) == 1:
            budgets = [total_seconds]
        else:
            first = total_seconds * 0.70
            rem = total_seconds - first
            budgets = [first] + [rem / (len(periods) - 1)] * (len(periods) - 1)

        best: Optional[tuple[float, list[list[str]], str, int, float]] = None
        best_key: Optional[tuple[float, float, float, float]] = None
        # best_key = (selection_score, wb, report_fit, obj_letters)

        base_steps_35s = 30000
        base_restarts_35s = 60

        for period, max_seconds in zip(periods, budgets):
            scale = max_seconds / 35.0
            restarts = int(max(20, min(240, round(base_restarts_35s * scale))))
            steps = int(max(15000, min(200000, round(base_steps_35s * scale))))

            out = crack_periodic_substitution_global(
                ciphertext,
                period=period,
                restarts=restarts,
                steps=steps,
                max_seconds=max_seconds,
                auto_calibrate_temps=True,
                temp_start=12.0,
                temp_end=0.2,
                verbose=True,
                enable_word_repair=True,
                word_repair_lam=0.35,
                word_repair_passes=3,
                word_repair_swaps_per_alpha=400,
                log_level=1,
                log_best_stride_steps=3000,
                log_best_min_delta=1.0,
                log_best_max_per_restart=10,
            )
            if out is None:
                continue

            obj_letters, maps, pt, report_fit = out
            wb = word_bonus(pt)

            period_penalty = 60.0 * max(0, period - 3)
            selection_score = report_fit + 8.0 * wb - period_penalty

            key = (selection_score, wb, report_fit, obj_letters)
            if best_key is None or key > best_key:
                best_key = key
                best = (obj_letters, maps, pt, period, report_fit)

            if period == 3:
                el3 = english_likeness_score(pt)
                if el3 >= 70.0 and wb >= 6.0:
                    break
        if best is None:
            alpha = sum(ch.isalpha() for ch in ciphertext)
            return [
                SolveResult(
                    cipher_name=self.name,
                    plaintext="",
                    key=None,
                    score=float("-inf"),
                    confidence=0.0,
                    notes=f"No periodic_sub result produced (alpha_letters={alpha}).",
                    meta={"fitness": "quadgram" if get_quadgram_scorer() else "fallback"},
                )
            ]

        obj_letters, maps, pt, period, report_fit = best
        key_str = " | ".join("".join(m) for m in maps)

        el = english_likeness_score(pt)
        wb = word_bonus(pt)
        label = _reliability_label(english_like=el, wb=wb, alpha_len=az_len)
        notice = _reliability_notice(label)

        conf = max(0.0, min(1.0, (el - 20.0) / 80.0))
        if label == "medium":
            conf = min(conf, 0.85)
        elif label == "low":
            conf = min(conf, 0.60)

        if az_len < 140:
            conf = min(conf, 0.55)
        elif az_len < 220:
            conf = min(conf, 0.75)

        notes = (
            f"Period ranking (IoC+Kasiski) + stream init + global annealing "
            f"over {period} alphabets (period={period}) + polish/repair; reliability={label}. {notice}"
        )

        return [
            SolveResult(
                cipher_name=self.name,
                plaintext=pt,
                key=key_str,
                score=float(report_fit),
                confidence=float(conf),
                notes=notes,
                meta={
                    "period": period,
                    "fitness": "quadgram" if get_quadgram_scorer() else "fallback",
                    "objective_letters": float(obj_letters),
                    "report_fitness": float(report_fit),
                    "english_likeness": float(el),
                    "word_bonus": float(wb),
                    "reliability": label,
                },
            )
        ]

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "polyalphabetic"}


# ---------------------------
# Helper: known maps decrypt
# ---------------------------

def decrypt_periodic(ciphertext: str, maps: list[str], period: int = 3) -> str:
    out = []
    li = 0
    for ch in ciphertext:
        up = ch.upper()
        if "A" <= up <= "Z":
            a = li % period
            p = maps[a][ord(up) - 65]
            out.append(p if ch.isupper() else p.lower())
            li += 1
        else:
            out.append(ch)
    return "".join(out)


if __name__ == "__main__":
    ct = "PJTE PD PJA AYKIQNTNN GC NVA ..."
    maps = [
        "PGFYSDOMICUJRQBTVNWEKHLAXZ",
        "ESFMDROWBHJKQXZINUPAYGCTLV",
        "ERDSYGMJCHAKLTQOVWUIBXPFNZ",
    ]
    print(decrypt_periodic(ct, maps, 3)[:80])


register_plugin(PeriodicSubstitutionCipher(), should_try=_should_try)
