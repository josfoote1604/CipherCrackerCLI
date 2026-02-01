from __future__ import annotations

import math
import random
import sys
import time
import re
from collections import Counter
from typing import Optional

from ciphercracker.core.features import analyze_text, ioc_scan
from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import (
    english_likeness_score,
    get_quadgram_scorer,
    plaintext_fitness,
    quadgram_score,
    word_bonus,
)
from ciphercracker.core.utils import normalize_az

_PRIMARY_NEAR_BAND = 20.0     # how close to “best_obj” counts as a tie (quadgram scale)
_WORDCHECK_EVERY = 250        # compute word_bonus at most every N steps
_WORDCHECK_PROB = 0.02        # …or randomly at this probability (whichever hits first)


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


_WORD_RE = re.compile(r"[A-Za-z]{2,}")

def _reliability_label(*, english_like: float, wb: float, alpha_len: int) -> str:
    """
    Heuristic label. Tuned to be conservative.
    - english_like: english_likeness_score(pt)
    - wb: word_bonus(pt) (your dictionary-ish signal)
    - alpha_len: len(normalize_az(ciphertext)) for mild length adjustment
    """
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
    return "Heuristic solve; output may be partially incorrect—treat as a strong hint, not a guarantee."



def _pick_period_candidates(ct: str, *, iocmax: int = 12, top_n: int = 6) -> list[int]:
    az = normalize_az(ct)
    n = len(az)

    # Small periods first (most likely in classical examples)
    base = [3, 2, 4, 5, 6]

    out: list[int] = []
    seen: set[int] = set()

    for k in base:
        if 2 <= k <= iocmax and k not in seen:
            out.append(k)
            seen.add(k)

    # For short ciphertexts, IoC peaks at large k are often noise—avoid them.
    if n < 220:
        return out

    top = ioc_scan(ct, max_len=iocmax)[:top_n]
    for k, _ in top:
        if 2 <= k <= iocmax and k not in seen:
            out.append(k)
            seen.add(k)

    return out


def _should_try(ct: str) -> bool:
    """
    Only try periodic substitution when there is a *credible* periodicity signal.
    """
    az = normalize_az(ct)
    n = len(az)

    if n < 90:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.70:
        return False

    # IMPORTANT: do NOT slice here; we want k=1..max_len coverage
    scan = ioc_scan(ct, max_len=12)
    if not scan:
        return False

    scan_map = {k: v for k, v in scan}
    ioc1 = scan_map.get(1)
    if ioc1 is None:
        return False

    min_col_len = 15

    # For short texts, only trust small periods; big-k peaks are usually noise.
    candidate_ks = range(2, 7) if n < 220 else range(2, 13)

    best_k: int | None = None
    best_val = float("-inf")

    for k in candidate_ks:
        v = scan_map.get(k)
        if v is None:
            continue
        if (n // k) < min_col_len:
            continue
        if v > best_val:
            best_val = v
            best_k = k

    if best_k is None:
        return False

    bump = best_val - ioc1

    min_bump = 0.008 if n < 220 else 0.006
    min_abs = 0.056 if n < 220 else 0.062

    return (best_val >= min_abs) and (bump >= min_bump)


def _letters_meta(text: str) -> tuple[list[tuple[int, int]], list[int], list[bool]]:
    """
    Returns:
      letters_meta: list of (alpha_idx, cipher_idx)
      positions: positions in original string for each letter
      cases: True if original was uppercase, False if lowercase
    alpha_idx counts letters only (ignores spaces/punct) so alpha_idx % period selects alphabet.
    """
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


def _init_maps_by_freq(ciphertext: str, period: int) -> list[list[str]]:
    letters_meta, _, _ = _letters_meta(ciphertext)

    streams = [""] * period
    for li, cidx in letters_meta:
        a = li % period
        streams[a] += chr(65 + cidx)

    maps: list[list[str]] = []
    for s in streams:
        counts = Counter(s)
        cipher_order = "".join(sorted(ALPHABET, key=lambda c: counts[c], reverse=True))

        mapping = ["?"] * 26
        for i, ciph in enumerate(cipher_order):
            mapping[ord(ciph) - 65] = _ENGLISH_FREQ_ORDER[i]
        maps.append(mapping)

    # Fill any '?' with unused letters
    for a in range(period):
        used = set(maps[a])
        unused = [c for c in ALPHABET if c not in used]
        it = iter(unused)
        for i in range(26):
            if maps[a][i] == "?":
                maps[a][i] = next(it)

    return maps


def _random_maps(rng: random.Random, period: int) -> list[list[str]]:
    maps: list[list[str]] = []
    for _ in range(period):
        letters = list(ALPHABET)
        rng.shuffle(letters)
        maps.append(letters)
    return maps


def _decrypt_letters_only(letters_meta: list[tuple[int, int]], maps: list[list[str]], period: int) -> str:
    out: list[str] = []
    for li, cidx in letters_meta:
        a = li % period
        out.append(maps[a][cidx])
    return "".join(out)


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


def _greedy_polish(
    occ: list[list[list[int]]],
    maps: list[list[str]],
    pt_letters: list[str],
    period: int,
    score_letters_fn,
    max_passes: int = 6,
) -> tuple[float, list[list[str]], list[str]]:
    """
    IMPORTANT: This polish optimizes the SAME objective used by SA:
    the scorer is applied to the *letters-only* plaintext (pt_letters).
    """
    cur_s = score_letters_fn(pt_letters)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for a in range(period):
            for i in range(26):
                for j in range(i + 1, 26):
                    # swap in the map
                    maps[a][i], maps[a][j] = maps[a][j], maps[a][i]

                    # update impacted plaintext letter positions
                    for idx in occ[a][i]:
                        pt_letters[idx] = maps[a][i]
                    for idx in occ[a][j]:
                        pt_letters[idx] = maps[a][j]

                    new_s = score_letters_fn(pt_letters)
                    if new_s > cur_s:
                        cur_s = new_s
                        improved = True
                    else:
                        # revert
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
    samples: int = 240,
    quantile: float = 0.85,
) -> tuple[float, float]:
    """
    Empirically estimate a reasonable temp_start/temp_end by sampling objective deltas
    from random swaps. This helps across different ciphertext lengths/scales.

    Strategy:
      - collect negative deltas (worsening moves)
      - temp_start picks a value so exp(delta/T) is not tiny for typical delta
      - temp_end is a small fraction of start, but not too small
    """
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

        # revert
        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        for idx in occ[a][i]:
            pt_letters[idx] = maps[a][i]
        for idx in occ[a][j]:
            pt_letters[idx] = maps[a][j]

    if not neg_deltas:
        # if we didn't see any negative deltas, use a safe-ish default
        return 12.0, 0.4

    neg_deltas.sort()
    # pick a "typical" negative delta (closer to 0 is mild, more negative is harsher)
    k = max(0, min(len(neg_deltas) - 1, int(quantile * (len(neg_deltas) - 1))))
    typical = neg_deltas[k]  # negative

    # want exp(typical/T) around ~0.3..0.5 early on
    # solve exp(typical/T)=0.35  -> T = typical / ln(0.35)
    target = 0.35
    t_start = float(typical / math.log(target))  # typical is negative, log(target) is negative
    t_start = max(5.0, min(400.0, t_start))

    t_end = max(0.8, min(20.0, 0.05 * t_start))
    return t_start, t_end


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
    # --- logging controls ---
    log_level: int = 1,                 # 0=silent, 1=compact, 2=more detail
    log_best_stride_steps: int = 2500,  # don’t print best-improve more often than this
    log_best_min_delta: float = 0.75,   # …unless improvement >= this
    log_best_max_per_restart: int = 12, # cap spam per restart
) -> Optional[tuple[float, list[list[str]], str, float]]:
    """
    Returns:
      (best_objective, best_maps, plaintext_full, report_fitness)

    Logging:
      - If verbose=False => silent regardless of log_level.
      - log_level=1 => temps/init + throttled best-improve + restart summaries
      - log_level=2 => same but slightly chattier (you can tune params)
    """
    rng = random.Random(seed)
    use_quad = get_quadgram_scorer() is not None

    # hard-disable logs if verbose is False
    if not verbose:
        log_level = 0

    def _log(msg: str, *, level: int = 1) -> None:
        if log_level >= level:
            print(msg, file=sys.stderr)

    letters_meta, positions, cases = _letters_meta(ciphertext)
    min_letters = max(60, period * 25)
    if len(letters_meta) < min_letters:
        return None

    # occ[a][c] = indices in pt_letters where alphabet a decrypts cipher letter c
    occ: list[list[list[int]]] = [[[] for _ in range(26)] for _ in range(period)]
    for idx, (li, cidx) in enumerate(letters_meta):
        a = li % period
        occ[a][cidx].append(idx)

    def objective_from_letters(pt_letters_list: list[str]) -> float:
        s = "".join(pt_letters_list)
        if use_quad:
            return quadgram_score(s)
        return english_likeness_score(s)

    # --- near-band tie-break bookkeeping (kept lightweight) ---
    best_obj = float("-inf")
    best_word = float("-inf")
    best_maps: Optional[list[list[str]]] = None
    best_letters: Optional[list[str]] = None

    def maybe_update_best_with_word_tiebreak(
        cur_obj: float,
        maps: list[list[str]],
        pt_letters: list[str],
        step: int,
    ) -> None:
        nonlocal best_obj, best_maps, best_letters, best_word

        # Always take clearly better primary objective
        if cur_obj > best_obj:
            best_obj = cur_obj
            best_maps = [m[:] for m in maps]
            best_letters = pt_letters[:]

            # Only compute word_bonus occasionally even on best_obj updates
            if step % _WORDCHECK_EVERY == 0:
                pt_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
                best_word = word_bonus(pt_full)
            return

        # Only consider word tie-break when we're "near" the best primary score
        if cur_obj < (best_obj - _PRIMARY_NEAR_BAND):
            return

        # Throttle expensive fulltext reconstruction + word scan
        if step % _WORDCHECK_EVERY != 0 and rng.random() > _WORDCHECK_PROB:
            return

        pt_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
        wb = word_bonus(pt_full)
        if wb > best_word:
            best_obj = cur_obj
            best_maps = [m[:] for m in maps]
            best_letters = pt_letters[:]
            best_word = wb

    # --- FINAL REPAIR POLISH: small greedy swap search using fulltext word evidence ---
    def final_word_repair(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        max_passes: int = 3,
        swaps_per_alpha: int = 400,
        lam: float = 0.35,
        near_band: float = 8.0,
    ) -> tuple[list[list[str]], list[str]]:
        maps = [m[:] for m in maps_in]
        pt_letters = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
        best_wb = word_bonus(best_full)
        best_mix = best_obj_local + lam * best_wb

        for _pass in range(max_passes):
            improved = False

            for a in range(period):
                for _ in range(swaps_per_alpha):
                    i = rng.randrange(26)
                    j = rng.randrange(26)
                    if i == j:
                        continue

                    maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                    for idx in occ[a][i]:
                        pt_letters[idx] = maps[a][i]
                    for idx in occ[a][j]:
                        pt_letters[idx] = maps[a][j]

                    obj = objective_from_letters(pt_letters)
                    if obj < best_obj_local - near_band:
                        # revert
                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                        for idx in occ[a][i]:
                            pt_letters[idx] = maps[a][i]
                        for idx in occ[a][j]:
                            pt_letters[idx] = maps[a][j]
                        continue

                    full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
                    wb = word_bonus(full)
                    mix = obj + lam * wb

                    if mix > best_mix:
                        best_mix = mix
                        best_obj_local = obj
                        improved = True
                    else:
                        # revert
                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                        for idx in occ[a][i]:
                            pt_letters[idx] = maps[a][i]
                        for idx in occ[a][j]:
                            pt_letters[idx] = maps[a][j]

            if not improved:
                break

        return maps, pt_letters

    def final_guided_polish(
        maps_in: list[list[str]],
        pt_letters_in: list[str],
        *,
        passes: int = 6,
        lam: float = 1.25,
        max_obj_drop: float = 6.0,
        candidate_letters: str = "BDVXZJQK",
    ) -> tuple[list[list[str]], list[str]]:
        maps = [m[:] for m in maps_in]
        pt_letters = pt_letters_in[:]

        best_obj_local = objective_from_letters(pt_letters)
        best_full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
        best_wb = word_bonus(best_full)
        best_mix = best_obj_local + lam * best_wb

        cand = [c for c in candidate_letters if c in ALPHABET]

        for _ in range(passes):
            improved = False

            for a in range(period):
                inv = {maps[a][i]: i for i in range(26)}

                for x_i in range(len(cand)):
                    for y_i in range(x_i + 1, len(cand)):
                        x = cand[x_i]
                        y = cand[y_i]
                        if x not in inv or y not in inv:
                            continue

                        i = inv[x]
                        j = inv[y]

                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                        for idx in occ[a][i]:
                            pt_letters[idx] = maps[a][i]
                        for idx in occ[a][j]:
                            pt_letters[idx] = maps[a][j]

                        obj = objective_from_letters(pt_letters)
                        if obj < best_obj_local - max_obj_drop:
                            # revert
                            maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                            for idx in occ[a][i]:
                                pt_letters[idx] = maps[a][i]
                            for idx in occ[a][j]:
                                pt_letters[idx] = maps[a][j]
                            continue

                        full = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps, period)
                        wb = word_bonus(full)
                        mix = obj + lam * wb

                        if mix > best_mix:
                            best_mix = mix
                            best_obj_local = obj
                            improved = True
                        else:
                            # revert
                            maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                            for idx in occ[a][i]:
                                pt_letters[idx] = maps[a][i]
                            for idx in occ[a][j]:
                                pt_letters[idx] = maps[a][j]

            if not improved:
                break

        return maps, pt_letters

    start = time.perf_counter()
    calibrated = False
    cal_temp_start = temp_start
    cal_temp_end = temp_end

    for r in range(max(1, restarts)):
        if r > 0 and (time.perf_counter() - start) > max_seconds:
            break

        # initialization
        if best_maps is not None and r > 0 and rng.random() < 0.60:
            maps = [m[:] for m in best_maps]
            for a in range(period):
                for _ in range(40):
                    i, j = rng.randrange(26), rng.randrange(26)
                    if i != j:
                        maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
        elif r < 12:
            maps = _init_maps_by_freq(ciphertext, period)
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

        # Calibrate temps once per call
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

        # per-restart log throttles
        last_best_log_step = -10**9
        last_best_log_obj = best_obj
        best_logs = 0

        # anneal
        for step in range(steps):
            if (time.perf_counter() - start) > max_seconds:
                break

            t = cal_temp_start * ((cal_temp_end / cal_temp_start) ** (step / max(1, steps - 1)))

            a = rng.randrange(period)
            i = rng.randrange(26)
            j = rng.randrange(26)
            if i == j:
                continue

            # propose swap
            maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
            for idx in occ[a][i]:
                pt_letters[idx] = maps[a][i]
            for idx in occ[a][j]:
                pt_letters[idx] = maps[a][j]

            new_obj = objective_from_letters(pt_letters)

            accept = False
            if new_obj >= cur_obj:
                accept = True
            elif t > 0:
                accept = (rng.random() < math.exp((new_obj - cur_obj) / t))

            if accept:
                cur_obj = new_obj
                prev_best = best_obj
                maybe_update_best_with_word_tiebreak(cur_obj, maps, pt_letters, step)

                # CLEAN LOG: throttled best-improvement printing
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
                            f"[periodic_sub p={period}] best obj {best_obj:.2f} (restart {r}, step {step}, +{improved_by:.2f})",
                            level=1,
                        )
                        last_best_log_step = step
                        last_best_log_obj = best_obj
                        best_logs += 1
            else:
                # revert swap
                maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                for idx in occ[a][i]:
                    pt_letters[idx] = maps[a][i]
                for idx in occ[a][j]:
                    pt_letters[idx] = maps[a][j]

        if log_level >= 2:
            _log(f"[periodic_sub p={period}] restart {r} done; current best_obj={best_obj:.2f}", level=2)

    if best_maps is None or best_letters is None:
        return None

    # greedy polish on SAME objective (letters-only)
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

    best_maps, best_letters = final_guided_polish(
        best_maps,
        best_letters,
        passes=6,
        lam=1.25,
        max_obj_drop=6.0,
        candidate_letters="BDVXZJQK",
    )

    plaintext = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, best_maps, period)
    report_fit = plaintext_fitness(plaintext)
    return best_obj, best_maps, plaintext, report_fit




class PeriodicSubstitutionCipher:
    name = "periodic_substitution"

    def decrypt(self, ciphertext: str, key: str) -> str:
        raise ValueError("Use crack() for periodic substitution. (Known-key mode not implemented yet.)")

    def crack(self, ciphertext: str) -> list[SolveResult]:
        periods = _pick_period_candidates(ciphertext, iocmax=12, top_n=6)

        total_seconds = 180.0
        az_len = len(normalize_az(ciphertext))
        if az_len < 160:
            total_seconds = 35.0
        elif az_len < 220:
            total_seconds = 70.0

        # prefer p=3 first (common in examples)
        if 3 in periods:
            periods = [3] + [p for p in periods if p != 3]

        # On shorter texts, high periods overfit easily; keep it tight
        if az_len < 220:
            periods = [p for p in periods if p <= 6][:3]  # typically [3,2,4]

        if not periods:
            periods = [3]

        # budget: spend more on the first candidate, then split remainder
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
                # cleaner logging defaults:
                log_level=1,
                log_best_stride_steps=3000,
                log_best_min_delta=1.0,
                log_best_max_per_restart=10,
                # Optional: make runs reproducible while debugging:
                # seed=0,
            )
            if out is None:
                continue

            obj_letters, maps, pt, report_fit = out
            wb = word_bonus(pt)

            # Penalize large periods a bit to prevent overfitting from winning selection.
            period_penalty = 60.0 * max(0, period - 3)
            selection_score = report_fit + 8.0 * wb - period_penalty

            key = (selection_score, wb, report_fit, obj_letters)

            if best_key is None or key > best_key:
                best_key = key
                best = (obj_letters, maps, pt, period, report_fit)

            # Early exit: if p=3 already looks strong, don't waste time on larger periods
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

        notes = (
            f"Global annealing over {period} substitution alphabets (period={period}) + greedy polish "
            f"+ final word-repair; reliability={label}. {notice}"
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
