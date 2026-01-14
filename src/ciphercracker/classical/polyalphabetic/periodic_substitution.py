from __future__ import annotations

import math
import random
import sys
import time
from collections import Counter
from typing import Optional

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import plaintext_fitness, get_quadgram_scorer, quadgram_score, english_likeness_score
from ciphercracker.core.features import analyze_text, ioc_scan
from ciphercracker.core.utils import normalize_az

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


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

    Key idea:
      - For short ciphertexts, IoC peaks at large k are often just variance
        because each column has too few letters.
      - So we:
          1) Require enough letters overall
          2) Require a meaningful IoC bump vs k=1
          3) Ignore k where average column length is too small
          4) For short texts, only consider small periods (2..6)
    """
    az = normalize_az(ct)
    n = len(az)

    if n < 90:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.70:
        return False

    scan = ioc_scan(ct, max_len=12)[:12]
    if not scan:
        return False

    # Locate IoC(k=1)
    ioc1: float | None = None
    for k, v in scan:
        if k == 1:
            ioc1 = v
            break
    if ioc1 is None:
        return False

    # Don't trust IoC at k where columns are tiny.
    # Rule of thumb: need ~15+ letters/column for IoC to stabilize.
    min_col_len = 15

    # For short texts, only trust small periods; big-k peaks are usually noise.
    if n < 220:
        candidate_ks = range(2, 7)  # 2..6
    else:
        candidate_ks = range(2, 13)  # 2..12

    best_k: int | None = None
    best_val = float("-inf")

    # Build a quick lookup from scan list
    scan_map = {k: v for k, v in scan}

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

    # Thresholds:
    # - short inputs are noisy, demand a stronger bump
    min_bump = 0.010 if n < 220 else 0.006
    # also require the candidate IoC to be "meaningfully above random-ish"
    min_abs = 0.058 if n < 220 else 0.062

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
    score_maps_fn,
    max_passes: int = 2,
) -> tuple[float, list[list[str]], list[str]]:
    cur_s = score_maps_fn(maps)

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

                    new_s = score_maps_fn(maps)
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


def crack_periodic_substitution_global(
    ciphertext: str,
    *,
    period: int = 3,
    restarts: int = 60,
    steps: int = 30000,
    temp_start: float = 12.0,
    temp_end: float = 0.2,
    max_seconds: float = 35.0,
    full_score_every: int = 250,
    seed: int | None = None,
    verbose: bool = True,
) -> Optional[tuple[float, list[list[str]], str]]:
    rng = random.Random(seed)
    use_quad = get_quadgram_scorer() is not None

    letters_meta, positions, cases = _letters_meta(ciphertext)
    min_letters = max(60, period * 25)
    if len(letters_meta) < min_letters:
        return None

    occ: list[list[list[int]]] = [[[] for _ in range(26)] for _ in range(period)]
    for idx, (li, cidx) in enumerate(letters_meta):
        a = li % period
        occ[a][cidx].append(idx)

    def score_fast(pt_letters_str: str) -> float:
        if use_quad:
            return quadgram_score(normalize_az(pt_letters_str))
        return english_likeness_score(pt_letters_str)

    def score_maps(maps_: list[list[str]]) -> float:
        pt = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, maps_, period)
        return plaintext_fitness(pt)

    best_full = float("-inf")
    best_maps: Optional[list[list[str]]] = None

    start = time.perf_counter()

    for r in range(max(1, restarts)):
        if r > 0 and (time.perf_counter() - start) > max_seconds:
            break

        if r < 12:
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
        cur_fast = score_fast("".join(pt_letters))
        cur_full = score_maps(maps)

        if r == 0 and verbose:
            print(f"[periodic_sub p={period}] init full={cur_full:.2f} fast={cur_fast:.2f}", file=sys.stderr)

        if cur_full > best_full:
            best_full = cur_full
            best_maps = [m[:] for m in maps]
            if verbose:
                print(f"[periodic_sub p={period}] best full now {best_full:.2f} (restart {r})", file=sys.stderr)

        for step in range(steps):
            if (time.perf_counter() - start) > max_seconds:
                break

            t = temp_start * ((temp_end / temp_start) ** (step / max(1, steps - 1)))

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

            new_fast = score_fast("".join(pt_letters))

            accept = False
            if new_fast >= cur_fast:
                accept = True
            elif t > 0:
                accept = (rng.random() < math.exp((new_fast - cur_fast) / t))

            if accept:
                cur_fast = new_fast
                if (step % full_score_every) == 0:
                    cur_full = score_maps(maps)
                    if cur_full > best_full:
                        best_full = cur_full
                        best_maps = [m[:] for m in maps]
                        if verbose:
                            print(
                                f"[periodic_sub p={period}] best full now {best_full:.2f} (restart {r}, step {step})",
                                file=sys.stderr,
                            )
            else:
                maps[a][i], maps[a][j] = maps[a][j], maps[a][i]
                for idx in occ[a][i]:
                    pt_letters[idx] = maps[a][i]
                for idx in occ[a][j]:
                    pt_letters[idx] = maps[a][j]

        cur_full = score_maps(maps)
        if cur_full > best_full:
            best_full = cur_full
            best_maps = [m[:] for m in maps]
            if verbose:
                print(f"[periodic_sub p={period}] best full now {best_full:.2f} (restart {r} end)", file=sys.stderr)

    if best_maps is None:
        return None

    best_maps = [m[:] for m in best_maps]
    best_letters = list(_decrypt_letters_only(letters_meta, best_maps, period))
    best_full, best_maps, best_letters = _greedy_polish(
        occ=occ,
        maps=best_maps,
        pt_letters=best_letters,
        period=period,
        score_maps_fn=score_maps,
        max_passes=2,
    )

    plaintext = _apply_to_fulltext(ciphertext, letters_meta, positions, cases, best_maps, period)
    return best_full, best_maps, plaintext


class PeriodicSubstitutionCipher:
    name = "periodic_substitution"

    def decrypt(self, ciphertext: str, key: str) -> str:
        raise ValueError("Use crack() for periodic substitution. (Known-key mode not implemented yet.)")

    def crack(self, ciphertext: str) -> list[SolveResult]:
        periods = _pick_period_candidates(ciphertext, iocmax=12, top_n=6)

        total_seconds = 90.0
        az_len = len(normalize_az(ciphertext))

        if 3 in periods:
            periods = [3] + [p for p in periods if p != 3]

        if az_len < 220:
            periods = periods[:3]  # [3,2,4] usually

        best: Optional[tuple[float, list[list[str]], str, int]] = None

        if not periods:
            periods = [3]

        if len(periods) == 1:
            budgets = [total_seconds]
        else:
            first = total_seconds * 0.70
            rem = total_seconds - first
            budgets = [first] + [rem / (len(periods) - 1)] * (len(periods) - 1)

        base_steps_for_90s = 30000

        for period, max_seconds in zip(periods, budgets):
            steps = max(10000, int(base_steps_for_90s * (max_seconds / 90.0)))
            restarts = 140 if period <= 4 else 90

            out = crack_periodic_substitution_global(
                ciphertext,
                period=period,
                restarts=restarts,
                steps=steps,
                max_seconds=max_seconds,
                full_score_every=200 if az_len >= 220 else 100,
                verbose=True,
            )

            if out is None:
                continue

            s, maps, pt = out
            if best is None or s > best[0]:
                best = (s, maps, pt, period)

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

        s, maps, pt, period = best
        key_str = " | ".join("".join(m) for m in maps)

        conf = english_likeness_score(pt) / 100.0

        return [
            SolveResult(
                cipher_name=self.name,
                plaintext=pt,
                key=key_str,
                score=float(s),
                confidence=float(conf),
                notes=f"Global annealing over {period} substitution alphabets (period={period}) + greedy polish",
                meta={"period": period, "fitness": "quadgram" if get_quadgram_scorer() else "fallback", "best_full": float(s)},
            )
        ]

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "polyalphabetic"}


register_plugin(PeriodicSubstitutionCipher(), should_try=_should_try)
