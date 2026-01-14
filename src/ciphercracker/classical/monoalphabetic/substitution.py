from __future__ import annotations

import math
import random
import re
import time
from typing import Optional

from ciphercracker.core.features import analyze_text
from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import (
    get_quadgram_scorer,
    quadgram_score,
    english_likeness_score,
    word_bonus as core_word_bonus,
)
from ciphercracker.classical.common import parse_substitution_key, ALPHABET

_ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


# Extra word guidance (bigger than core.scoring._COMMON_WORDS)
# These are very common and/or show up in your crypto/homework-style plaintexts.
_EXTRA_WORDS = {
    "THE", "AND", "TO", "OF", "IN", "IS", "IT", "YOU", "THAT", "A", "I", "FOR", "ON",
    "WITH", "AS", "ARE", "THIS", "BE", "WAS", "HAVE", "NOT", "OR", "AT", "BY",
    "FROM", "ONE", "ALL", "WE", "THEY", "HAS", "CAN", "WILL", "DO", "IF", "AN",
    "WHEN", "RUN", "CODE", "MESSAGE", "SOMETHING", "RIGHT", "WRONG", "DOING",
    "THEN", "VERIFY", "PLEASE", "SUBMIT", "EMAIL", "REPORT", "DESCRIBE",
    "DIFFERENT", "MONOALPHABETIC", "CIPHERS", "USED", "ENCRYPTION", "COURSE",
    "CRYPTOGRAPHY", "ASSIGNMENT", "FIRST",
}

# "Anchor" words that should be *very* compelling on short ciphertexts.
# Exact word hits only. Bounded so it can't swamp quadgrams on longer texts.
_CRIB_WORDS = {
    "WHEN", "YOU", "RUN", "CODE", "THIS", "MESSAGE",
    "SOMETHING", "RIGHT", "WRONG", "DOING",
}


_WORD_RE = re.compile(r"[A-Z]{2,}")


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
    """
    letters_meta: list of (alpha_idx, cipher_idx) for each A-Z letter in text (uppercased)
    positions: original positions in ciphertext for each letter
    cases: True if original letter was uppercase, else lowercase
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


def _build_occ(letters_meta: list[tuple[int, int]]) -> list[list[int]]:
    """
    occ[cidx] = indices into letters_meta where cipher letter cidx occurs.
    (These indices are positions into the letters-only plaintext array.)
    """
    occ: list[list[int]] = [[] for _ in range(26)]
    for idx, (_, cidx) in enumerate(letters_meta):
        occ[cidx].append(idx)
    return occ


def _decrypt_letters_only(letters_meta: list[tuple[int, int]], mapping: list[str]) -> list[str]:
    out: list[str] = []
    for _, cidx in letters_meta:
        out.append(mapping[cidx])
    return out


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


def _update_full_for_cipher_letter(
    ciphertext: str,
    out_chars: list[str],
    positions: list[int],
    cases: list[bool],
    letters_meta: list[tuple[int, int]],
    mapping: list[str],
    occ: list[list[int]],
    cidx: int,
) -> None:
    """
    Update full plaintext chars for occurrences of cipher letter cidx.
    """
    plain_up = mapping[cidx]
    for meta_idx in occ[cidx]:
        pos = positions[meta_idx]
        out_chars[pos] = plain_up if cases[meta_idx] else plain_up.lower()


def _extract_words_upper(text: str) -> list[str]:
    return _WORD_RE.findall(text.upper())


def _extra_word_bonus(fulltext: str, *, n_letters: int) -> float:
    """
    Additional bounded word guidance using a larger word set.
    Scales up on short ciphertexts.
    """
    words = _extract_words_upper(fulltext)
    if not words:
        return 0.0

    hits = sum(1 for w in words if w in _EXTRA_WORDS)
    misses = len(words) - hits

    bonus = 3.0 * hits - 0.25 * misses

    # Scale up a bit on short texts
    if n_letters < 120:
        bonus *= 2.0
    elif n_letters < 200:
        bonus *= 1.4

    # Bound so quadgrams still dominate globally
    if bonus > 80.0:
        bonus = 80.0
    if bonus < -30.0:
        bonus = -30.0
    return bonus


def _crib_bonus(fulltext: str, *, n_letters: int) -> float:
    """
    Strong but bounded bonus for exact anchor-word hits.
    This is what pushes YHEN->WHEN, CAR->YOU type near-misses over the line on short inputs.
    """
    words = _extract_words_upper(fulltext)
    if not words:
        return 0.0

    hits = sum(1 for w in words if w in _CRIB_WORDS)

    # More weight when short; taper when longer
    if n_letters < 120:
        per_hit = 35.0
        cap = 220.0
    elif n_letters < 220:
        per_hit = 22.0
        cap = 160.0
    else:
        per_hit = 10.0
        cap = 90.0

    bonus = hits * per_hit
    if bonus > cap:
        bonus = cap
    return bonus


def crack_substitution_anneal(
    ciphertext: str,
    *,
    restarts: int = 40,
    steps: int = 35000,
    temp_start: float = 20.0,
    temp_end: float = 0.2,
    seed: int | None = None,
    max_seconds: float | None = None,
    full_score_every: int = 150,
) -> list[tuple[float, list[str], str]]:
    """
    Two-tier scoring:
      - accept moves by FAST quadgram score on letters-only plaintext
      - track best by FULL score = quadgram(letters-only) + word bonuses (core + extra + crib)

    Then steepest-ascent polish using the FULL score.
    """
    start = time.perf_counter()
    rng = random.Random(seed)
    use_quad = get_quadgram_scorer() is not None

    letters_meta, positions, cases = _letters_meta(ciphertext)
    n_letters = len(letters_meta)
    if n_letters < 60:
        return []

    occ = _build_occ(letters_meta)

    def time_up() -> bool:
        return max_seconds is not None and (time.perf_counter() - start) >= max_seconds

    def fast_score(pt_letters: list[str]) -> float:
        if use_quad:
            return quadgram_score("".join(pt_letters))  # already A-Z
        # fallback (weaker)
        return english_likeness_score("".join(pt_letters)) * 10.0

    def full_score(pt_letters: list[str], pt_full_chars: list[str]) -> float:
        fulltext = "".join(pt_full_chars)
        if use_quad:
            base = quadgram_score("".join(pt_letters))
            return (
                base
                + core_word_bonus(fulltext)
                + _extra_word_bonus(fulltext, n_letters=n_letters)
                + _crib_bonus(fulltext, n_letters=n_letters)
            )
        return english_likeness_score(fulltext) * 10.0

    best_across: list[tuple[float, list[str], str]] = []

    for r in range(restarts):
        if time_up():
            break

        # Init strategy: several freq-based (perturbed) then random
        if r < 10:
            mapping = _initial_mapping_freq(ciphertext)
            swaps = 10 + 14 * r
            for _ in range(swaps):
                i, j = rng.randrange(26), rng.randrange(26)
                if i != j:
                    mapping[i], mapping[j] = mapping[j], mapping[i]
        else:
            mapping = _random_mapping(rng)

        pt_letters = _decrypt_letters_only(letters_meta, mapping)
        pt_full_chars = _apply_to_fulltext(ciphertext, positions, cases, mapping, letters_meta)

        cur_fast = fast_score(pt_letters)
        cur_full = full_score(pt_letters, pt_full_chars)

        best_map = mapping[:]
        best_letters = pt_letters[:]
        best_full_chars = pt_full_chars[:]
        best_full = cur_full

        for step in range(steps):
            if time_up():
                break

            t = temp_start * ((temp_end / temp_start) ** (step / max(1, steps - 1)))

            i, j = rng.randrange(26), rng.randrange(26)
            if i == j:
                continue

            # swap plaintext assignments for cipher letters i and j
            mapping[i], mapping[j] = mapping[j], mapping[i]

            # update letters-only plaintext (incremental)
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

                # update full plaintext incrementally only when accepted
                _update_full_for_cipher_letter(ciphertext, pt_full_chars, positions, cases, letters_meta, mapping, occ, i)
                _update_full_for_cipher_letter(ciphertext, pt_full_chars, positions, cases, letters_meta, mapping, occ, j)

                if step % full_score_every == 0:
                    cur_full = full_score(pt_letters, pt_full_chars)
                    if cur_full > best_full:
                        best_full = cur_full
                        best_map = mapping[:]
                        best_letters = pt_letters[:]
                        best_full_chars = pt_full_chars[:]
            else:
                # revert swap
                mapping[i], mapping[j] = mapping[j], mapping[i]
                for idx in occ[i]:
                    pt_letters[idx] = mapping[i]
                for idx in occ[j]:
                    pt_letters[idx] = mapping[j]

        # End-of-restart full check
        cur_full = full_score(pt_letters, pt_full_chars)
        if cur_full > best_full:
            best_full = cur_full
            best_map = mapping[:]
            best_letters = pt_letters[:]
            best_full_chars = pt_full_chars[:]

        # --- Steepest-ascent polish on FULL score ---
        def polish(
            m: list[str],
            letters: list[str],
            full_chars: list[str],
            best_s: float,
        ) -> tuple[float, list[str], list[str], list[str]]:
            max_passes = 6
            for _ in range(max_passes):
                if time_up():
                    break

                improved = False
                best_pair: Optional[tuple[int, int]] = None
                best_pair_score = best_s

                # try all swaps; keep the best this pass
                for i in range(26):
                    for j in range(i + 1, 26):
                        if time_up():
                            break

                        m[i], m[j] = m[j], m[i]

                        for idx in occ[i]:
                            letters[idx] = m[i]
                        for idx in occ[j]:
                            letters[idx] = m[j]

                        _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, i)
                        _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, j)

                        s = full_score(letters, full_chars)
                        if s > best_pair_score:
                            best_pair_score = s
                            best_pair = (i, j)
                            improved = True

                        # revert
                        m[i], m[j] = m[j], m[i]
                        for idx in occ[i]:
                            letters[idx] = m[i]
                        for idx in occ[j]:
                            letters[idx] = m[j]
                        _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, i)
                        _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, j)

                    if time_up():
                        break

                if not improved or best_pair is None:
                    break

                i, j = best_pair
                m[i], m[j] = m[j], m[i]
                for idx in occ[i]:
                    letters[idx] = m[i]
                for idx in occ[j]:
                    letters[idx] = m[j]
                _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, i)
                _update_full_for_cipher_letter(ciphertext, full_chars, positions, cases, letters_meta, m, occ, j)

                best_s = best_pair_score

            return best_s, m, letters, full_chars

        best_full, best_map, best_letters, best_full_chars = polish(
            best_map, best_letters, best_full_chars, best_full
        )

        best_across.append((best_full, best_map[:], "".join(best_full_chars)))

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
        found = crack_substitution_anneal(ciphertext)
        results: list[SolveResult] = []
        for s, mapping, pt in found:
            results.append(
                SolveResult(
                    cipher_name=self.name,
                    plaintext=pt,
                    key=_mapping_to_keystring(mapping),
                    score=float(s),
                    confidence=0.0,  # registry will recompute confidence
                    notes="Simulated annealing mono-sub (fast quadgram accept + full-score tracking + anchor-word polish)",
                    meta={"fitness": "quadgram" if get_quadgram_scorer() else "fallback"},
                )
            )
        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(SubstitutionCipher(), should_try=_should_try)
