from __future__ import annotations

from collections import Counter

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import chi_squared_english, plaintext_fitness
from ciphercracker.classical.common import norm_key_alpha, shift_char, ALPHABET
from ciphercracker.classical.polyalphabetic.gating import should_try_vigenere


def _vigenere_decrypt(text: str, key: str) -> str:
    k = norm_key_alpha(key)
    if not k:
        raise ValueError("Vigenère key must contain at least one A-Z letter.")
    shifts = [ord(ch) - ord("A") for ch in k]

    out = []
    j = 0
    for ch in text:
        if ch.isalpha():
            up = ch.upper()
            if up in ALPHABET:
                shift = shifts[j % len(shifts)]
                plain = shift_char(up, -shift)  # decrypt: shift backwards
                out.append(plain if ch.isupper() else plain.lower())
                j += 1
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _only_az(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch in ALPHABET)


def _avg_ioc_for_keylen(az: str, keylen: int) -> float:
    if keylen <= 0:
        return 0.0
    cols = [az[i::keylen] for i in range(keylen)]
    iocs = []
    for col in cols:
        n = len(col)
        if n < 2:
            continue
        counts = Counter(col)
        num = sum(c * (c - 1) for c in counts.values())
        den = n * (n - 1)
        iocs.append(num / den if den else 0.0)
    return sum(iocs) / len(iocs) if iocs else 0.0


def _reduce_repeating_key(key: str) -> str:
    """
    If a key is a perfect repetition of a shorter pattern, reduce it.
    Example: CYBERCYBER -> CYBER
    """
    k = norm_key_alpha(key)
    if not k:
        return key

    for p in range(1, len(k) // 2 + 1):
        if len(k) % p != 0:
            continue
        base = k[:p]
        if base * (len(k) // p) == k:
            return base

    return k


def _best_folded_prefix_key(ciphertext: str, key_full: str) -> tuple[str, float]:
    """
    Even if key_full isn't a perfect repetition, try folding it to a shorter prefix.
    For each p < len(key_full), try key = key_full[:p] and keep whichever gives best plaintext_fitness.

    This is the key fix for cases like:
      CYBERCYCRR  -> best prefix likely CYBER
    """
    k = norm_key_alpha(key_full)
    if not k or len(k) < 2:
        pt = _vigenere_decrypt(ciphertext, key_full)
        return key_full, plaintext_fitness(pt)

    best_key = k
    best_fit = plaintext_fitness(_vigenere_decrypt(ciphertext, k))

    # Try all shorter prefix lengths.
    # (You can restrict this to divisors if you want speed, but this is still cheap: <= 24 tries.)
    for p in range(1, len(k)):
        cand = k[:p]
        pt = _vigenere_decrypt(ciphertext, cand)
        fit = plaintext_fitness(pt)
        if fit > best_fit:
            best_fit = fit
            best_key = cand

    return best_key, best_fit


class VigenereCipher:
    name = "vigenere"

    def decrypt(self, ciphertext: str, key: str) -> str:
        return _vigenere_decrypt(ciphertext, key)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        az = _only_az(ciphertext)
        n = len(az)
        if n < 20:
            return []

        # --- key length candidates ---
        max_klen = min(24, max(2, n // 3))
        lens: list[tuple[float, float, int]] = []
        for klen in range(2, max_klen + 1):
            cols = [az[i::klen] for i in range(klen)]
            if min(len(c) for c in cols) < 6:
                continue
            avg_ioc = _avg_ioc_for_keylen(az, klen)

            # Stronger penalty for longer keys so fundamentals (e.g., 5) beat multiples (e.g., 10)
            score = avg_ioc - (0.006 * klen)
            lens.append((score, avg_ioc, klen))

        if not lens:
            return []

        lens.sort(reverse=True)
        top_lens = [klen for _, _, klen in lens[:10]]

        results: list[SolveResult] = []

        # --- beam search settings ---
        TOP_SHIFTS_PER_COL = 4
        BEAM_WIDTH = 250

        for klen in top_lens:
            cols = [az[i::klen] for i in range(klen)]

            # Precompute chi-squared table per column/shift.
            chi_table: list[list[float]] = []
            for c in cols:
                row = []
                for shift in range(26):
                    dec = "".join(shift_char(ch, -shift) for ch in c)
                    row.append(chi_squared_english(dec))
                chi_table.append(row)

            def top_shifts_for_column(col_idx: int, top_k: int) -> list[int]:
                row = chi_table[col_idx]
                ranked = sorted(range(26), key=lambda s: row[s])
                return ranked[:top_k]

            col_shift_options = [
                top_shifts_for_column(i, TOP_SHIFTS_PER_COL) for i in range(klen)
            ]

            # Beam holds tuples: (score, key_shifts_list)
            beam: list[tuple[float, list[int]]] = [(0.0, [])]

            for col_idx in range(klen):
                new_beam: list[tuple[float, list[int]]] = []
                for score, partial in beam:
                    for shift in col_shift_options[col_idx]:
                        new_score = score - chi_table[col_idx][shift]
                        new_beam.append((new_score, partial + [shift]))

                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            # Evaluate final candidates from beam using true plaintext fitness.
            # Also try:
            #  - perfect-repeat reduction (CYBERCYBER -> CYBER)
            #  - best prefix folding (CYBERCYCRR -> CYBER if that scores better)
            for _, shifts in beam[:40]:
                key_full = "".join(chr(ord("A") + sh) for sh in shifts)

                candidate_keys: list[tuple[str, str]] = [(key_full, "raw")]

                key_rep = _reduce_repeating_key(key_full)
                if key_rep != key_full:
                    candidate_keys.append((key_rep, "reduced_repeat"))

                key_fold, _ = _best_folded_prefix_key(ciphertext, key_full)
                if key_fold != key_full and key_fold != key_rep:
                    candidate_keys.append((key_fold, "folded_prefix"))

                for key, kind in candidate_keys:
                    pt = _vigenere_decrypt(ciphertext, key)
                    fit = plaintext_fitness(pt)

                    results.append(
                        SolveResult(
                            cipher_name=self.name,
                            plaintext=pt,
                            key=key,
                            notes=f"Beam search keylen={klen} ({kind})",
                            meta={"key_length": klen, "fitness": fit, "postprocess": kind},
                        )
                    )

        results.sort(key=lambda r: r.meta.get("fitness", float("-inf")), reverse=True)
        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "polyalphabetic"}


register_plugin(VigenereCipher(), should_try=should_try_vigenere)
