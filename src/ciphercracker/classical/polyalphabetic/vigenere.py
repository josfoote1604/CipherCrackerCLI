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


class VigenereCipher:
    name = "vigenere"

    def decrypt(self, ciphertext: str, key: str) -> str:
        return _vigenere_decrypt(ciphertext, key)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        az = _only_az(ciphertext)
        n = len(az)
        # Be a bit more permissive for short-but-usable samples
        if n < 20:
            return []

        # --- key length candidates ---
        # Try more lengths, but avoid lengths that make columns too short.
        max_klen = min(24, max(2, n // 3))  # keep columns reasonably sized
        lens: list[tuple[float, float, int]] = []
        for klen in range(2, max_klen + 1):
            cols = [az[i::klen] for i in range(klen)]
            # Require each column to have at least 6 chars (tweakable)
            if min(len(c) for c in cols) < 6:
                continue
            avg_ioc = _avg_ioc_for_keylen(az, klen)
            # Mild penalty for longer keys (helps prevent huge-key overfit)
            score = avg_ioc - (0.002 * klen)
            lens.append((score, avg_ioc, klen))

        if not lens:
            return []

        lens.sort(reverse=True)
        # Evaluate more candidate key lengths (helps on shorter texts)
        top_lens = [klen for _, _, klen in lens[:10]]

        results: list[SolveResult] = []

        # --- beam search settings ---
        TOP_SHIFTS_PER_COL = 4   # try top 3-5
        BEAM_WIDTH = 250         # keep best partial keys; a bit wider is safer

        for klen in top_lens:
            cols = [az[i::klen] for i in range(klen)]

            # Precompute chi-squared table per column/shift.
            # This is the critical fix: score partial keys by column evidence,
            # not by decrypting the entire text with an incomplete key.
            chi_table: list[list[float]] = []
            for c in cols:
                row = []
                for shift in range(26):
                    dec = "".join(shift_char(ch, -shift) for ch in c)
                    row.append(chi_squared_english(dec))
                chi_table.append(row)

            def top_shifts_for_column(col_idx: int, top_k: int) -> list[int]:
                row = chi_table[col_idx]
                ranked = sorted(range(26), key=lambda s: row[s])  # lower chi is better
                return ranked[:top_k]

            col_shift_options = [
                top_shifts_for_column(i, TOP_SHIFTS_PER_COL) for i in range(klen)
            ]

            # Beam holds tuples: (score, key_shifts_list)
            # Higher is better => use negative chi-squared (lower chi -> higher score)
            beam: list[tuple[float, list[int]]] = [(0.0, [])]

            for col_idx in range(klen):
                new_beam: list[tuple[float, list[int]]] = []
                for score, partial in beam:
                    for shift in col_shift_options[col_idx]:
                        new_score = score - chi_table[col_idx][shift]
                        new_beam.append((new_score, partial + [shift]))

                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            # Evaluate final candidates from beam using true plaintext fitness
            # (quadgrams + small tie-breakers).
            for _, shifts in beam[:40]:
                key = "".join(chr(ord("A") + sh) for sh in shifts)
                pt = _vigenere_decrypt(ciphertext, key)
                fit = plaintext_fitness(pt)

                results.append(
                    SolveResult(
                        cipher_name=self.name,
                        plaintext=pt,
                        key=key,
                        notes=f"Beam search keylen={klen}",
                        meta={"key_length": klen, "fitness": fit},
                    )
                )

        # Sort best-first so the CLI ranks these sensibly vs other ciphers too.
        results.sort(key=lambda r: r.meta.get("fitness", float("-inf")), reverse=True)
        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "polyalphabetic"}


# Single, consistent plugin registration (avoid duplicate registrations)
register_plugin(VigenereCipher(), should_try=should_try_vigenere)
