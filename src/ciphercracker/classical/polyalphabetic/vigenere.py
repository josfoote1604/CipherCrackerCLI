from __future__ import annotations

from collections import Counter

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.scoring import chi_squared_english
from ciphercracker.classical.common import norm_key_alpha, shift_char, ALPHABET
from .gating import should_try_vigenere

from ciphercracker.core.features import analyze_text, ioc_scan
from ciphercracker.core.utils import normalize_az

def _has_periodicity_signal(ct: str, *, max_len: int, min_len: int, thresh: float) -> bool:
    az = normalize_az(ct)
    if len(az) < min_len:
        return False

    # Avoid wasting time on low-alpha / weird charsets
    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.70:
        return False

    # Look for k>1 with elevated average IoC
    top = ioc_scan(ct, max_len=max_len)[:8]
    return any(k >= 2 and val >= thresh for k, val in top)


# assuming your plugin class is named VigenerePlugin
def register() -> None:
    register_plugin(VigenereCipher(), should_try=should_try_vigenere)

def _should_try(ct: str) -> bool:
    az = normalize_az(ct)
    if len(az) < 60:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.70:
        return False

    top = ioc_scan(ct, max_len=12)[:8]
    return any(k >= 2 and val >= 0.072 for k, val in top)



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
                # decrypt: shift backwards
                plain = shift_char(up, -shift)
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


def _best_shift_for_column(col: str) -> int:
    """
    Find Caesar shift that best 'decrypts' the column into English distribution.
    For Vigenère, each column is Caesar-encrypted with some shift.
    """
    best_shift = 0
    best_chi = float("inf")
    for shift in range(26):
        # decrypt column by shifting backward
        dec = "".join(shift_char(ch, -shift) for ch in col)
        chi = chi_squared_english(dec)
        if chi < best_chi:
            best_chi = chi
            best_shift = shift
    return best_shift


class VigenereCipher:
    name = "vigenere"

    def decrypt(self, ciphertext: str, key: str) -> str:
        return _vigenere_decrypt(ciphertext, key)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        from ciphercracker.core.scoring import english_likeness_score

        az = _only_az(ciphertext)
        n = len(az)
        if n < 30:
            return []

        # --- key length candidates ---
        # Try more lengths, but avoid lengths that make columns too short.
        max_klen = min(24, max(2, n // 3))  # keep columns reasonably sized
        lens = []
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
        top_lens = [klen for _, _, klen in lens[:6]]

        results: list[SolveResult] = []

        # --- beam search settings ---
        TOP_SHIFTS_PER_COL = 4  # try top 3-5
        BEAM_WIDTH = 150  # keep best partial keys

        def top_shifts_for_column(col: str, top_k: int) -> list[int]:
            # rank shifts by chi-squared on that column
            scored = []
            for shift in range(26):
                dec = "".join(shift_char(ch, -shift) for ch in col)
                chi = chi_squared_english(dec)
                scored.append((chi, shift))
            scored.sort(key=lambda x: x[0])  # lower chi is better
            return [s for _, s in scored[:top_k]]

        for klen in top_lens:
            cols = [az[i::klen] for i in range(klen)]
            col_shift_options = [top_shifts_for_column(c, TOP_SHIFTS_PER_COL) for c in cols]

            # Beam holds tuples: (score, key_shifts_list)
            beam: list[tuple[float, list[int]]] = [(0.0, [])]

            for col_idx in range(klen):
                new_beam: list[tuple[float, list[int]]] = []
                for _, partial in beam:
                    for shift in col_shift_options[col_idx]:
                        candidate = partial + [shift]
                        # Build partial key (A=0, B=1, ...)
                        key = "".join(chr(ord("A") + s) for s in candidate)
                        pt = _vigenere_decrypt(ciphertext, key)

                        # Score using full plaintext (even partial key still decrypts every Nth letter correctly)
                        s = english_likeness_score(pt)

                        new_beam.append((s, candidate))

                # Keep best partials
                new_beam.sort(key=lambda x: x[0], reverse=True)
                beam = new_beam[:BEAM_WIDTH]

            # Evaluate final candidates from beam
            for s, shifts in beam[:12]:  # take top few final keys
                key = "".join(chr(ord("A") + sh) for sh in shifts)
                pt = _vigenere_decrypt(ciphertext, key)
                results.append(
                    SolveResult(
                        cipher_name=self.name,
                        plaintext=pt,
                        key=key,
                        notes=f"Beam search keylen={klen}",
                        meta={"key_length": klen},
                    )
                )

        return results

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "polyalphabetic"}


register_plugin(VigenereCipher(), should_try=_should_try)
