from __future__ import annotations

import math

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.features import analyze_text
from ciphercracker.classical.common import ALPHABET, modinv, parse_two_ints
from ciphercracker.core.utils import normalize_az
from ciphercracker.core.scoring import english_likeness_score


def _should_try(ct: str) -> bool:
    az = normalize_az(ct)
    if len(az) < 30:
        return False

    info = analyze_text(ct)
    if info.get("alpha_ratio", 0.0) < 0.75:
        return False

    # If it already looks quite English-like, affine brute force tends to add noise.
    # (Atbash/Caesar/Substitution will handle those cases better.)
    if english_likeness_score(ct) >= 55:
        return False

    return True


def _affine_decrypt(text: str, a: int, b: int) -> str:
    inv = modinv(a, 26)
    out = []
    for ch in text:
        if ch.isalpha():
            up = ch.upper()
            if up in ALPHABET:
                y = ord(up) - ord("A")
                x = (inv * (y - b)) % 26
                plain = chr(ord("A") + x)
                out.append(plain if ch.isupper() else plain.lower())
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


class AffineCipher:
    name = "affine"

    def decrypt(self, ciphertext: str, key: str) -> str:
        a, b = parse_two_ints(key)
        if math.gcd(a, 26) != 1:
            raise ValueError("Affine key 'a' must be coprime with 26 (e.g., 1,3,5,7,9,11,15,17,19,21,23,25).")
        return _affine_decrypt(ciphertext, a, b)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        out: list[SolveResult] = []
        for a in range(1, 26):
            if math.gcd(a, 26) != 1:
                continue
            for b in range(26):
                pt = _affine_decrypt(ciphertext, a, b)
                out.append(
                    SolveResult(
                        cipher_name=self.name,
                        plaintext=pt,
                        key=f"{a},{b}",
                        notes=f"Affine a={a}, b={b}",
                    )
                )
        return out

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(AffineCipher(), should_try=_should_try)
