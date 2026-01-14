from __future__ import annotations

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.features import analyze_text
from ciphercracker.classical.common import shift_text

from ciphercracker.core.registry import register_plugin

def register() -> None:
    register_plugin(CaesarCipher())

def _should_try(text: str) -> bool:
    feats = analyze_text(text)
    # Caesar is primarily alphabetic; allow some punctuation/spaces.
    return feats["alpha_ratio"] >= 0.60 and feats["length"] >= 4


class CaesarCipher:
    name = "caesar"

    def decrypt(self, ciphertext: str, key: str) -> str:
        try:
            k = int(key) % 26
        except ValueError as e:
            raise ValueError("Caesar key must be an integer 0..25.") from e
        # Decrypt means shift backwards by k
        return shift_text(ciphertext, -k)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        out: list[SolveResult] = []
        for k in range(26):
            pt = shift_text(ciphertext, -k)
            out.append(
                SolveResult(
                    cipher_name=self.name,
                    plaintext=pt,
                    key=str(k),
                    notes=f"Caesar shift {k}",
                )
            )
        return out

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(CaesarCipher(), should_try=_should_try)
