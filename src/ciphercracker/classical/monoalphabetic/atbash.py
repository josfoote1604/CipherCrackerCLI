from __future__ import annotations

from ciphercracker.core.registry import register_plugin
from ciphercracker.core.results import SolveResult
from ciphercracker.core.features import analyze_text
from ciphercracker.classical.common import ALPHABET

_ATBASH = {ALPHABET[i]: ALPHABET[25 - i] for i in range(26)}


def _apply(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            up = ch.upper()
            if up in _ATBASH:
                mapped = _ATBASH[up]
                out.append(mapped if ch.isupper() else mapped.lower())
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def _should_try(text: str) -> bool:
    feats = analyze_text(text)
    return feats["alpha_ratio"] >= 0.60 and feats["length"] >= 4


class AtbashCipher:
    name = "atbash"

    def decrypt(self, ciphertext: str, key: str) -> str:
        # key unused; accept empty string for CLI consistency
        return _apply(ciphertext)

    def crack(self, ciphertext: str) -> list[SolveResult]:
        return [
            SolveResult(
                cipher_name=self.name,
                plaintext=_apply(ciphertext),
                key=None,
                notes="Atbash is keyless",
            )
        ]

    def fingerprint(self, ciphertext: str) -> dict:
        return {"family": "monoalphabetic"}


register_plugin(AtbashCipher(), should_try=_should_try)
