from __future__ import annotations

def register_all() -> None:
    from .monoalphabetic import caesar, atbash, affine, substitution  # noqa: F401
    from .polyalphabetic import vigenere  # noqa: F401
    from .polyalphabetic import periodic_substitution  # noqa: F401
