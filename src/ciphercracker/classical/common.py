from __future__ import annotations

import math
import string
from typing import Dict, Tuple

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
A_ORD = ord("A")
Z_ORD = ord("Z")

def is_az(ch: str) -> bool:
    o = ord(ch)
    return A_ORD <= o <= Z_ORD

def norm_key_alpha(key: str) -> str:
    """Uppercase and keep only A-Z."""
    return "".join(ch for ch in key.upper() if "A" <= ch <= "Z")

def shift_char(ch: str, shift: int) -> str:
    """Shift one A-Z character by 'shift' (can be negative)."""
    idx = (ord(ch) - A_ORD + shift) % 26
    return chr(A_ORD + idx)

def shift_text(text: str, shift: int) -> str:
    """Caesar shift; preserves non-letters; preserves case."""
    out = []
    for ch in text:
        if ch.isalpha():
            up = ch.upper()
            if "A" <= up <= "Z":
                shifted = shift_char(up, shift)
                out.append(shifted if ch.isupper() else shifted.lower())
            else:
                out.append(ch)
        else:
            out.append(ch)
    return "".join(out)

def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a: int, m: int) -> int:
    """Modular inverse of a under mod m; raises ValueError if none."""
    a %= m
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for a={a} mod {m}.")
    return x % m

def parse_two_ints(key: str) -> tuple[int, int]:
    """
    Parse keys like: "5,8" or "5:8" or "5 8"
    Returns (a, b).
    """
    raw = key.strip().replace(":", ",").replace(" ", ",")
    parts = [p for p in raw.split(",") if p]
    if len(parts) != 2:
        raise ValueError("Expected key format like 'a,b' (e.g., '5,8').")
    return int(parts[0]), int(parts[1])

def parse_substitution_key(key: str) -> Dict[str, str]:
    """
    Accept either:
      1) 26-letter string, meaning: cipher A->key[0], cipher B->key[1], ...
         (i.e., key is a ciphertext-alphabet mapped to plaintext letters)
      2) pair mapping like: "A:E,B:T,C:A" (cipher:plain pairs)
    Returns dict mapping cipher uppercase letter -> plaintext uppercase letter.
    """
    k = key.strip().upper()

    # Case 1: 26-letter alphabet
    only_letters = "".join(ch for ch in k if ch in ALPHABET)
    if len(only_letters) == 26:
        mapping = {}
        for i, c in enumerate(ALPHABET):
            mapping[c] = only_letters[i]
        if len(set(mapping.values())) != 26:
            raise ValueError("26-letter key must be a permutation with no repeats.")
        return mapping

    # Case 2: pairs
    mapping: Dict[str, str] = {}
    items = [x.strip() for x in k.split(",") if x.strip()]
    for item in items:
        if ":" not in item:
            raise ValueError("Pair mapping must look like 'A:E,B:T,...'")
        ciph, plain = [p.strip() for p in item.split(":", 1)]
        if len(ciph) != 1 or len(plain) != 1 or ciph not in ALPHABET or plain not in ALPHABET:
            raise ValueError(f"Bad pair '{item}'. Use single letters like 'A:E'.")
        mapping[ciph] = plain

    if not mapping:
        raise ValueError("Empty substitution key.")
    # It’s okay if partial for decrypt? Here we require full for deterministic decrypt.
    if len(mapping) != 26:
        raise ValueError("Substitution decrypt expects a full 26-letter mapping.")
    if len(set(mapping.values())) != 26:
        raise ValueError("Substitution mapping must not repeat plaintext letters.")
    return mapping
