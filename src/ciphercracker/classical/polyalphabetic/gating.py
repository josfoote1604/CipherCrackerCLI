from __future__ import annotations

from ciphercracker.core.features import ioc_scan
from ciphercracker.core.utils import normalize_az

def _has_periodicity_signal(ct: str, *, max_len: int, min_len: int, thresh: float) -> bool:
    az = normalize_az(ct)
    if len(az) < min_len:
        return False
    top = ioc_scan(ct, max_len=max_len)[:8]
    return any(k >= 2 and val >= thresh for k, val in top)

def should_try_vigenere(ct: str) -> bool:
    return _has_periodicity_signal(ct, max_len=12, min_len=60, thresh=0.072)

def should_try_periodic_sub(ct: str) -> bool:
    return _has_periodicity_signal(ct, max_len=12, min_len=120, thresh=0.073)
