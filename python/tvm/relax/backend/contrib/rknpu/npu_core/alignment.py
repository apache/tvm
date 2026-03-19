# SPDX-License-Identifier: MIT
"""Alignment utilities shared across NPU compiler modules."""


def align_up(val: int, align: int) -> int:
    """Round *val* up to the next multiple of *align*."""
    return ((val + align - 1) // align) * align


def pad_m(M: int) -> int:
    """Pad M to the next multiple of 4 (NPU processes height in groups of 4)."""
    if M <= 1:
        return M
    return ((M + 3) // 4) * 4
