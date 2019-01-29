"""Common x86 related utilities"""
from __future__ import absolute_import as _abs
import tvm

def get_fp32_len():
    fp32_vec_len = 8
    target = tvm.target.current_target()
    if target is not None:
        for opt in target.options:
            if opt == '-mcpu=skylake-avx512':
                fp32_vec_len = 16
    return fp32_vec_len

def get_max_power2_factor(n, max_value=None):
    x = 1
    while n % 2 == 0:
        if max_value is not None and max_value < x * 2:
            break
        x *= 2
        n /= 2
    return x
