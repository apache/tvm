"""Common topi utilities"""
from __future__ import absolute_import as _abs
import tvm

def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int."""
    out_tuple = ()
    for elem in in_tuple:
        assert isinstance(elem, tvm.expr.IntImm)
        out_tuple = out_tuple + (elem.value, )
    return out_tuple
