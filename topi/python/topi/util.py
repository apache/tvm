"""Common topi utilities"""
from __future__ import absolute_import as _abs
import tvm

def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm, returns tuple of int.

    Parameters
    ----------
    in_tuple : tuple of tvm.expr.IntImm
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    out_tuple = ()
    for elem in in_tuple:
        if not isinstance(elem, tvm.expr.IntImm):
            raise ValueError("Element of input tuple should be IntImm")
        out_tuple = out_tuple + (elem.value, )
    return out_tuple

def is_output(op, schedule):
    """Determines whether op is the last stage of schedule."""
    return op in schedule.outputs
