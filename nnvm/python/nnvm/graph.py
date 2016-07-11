# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys
from .base import _LIB
from .base import c_array, c_str, nn_uint, py_str, string_types
from .base import GraphHandle, SymbolHandle
from .base import check_call
from .symbol import Symbol

class Graph(object):
    """Graph is the graph object that can be used to apply optimization pass.
    It contains additional graphwise attribute besides the internal symbol.

    """

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : GraphHandle
            the handle to the underlying C++ Graph
        """
        self.handle = handle

    def __del__(self):
        check_call(_LIB.NNGraphFree(self.handle))

    def attr(self, key):
        """Get attribute string from the graph.

        Parameters
        ----------
        key : str
            The key to get attribute from.

        Returns
        -------
        value : str
            The attribute value of the key, returns None if attribute do not exist.
        """
        ret = ctypes.c_char_p()
        success = ctypes.c_int()
        check_call(_LIB.NNGraphGetStrAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            return py_str(ret.value)
        else:
            return None

    def _set_attr(self, **kwargs):
        """Set the attribute of the symbol.

        Parameters
        ----------
        **kwargs
            The attributes to set
        """
        for k, v in kwargs.items():
            check_call(_LIB.NNGraphSetStrAttr(
                self.handle, c_str(k), c_str(v)))

    @property
    def symbol(self):
        shandle = SymbolHandle()
        check_call(_LIB.NNGraphGetSymbol(self.handle, ctypes.byref(shandle)))
        return Symbol(shandle)

    def apply(self, passes):
        """Apply passes to the graph

        Parameters
        ----------

        """
        if isinstance(passes, string_types):
            passes = [passes]
        cpass = c_array(ctypes.c_char_p, [c_str(key) for key in passes])
        ghandle = GraphHandle()
        npass = nn_uint(len(passes))
        check_call(_LIB.NNGraphApplyPass(self.handle, npass, cpass, ctypes.byref(ghandle)))
        return Graph(ghandle)


def create(symbol):
    """Create a new graph from symbol.

    Parameters
    ----------
    symbol : Symbol
        The symbolic graph used to create Graph object.

    Returns
    -------
    graph : Graph
        A generated new graph object.
    """
    ghandle = GraphHandle()
    check_call(_LIB.NNGraphCreate(
        symbol.handle, ctypes.byref(ghandle)))
    return Graph(ghandle)
