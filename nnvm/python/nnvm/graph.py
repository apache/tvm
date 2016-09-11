# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys
import json
from ._base import _LIB
from ._base import c_array, c_str, nn_uint, py_str, string_types
from ._base import GraphHandle, SymbolHandle
from ._base import check_call
from .symbol import Symbol, Group as _Group


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

    def json_attr(self, key):
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
        check_call(_LIB.NNGraphGetJSONAttr(
            self.handle, c_str(key), ctypes.byref(ret), ctypes.byref(success)))
        if success.value != 0:
            json_str = py_str(ret.value)
            return json.loads(json_str)[1]
        else:
            return None

    def _set_symbol_list_attr(self, key, value):
        """Set the attribute of the graph.

        Parameters
        ----------
        key : string
            The key of the attribute
        value : value
            The any type that can be dumped to json
        type_name : string
            The typename registered on c++ side.
        """
        if isinstance(value, list):
            value = _Group(value)
        if not isinstance(value, Symbol):
            raise ValueError("value need to be grouped symbol")
        check_call(_LIB.NNGraphSetNodeEntryListAttr_(
            self.handle, c_str(key), value.handle))

    def _set_json_attr(self, key, value, type_name=None):
        """Set the attribute of the graph.

        Parameters
        ----------
        key : string
            The key of the attribute
        value : value
            The any type that can be dumped to json
        type_name : string
            The typename registered on c++ side.
        """
        if isinstance(value, string_types):
            type_name = 'str'
        elif type_name is None:
            raise ValueError("Need to specify type_name")
        json_value = json.dumps([type_name, value])
        check_call(_LIB.NNGraphSetJSONAttr(
            self.handle, c_str(key), c_str(json_value)))

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
        check_call(_LIB.NNGraphApplyPasses(self.handle, npass, cpass, ctypes.byref(ghandle)))
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
