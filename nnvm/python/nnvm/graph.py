# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments, too-many-lines
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import json
from ._base import _LIB
from ._base import c_array, c_str, nn_uint, py_str, string_types
from ._base import GraphHandle, SymbolHandle
from ._base import check_call
from .symbol import Symbol, Group as _Group

class GraphIndex(object):
    """Index for quickly accessing graph attributes.

    Parameters
    ----------
    graph : Graph
        The graph to create index.
    """
    def __init__(self, graph):
        jgraph = json.loads(create(graph).apply("SaveJSON").json_attr("json"))
        self.nodes = jgraph["nodes"]
        self.entry_ptr = jgraph["node_row_ptr"]
        self._name2nodeid = {n["name"]: i for i, n in enumerate(self.nodes)}
        self.input_names = graph.symbol.list_input_names()
        self.output_entries = jgraph["heads"]

    @property
    def num_nodes(self):
        """Number of nodes in graph."""
        return len(self.entry_ptr) - 1

    @property
    def num_node_entries(self):
        """Number of nodes in graph."""
        return self.entry_ptr[-1]

    def node_id(self, key):
        """Get the node index for a given key.

        Parameters
        ----------
        key : str or int
            The node key or index

        Returns
        -------
        index : int
            The entry index
        """
        return self._name2nodeid[key]

    def entry_id(self, key, value_index=0):
        """Get the entry id of a node entry.

        Parameters
        ----------
        key : str or int
            The node key or index

        value_index : int
            The value index of output

        Returns
        -------
        index : int
            The entry index
        """
        if isinstance(key, (list, tuple)):
            if len(key) != 3:
                raise ValueError("Expect entry index to be tuple of 3 elems")
            key, value_index, _ = key
        idx = self.node_id(key) if isinstance(key, str) else key
        assert value_index < self.entry_ptr[idx + 1]
        return self.entry_ptr[idx] + value_index



class Graph(object):
    """Graph is the graph object that can be used to apply optimization pass.

    It contains additional graphwise attribute besides the internal symbol.
    """
    _tvm_tcode = 17

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : GraphHandle
            the handle to the underlying C++ Graph
        """
        self.handle = handle
        self._index = None

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
    def _tvm_handle(self):
        return self.handle.value

    @property
    def symbol(self):
        shandle = SymbolHandle()
        check_call(_LIB.NNGraphGetSymbol(self.handle, ctypes.byref(shandle)))
        return Symbol(shandle)

    @property
    def index(self):
        if not self._index:
            self._index = GraphIndex(self)
        return self._index

    def apply(self, passes):
        """Apply passes to the graph

        Parameters
        ----------
        passes : str or list of str
            The passes to be applied

        Returns
        -------
        g : Graph
            The transformed graph.
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
