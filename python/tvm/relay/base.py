# pylint: disable=no-else-return, unidiomatic-typecheck
"""The base node types for the Relay language."""
from __future__ import absolute_import as _abs
from typing import Union
from .._ffi.node import NodeBase, register_node as _register_tvm_node

NodeBase = NodeBase

def register_relay_node(type_key=None):
    """register relay node type

    Parameters
    ----------
    type_key : str or cls
        The type key of the node
    """
    if not isinstance(type_key, str):
        return _register_tvm_node(
            "relay." + type_key.__name__)(type_key)
    return _register_tvm_node(type_key)


@register_relay_node
class Span(NodeBase):
    source: "FileSource"
    lineno: int
    col_offset: int
