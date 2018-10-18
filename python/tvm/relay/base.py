# pylint: disable=no-else-return, unidiomatic-typecheck
"""The base node types for the Relay language."""
from __future__ import absolute_import as _abs
from .._ffi.node import NodeBase, register_node as _register_tvm_node
from . import _make
from . import _expr

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


class RelayNode(NodeBase):
    def astext(self):
        """Get the text format of the expression.

        Returns
        -------
        text : str
            The text format of the expression.
        """
        return _expr._text_print(self)


@register_relay_node
class Span(RelayNode):
    def __init__(self, source, lineno, col_offset):
        self.__init_handle_by_constructor__(_make.Span, source, lineno, col_offset)
