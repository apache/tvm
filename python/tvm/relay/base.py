# pylint: disable=no-else-return, unidiomatic-typecheck
"""The base node types for the Relay language."""
from __future__ import absolute_import as _abs
from .._ffi.node import NodeBase, register_node as _register_tvm_node
from . import _make
from . import _expr
from . import _base

NodeBase = NodeBase

def register_relay_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return _register_tvm_node(
            "relay." + type_key.__name__)(type_key)
    return _register_tvm_node(type_key)


def register_relay_attr_node(type_key=None):
    """Register a Relay attribute node.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return _register_tvm_node(
            "relay.attrs." + type_key.__name__)(type_key)
    return _register_tvm_node(type_key)


class RelayNode(NodeBase):
    """Base class of all Relay nodes."""
    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[relay.Expr->str]
            Optional annotate function to provide additional
            information in the comment block.

        Note
        ----
        The metadata section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.

        Returns
        -------
        text : str
            The text format of the expression.
        """
        return _expr.RelayPrint(self, show_meta_data, annotate)

    def set_span(self, span):
        _base.set_span(self, span)

    def __str__(self):
        return self.astext(show_meta_data=False)


@register_relay_node
class Span(RelayNode):
    """Specifies a location in a source program."""

    def __init__(self, source, lineno, col_offset):
        self.__init_handle_by_constructor__(_make.Span, source, lineno, col_offset)

@register_relay_node
class SourceName(RelayNode):
    """A identifier for a source location"""

    def __init__(self, name):
        self.__init_handle_by_constructor__(_make.SourceName, name)

@register_relay_node
class Id(NodeBase):
    """Unique identifier(name) used in Var.
       Guaranteed to be stable across all passes.
    """
    def __init__(self):
        raise RuntimeError("Cannot directly construct Id")
