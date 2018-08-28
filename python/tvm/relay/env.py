# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global environment storing everything needed to interpret or compile a Realy program."""
from typing import Union, List
from .base import register_relay_node, NodeBase
from . import _make
import tvm

@register_relay_node
class Environment(NodeBase):
    """The global Relay environment containing definitions,
       primitives, options, and more.
    """
    def __init__(self, funcs) -> None:
        self.__init_handle_by_constructor__(_make.Environment, funcs)
