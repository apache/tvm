# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, wildcard-import
"""A global environment storing everything needed to interpret or compile a Realy program."""
from typing import Union, List
from .base import register_relay_node, NodeBase
from . import _make
from . import _env
import tvm

@register_relay_node
class Environment(NodeBase):
    """The global Relay environment containing definitions,
       primitives, options, and more.
    """
    def __init__(self, funcs) -> None:
        self.__init_handle_by_constructor__(_make.Environment, funcs)
    
    def add(self, var, func) -> None:
        if isinstance(var, str):
            var = _env.Environment_GetGlobalVar(self, var)

        _env.Environment_Add(self, var, func)
    
    def merge(self, other):
        return _env.Environment_Merge(self, other)
    
    def lookup(self, var):
        if isinstance(var, str):
            return _env.Environment_Lookup_str(self, var)
        else:
            return _env.Environment_Lookup(self, var)
