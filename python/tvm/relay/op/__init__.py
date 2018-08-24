"""Relay core operators."""
# operator defs
from .tensor import *

# operator registry
from . import _tensor
from ..expr import Expr
from ..base import register_relay_node

@register_relay_node
class Op(Expr):
    pass
