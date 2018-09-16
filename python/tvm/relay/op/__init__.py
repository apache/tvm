#pylint: disable=wildcard-import
"""Relay core operators."""
# operator defs
from .op import get, register, Op

# Operators
from .tensor import *

# operator registry
from . import _tensor
from ..expr import Expr
from ..base import register_relay_node
