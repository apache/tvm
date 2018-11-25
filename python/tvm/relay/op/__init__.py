#pylint: disable=wildcard-import, redefined-builtin
"""Relay core operators."""
# operator defs
from .op import get, register, register_schedule, register_compute, Op

# Operators
from .reduce import *
from .tensor import *
from .transform import *
from . import nn
from . import image
from . import vision

# operator registry
from . import _tensor
from . import _transform
from . import _reduce
from ..expr import Expr
from ..base import register_relay_node


def _register_op_make():
    from . import _make
    from .. import expr
    expr._op_make = _make

_register_op_make()
