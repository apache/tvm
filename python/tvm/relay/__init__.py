# pylint: disable=wildcard-import, redefined-builtin, invalid-name
"""The Relay IR namespace containing the IR definition and compiler."""
from __future__ import absolute_import
from ..api import register_func
from . import base
from . import ty
from . import expr
from . import module
from . import ir_pass
from .build_module import build, create_executor

# Root operators
from .op import Op
from .op.reduce import *
from .op.tensor import *
from .op.transform import *
from . import nn
from . import vision
from . import image
from . import backend

from .scope_builder import ScopeBuilder

# Span
Span = base.Span

# Env
Module = module.Module

# Type
Type = ty.Type
TupleType = ty.TupleType
TensorType = ty.TensorType
Kind = ty.Kind
TypeVar = ty.TypeVar
TypeConstraint = ty.TypeConstraint
FuncType = ty.FuncType
TypeRelation = ty.TypeRelation
IncompleteType = ty.IncompleteType
scalar_type = ty.scalar_type

# Expr
Constant = expr.Constant
Tuple = expr.Tuple
Var = expr.Var
GlobalVar = expr.GlobalVar
Function = expr.Function
Call = expr.Call
Let = expr.Let
If = expr.If
TupleGetItem = expr.TupleGetItem


# helper functions
var = expr.var
const = expr.const


# pylint: disable=unused-argument
@register_func("relay.debug")
def _debug(*args):
    import pdb
    pdb.set_trace()
