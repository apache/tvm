# pylint: disable=wildcard-import, redefined-builtin
"""The Relay IR namespace containing the IR definition and compiler."""
from . import base
from . import ty
from . import expr
from . import env
from . import ir_pass
from . import ir_builder

# Root operators
from .op import Op
from .op.tensor import *
from .op.transform import *
from . import nn
from . import vision
from . import image

# Span
Span = base.Span

# Type
Type = ty.Type
TupleType = ty.TupleType
TensorType = ty.TensorType
Kind = ty.Kind
TypeParam = ty.TypeParam
TypeConstraint = ty.TypeConstraint
FuncType = ty.FuncType
TypeRelation = ty.TypeRelation
IncompleteType = ty.IncompleteType

# Expr
Constant = expr.Constant
Tuple = expr.Tuple
Var = expr.Var
GlobalVar = expr.GlobalVar
Param = expr.Param
Function = expr.Function
Call = expr.Call
Let = expr.Let
If = expr.If
