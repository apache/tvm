# pylint: disable=wildcard-import
"""The Relay IR namespace containing the IR definition and compiler."""
from . import base
from . import type as tpe
from . import expr
from . import to_tvm
from . import env
from . import ir_pass
from . import ir_builder
# Operators
from .op import Op
from .op.tensor import *

# Span
Span = base.Span

# Type
Type = tpe.Type
TensorType = tpe.TensorType
Kind = tpe.Kind
TypeParam = tpe.TypeParam
TypeConstraint = tpe.TypeConstraint
FuncType = tpe.FuncType

# Expr
Constant = expr.Constant
Tuple = expr.Tuple
LocalVar = expr.LocalVar
GlobalVar = expr.GlobalVar
Param = expr.Param
Function = expr.Function
Call = expr.Call
Let = expr.Let
If = expr.If
Var = LocalVar
