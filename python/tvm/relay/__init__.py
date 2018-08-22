"""The Relay IR namespace containing the IR definition and compiler."""
from . import base
from . import type as tpe
from . import make
from . import op

# Type
Type = tpe.Type
TensorType = tpe.TensorType
Kind = tpe.Kind
TypeParam = tpe.TypeParam
TypeConstraint = tpe.TypeConstraint
FuncType = tpe.FuncType
