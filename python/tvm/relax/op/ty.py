import typing

import numpy as np

from tvm import ir
from tvm import runtime as rt
from tvm import tir

from .. import expr as rx
from ..struct_info import StructInfo as _StructInfo

##### Containers #####

_ArrayItem = typing.TypeVar("_ArrayItem")
Array = typing.Union[typing.Sequence[_ArrayItem], ir.Array]

_OptionalItem = typing.TypeVar("_OptionalItem")
Optional = typing.Union[_OptionalItem, None]

_MapKey = typing.TypeVar("_MapKey")
_MapValue = typing.TypeVar("_MapValue")
Map = typing.Union[typing.Dict[_MapKey, _MapValue], ir.Map]

Union = typing.Union

##### PrimValues #####

Int = typing.Union[int, tir.IntImm, rx.PrimValue]
Float = typing.Union[int, float, tir.IntImm, tir.FloatImm, rx.PrimValue]
Bool = typing.Union[bool, tir.IntImm, rx.PrimValue]
PrimExpr = typing.Union[int, bool, float, tir.PrimExpr, rx.PrimValue]
IntPrimExpr = typing.Union[int, tir.PrimExpr, rx.PrimValue]
FloatPrimExpr = typing.Union[int, float, PrimExpr, rx.PrimValue]
BoolPrimExpr = typing.Union[bool, tir.PrimExpr, rx.PrimValue]
TIRVar = typing.Union[tir.Var]  # type: ignore
IntTIRVar = typing.Union[tir.Var]  # type: ignore
FloatTIRVar = typing.Union[tir.Var]  # type: ignore
BoolTIRVar = typing.Union[tir.Var]  # type: ignore

##### Tensor #####

Tensor = typing.Union[rx.Expr, np.ndarray, rt.NDArray]
IntTensor = typing.Union[rx.Expr, np.ndarray, rt.NDArray]
FloatTensor = typing.Union[rx.Expr, np.ndarray, rt.NDArray]
BoolTensor = typing.Union[rx.Expr, np.ndarray, rt.NDArray]

##### Misc #####

AnyRelaxExpr = typing.Union[rx.Expr]  # type: ignore
TupleExpr = typing.Union[rx.Tuple, Array[AnyRelaxExpr]]
Str = typing.Union[str, rt.String, tir.StringImm, rx.StringImm]
DType = typing.Union[None, str, rt.DataType, rx.DataTypeImm]
Shape = typing.Union[rx.ShapeExpr, Array[IntPrimExpr]]
Axis = typing.Union[Int]  # type: ignore
Axes = typing.Union[Array[Axis], None]
GlobalVar = typing.Union[ir.GlobalVar]  # type: ignore
ExternFunc = typing.Union[rx.ExternFunc, str]
IndexMap = typing.Union[typing.Callable, tir.IndexMap]
StructInfo = typing.Union[_StructInfo, Array[_StructInfo]]
