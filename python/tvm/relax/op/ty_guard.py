import typing
from abc import ABC, abstractmethod

import numpy as np

from tvm import ir
from tvm import runtime as rt
from tvm import tir

from .. import expr as rx
from .. import struct_info as sinfo
from . import ty


class _TypeChecker(ABC):
    @abstractmethod
    def __call__(
        self,
        arg: typing.Any,
    ) -> typing.Any:
        raise NotImplementedError


def check(
    index: int,
    name: str,
    checker: _TypeChecker,
    arg: typing.Any,
) -> typing.Any:
    try:
        return checker(arg)
    except Exception as error:
        raise TypeError(f"Invalid type on argument {index} ({name}): " + str(error).strip())


##### Containers #####


class Array(_TypeChecker):  # pylint: disable=too-few-public-methods
    elem: _TypeChecker
    length: typing.List[int]
    restrict: bool

    def __init__(
        self,
        elem: _TypeChecker,
        length: typing.List[int],
        restrict: bool = False,
    ):
        super().__init__()
        self.elem = elem
        self.length = length
        self.restrict = restrict

    def __call__(self, arg: ty.Array) -> typing.List[typing.Any]:
        if isinstance(arg, rx.Tuple):
            arg = arg.fields
        elif isinstance(arg, rx.Expr):
            if isinstance(arg.struct_info_, sinfo.TupleStructInfo):
                arg = [rx.TupleGetItem(arg, i) for i, _ in enumerate(arg.struct_info_.fields)]
        if (not self.restrict) and (not isinstance(arg, (list, tuple, ir.Array))):
            arg = [arg]
        if self.length:
            if len(arg) == 1 and len(self.length) == 1:
                (a,), (l,) = arg[0], self.length
                return [self.elem(a) for _ in range(l)]
            if len(arg) not in self.length:
                raise TypeError(f"Expected length {self.length}, got {len(arg)}")
        return [self.elem(elem) for elem in arg]


def _convert_elem_to_arg(arg, elem):
    if isinstance(elem, Int) and isinstance(arg, int):
        arg = tir.IntImm("int64", arg)
    elif isinstance(elem, Float) and isinstance(arg, (int, float)):
        arg = tir.FloatImm("float32", arg)
    elif isinstance(arg, str):
        arg = rt.String(arg)
    if isinstance(arg, int):
        return tir.IntImm("int64", arg)
    if isinstance(arg, float):
        return tir.FloatImm("float32", arg)
    return arg


class Optional(_TypeChecker):  # pylint: disable=too-few-public-methods
    elem: _TypeChecker

    def __init__(self, elem: _TypeChecker):
        super().__init__()
        self.elem = elem

    def __call__(self, arg: ty.Optional) -> typing.Optional[typing.Any]:
        if arg is None:
            return None
        arg = self.elem(arg)
        return _convert_elem_to_arg(arg, self.elem)


class Map(_TypeChecker):  # pylint: disable=too-few-public-methods
    key: _TypeChecker
    value: _TypeChecker

    def __init__(self, key: _TypeChecker, value: _TypeChecker):
        super().__init__()
        self.key = key
        self.value = value

    def __call__(self, arg: ty.Map) -> typing.Dict[typing.Any, typing.Any]:
        if not isinstance(arg, (dict, ir.Map)):
            raise TypeError(f"Expected dict, got {type(arg)}")
        return {self.key(key): self.value(value) for key, value in arg.items()}


class Union(_TypeChecker):  # pylint: disable=too-few-public-methods
    elems: typing.Tuple[_TypeChecker, ...]

    def __init__(self, *elems: _TypeChecker):
        super().__init__()
        self.elems = elems

    def __call__(self, arg: typing.Any) -> typing.Any:
        for elem in self.elems:
            try:
                arg = elem(arg)
            except TypeError:
                continue
            return _convert_elem_to_arg(arg, elem)
        raise TypeError(f"Expected one of {self.elems}, got {type(arg)}")


##### PrimValues #####


def _convert_zero_dim_tensor(arg: rx.Constant) -> typing.Union[int, float, bool, rx.Constant]:
    a: np.ndarray = arg.data.numpy()
    if a.size != 1:
        return arg
    return tir.const(a.reshape(()).item(), str(arg.struct_info.dtype))


class Int(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Int) -> int:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, int):
            return arg
        if isinstance(arg, tir.IntImm):
            return arg.value
        raise TypeError(f"Expected int or IntImm, got {type(arg)}")


class Float(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Float) -> float:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, (int, float)):
            return float(arg)
        if isinstance(arg, (tir.IntImm, tir.FloatImm)):
            return float(arg.value)
        raise TypeError(f"Expected int, float, or FloatImm, got {type(arg)}")


class Bool(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Bool) -> bool:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, bool):
            return arg
        if isinstance(arg, tir.IntImm):
            return bool(arg.value)
        raise TypeError(f"Expected bool or IntImm, got {type(arg)}")


class PrimExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.PrimExpr) -> tir.PrimExpr:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, tir.PrimExpr):
            return arg
        if isinstance(arg, int):
            return tir.IntImm("int64", arg)
        if isinstance(arg, bool):
            return tir.IntImm("bool", arg)
        if isinstance(arg, float):
            return tir.FloatImm("float32", arg)
        raise TypeError(f"Expected int, bool, float, or PrimExpr, got {type(arg)}")


class IntPrimExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.IntPrimExpr) -> tir.PrimExpr:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, tir.PrimExpr):
            if str(arg.dtype).startswith("int"):
                return arg
            raise TypeError(f"Expected an integer PrimExpr, got {arg.dtype}")
        if isinstance(arg, int):
            return tir.IntImm("int64", arg)
        raise TypeError(f"Expected int or integer PrimExpr, got {type(arg)}")


class FloatPrimExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.FloatPrimExpr) -> tir.PrimExpr:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, tir.PrimExpr):
            if str(arg.dtype).startswith("float"):
                return arg
            raise TypeError(f"Expected a floating-point PrimExpr, got {arg.dtype}")
        if isinstance(arg, (int, float)):
            return tir.FloatImm("float32", float(arg))
        raise TypeError(f"Expected int, float, or floating-point PrimExpr, got {type(arg)}")


class BoolPrimExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.BoolPrimExpr) -> tir.PrimExpr:
        if isinstance(arg, rx.Constant):
            arg = _convert_zero_dim_tensor(arg)
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, tir.PrimExpr):
            if str(arg.dtype) == "bool":
                return arg
            raise TypeError(f"Expected a boolean PrimExpr, got {arg.dtype}")
        if isinstance(arg, bool):
            return tir.IntImm("bool", arg)
        raise TypeError(f"Expected bool or boolean PrimExpr, got {type(arg)}")


class TIRVar(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.TIRVar) -> tir.Var:
        if isinstance(arg, tir.Var):
            return arg
        raise TypeError(f"Expected Var, got {type(arg)}")


class IntTIRVar(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.IntTIRVar) -> tir.Var:
        if isinstance(arg, tir.Var):
            if str(arg.dtype).startswith("int"):
                return arg
            raise TypeError(f"Expected an integer Var, got {arg.dtype}")
        raise TypeError(f"Expected Var, got {type(arg)}")


class FloatTIRVar(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.FloatTIRVar) -> tir.Var:
        if isinstance(arg, tir.Var):
            if str(arg.dtype).startswith("float"):
                return arg
            raise TypeError(f"Expected a floating-point Var, got {arg.dtype}")
        raise TypeError(f"Expected Var, got {type(arg)}")


class BoolTIRVar(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.BoolTIRVar) -> tir.Var:
        if isinstance(arg, tir.Var):
            if str(arg.dtype) == "bool":
                return arg
            raise TypeError(f"Expected a boolean Var, got {arg.dtype}")
        raise TypeError(f"Expected Var, got {type(arg)}")


##### Tensor #####


class Tensor(_TypeChecker):  # pylint: disable=too-few-public-methods
    ndim: typing.List[int]

    def __init__(
        self,
        ndim: ty.Array[int],
    ):
        super().__init__()
        self.ndim = ndim

    def __call__(self, arg: ty.Tensor) -> rx.Expr:
        if isinstance(arg, (np.ndarray, rt.NDArray)):
            arg = rx.const(arg)
        if isinstance(arg, rx.Expr):
            return arg
        raise TypeError(f"Expected NDArray or relax.Expr whose sinfo is Tensor, got {type(arg)}")


class IntTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        ndim: ty.Array[int],
    ):
        super().__init__(ndim)

    def __call__(self, arg: ty.IntTensor) -> rx.Expr:
        return super().__call__(arg)


class FloatTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        ndim: ty.Array[int],
    ):
        super().__init__(ndim)

    def __call__(self, arg: ty.FloatTensor) -> rx.Expr:
        return super().__call__(arg)


class BoolTensor(Tensor):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        ndim: ty.Array[int],
    ):
        super().__init__(ndim)

    def __call__(self, arg: ty.BoolTensor) -> rx.Expr:
        return super().__call__(arg)


##### Misc #####


class AnyRelaxExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.AnyRelaxExpr) -> rx.Expr:
        if not isinstance(arg, rx.Expr):
            raise TypeError(f"Expected tvm.relax.Expr, got {type(arg)}")
        return arg


class TupleExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.TupleExpr) -> rx.Tuple:
        if isinstance(arg, rx.Expr) and not isinstance(arg, rx.Tuple):
            arg = [arg]
        if isinstance(arg, (tuple, list, ir.Array)):
            arg = rx.Tuple([AnyRelaxExpr()(x) for x in arg])
        if not isinstance(arg, rx.Tuple):
            raise TypeError(f"Expected relax.Tuple, got {type(arg)}")
        return arg


class Str(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Str) -> str:
        if isinstance(arg, tir.StringImm):
            arg = str(arg.value)
        if isinstance(arg, rx.StringImm):
            arg = str(arg.value)
        if isinstance(arg, (rt.String, str)):
            return arg
        raise TypeError(f"Expected str, got {type(arg)}")


class DType(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.DType) -> rt.DataType:
        if arg is None or arg == "":
            return rt.DataType("void")
        if isinstance(arg, str):
            try:
                return rt.DataType(arg)
            except ValueError:
                raise TypeError(f"Expected DataType, but cannot parse string: {arg}")
        if isinstance(arg, rx.DataTypeImm):
            return rt.DataType(arg.value)
        raise TypeError(f"Expected str or DataType, got {type(arg)}")


class Shape(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Shape) -> rx.Expr:
        if isinstance(arg, rx.Expr):
            return arg
        if not isinstance(arg, (tuple, list, ir.Array)):
            arg = [arg]
        try:
            return rx.ShapeExpr(list(IntPrimExpr()(x) for x in arg))
        except TypeError:
            raise TypeError(
                f"Expected Sequence[PrimExpr] or relax.ShapeExpr, got {type(arg)}"
            ) from None


class Axis(_TypeChecker):  # pylint: disable=too-few-public-methods
    of: rx.Expr
    is_insertion: bool
    normalize: bool

    def __init__(
        self,
        of: rx.Expr,
        is_insertion: bool,
        normalize: bool,
    ):
        super().__init__()
        self.of = of
        self.is_insertion = is_insertion
        self.normalize = normalize

    def __call__(self, arg: ty.Axis) -> int:
        return Int()(arg)


class Axes(_TypeChecker):  # pylint: disable=too-few-public-methods
    of: rx.Expr
    is_insertion: bool
    normalize: bool

    def __init__(
        self,
        of: rx.Expr,
        is_insertion: bool,
        normalize: bool,
    ):
        super().__init__()
        self.of = of
        self.is_insertion = is_insertion
        self.normalize = normalize

    def __call__(self, arg: ty.Axes) -> typing.Optional[typing.List[int]]:
        if arg is None:
            return None
        return Array(Int(), length=[])(arg)


class GlobalVar(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.GlobalVar) -> ir.GlobalVar:
        if isinstance(arg, ir.GlobalVar):
            return arg
        raise TypeError(f"Expected GlobalVar, got {type(arg)}")


class ExternFunc(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.ExternFunc) -> rx.ExternFunc:
        if isinstance(arg, rx.ExternFunc):
            return arg
        if isinstance(arg, str):
            return rx.ExternFunc(global_symbol=arg)
        raise TypeError(f"Expected ExternFunc, got {type(arg)}")


class IndexMap(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.IndexMap) -> tir.IndexMap:
        if callable(arg):
            arg = tir.IndexMap.from_func(arg)
        if isinstance(arg, tir.IndexMap):
            return arg
        raise TypeError(f"Expected IndexMap, got {type(arg)}")


class StructInfo(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.StructInfo) -> sinfo.StructInfo:
        if hasattr(arg, "as_struct_info"):
            arg = arg.as_struct_info()
        if isinstance(arg, sinfo.StructInfo):
            return arg
        if isinstance(arg, (tuple, list, ir.Array)):
            return sinfo.TupleStructInfo([StructInfo()(x) for x in arg])
        raise TypeError(f"Expected StructInfo, got {type(arg)}")
