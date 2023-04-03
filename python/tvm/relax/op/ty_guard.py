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

    def __init__(
        self,
        elem: _TypeChecker,
        length: typing.List[int],
    ):
        super().__init__()
        self.elem = elem
        self.length = length

    def __call__(self, arg: ty.Array) -> typing.List[typing.Any]:
        if not isinstance(arg, (list, tuple, ir.Array)):
            arg = [arg]
        if self.length:
            if len(arg) == 1 and len(self.length) == 1:
                (a,), (l,) = arg[0], self.length
                return [self.elem(a) for _ in range(l)]
            if len(arg) not in self.length:
                raise TypeError(f"Expected length {self.length}, got {len(arg)}")
        return [self.elem(elem) for elem in arg]


class Optional(_TypeChecker):  # pylint: disable=too-few-public-methods
    elem: _TypeChecker

    def __init__(self, elem: _TypeChecker):
        super().__init__()
        self.elem = elem

    def __call__(self, arg: ty.Optional) -> typing.Optional[typing.Any]:
        if arg is None:
            return None
        return self.elem(arg)


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
                return elem(arg)
            except TypeError:
                continue
        raise TypeError(f"Expected one of {self.elems}, got {type(arg)}")


##### PrimValues #####


class Int(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Int) -> int:
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, int):
            return arg
        if isinstance(arg, tir.IntImm):
            return arg.value
        raise TypeError(f"Expected int or IntImm, got {type(arg)}")


class Float(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Float) -> float:
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, (int, float)):
            return float(arg)
        if isinstance(arg, (tir.IntImm, tir.FloatImm)):
            return float(arg.value)
        raise TypeError(f"Expected int, float, or FloatImm, got {type(arg)}")


class Bool(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.Bool) -> bool:
        if isinstance(arg, rx.PrimValue):
            arg = arg.value
        if isinstance(arg, bool):
            return arg
        if isinstance(arg, tir.IntImm):
            return bool(arg.value)
        raise TypeError(f"Expected bool or IntImm, got {type(arg)}")


class PrimExpr(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.PrimExpr) -> tir.PrimExpr:
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
            if isinstance(arg.struct_info, sinfo.TensorStructInfo):
                if self.ndim and int(arg.struct_info.ndim) not in self.ndim:
                    raise TypeError(
                        f"Expected Tensor with ndim in {self.ndim}, but got {arg.struct_info.ndim}"
                    )
                return arg
            raise TypeError(f"Expected TensorStructInfo, but got {arg.struct_info}")
        raise TypeError(f"Expected NDArray, Tensor, or relax.Expr, got {type(arg)}")


class IntTensor(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.IntTensor) -> rx.Expr:
        arg = super().__call__(arg)
        assert isinstance(arg, rx.Expr) and isinstance(arg.struct_info, sinfo.TensorStructInfo)
        if str(arg.struct_info.dtype).startswith("int"):
            return arg
        raise TypeError(f"Expected integer Tensor, but got {arg.struct_info.dtype}")


class FloatTensor(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.FloatTensor) -> rx.Expr:
        arg = super().__call__(arg)
        assert isinstance(arg, rx.Expr) and isinstance(arg.struct_info, sinfo.TensorStructInfo)
        if str(arg.struct_info.dtype).startswith("float"):
            return arg
        raise TypeError(f"Expected floating-point Tensor, but got {arg.struct_info.dtype}")


class BoolTensor(_TypeChecker):  # pylint: disable=too-few-public-methods
    def __call__(self, arg: ty.BoolTensor) -> rx.Expr:
        arg = super().__call__(arg)
        assert isinstance(arg, rx.Expr) and isinstance(arg.struct_info, sinfo.TensorStructInfo)
        if str(arg.struct_info.dtype) == "bool":
            return arg
        raise TypeError(f"Expected boolean Tensor, but got {arg.struct_info.dtype}")


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
        if arg is None:
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
            if isinstance(arg.struct_info, sinfo.ShapeStructInfo):
                return arg
            raise TypeError(
                f"Expected relax expression with ShapeStructInfo, but got {arg.struct_info}"
            )
        if isinstance(arg, (tuple, list, ir.Array)):
            return rx.ShapeExpr(list(IntPrimExpr()(x) for x in arg))
        raise TypeError(f"Expected Sequence[PrimExpr] or relax.ShapeExpr, got {type(arg)}")


def _get_equal_ndim(
    tensors: typing.Union[
        rx.Expr,
        typing.List[rx.Expr],
    ]
) -> int:
    if isinstance(tensors, rx.Expr):
        tensors = [tensors]
    if len(tensors) == 0:
        raise TypeError("Expected at least one tensor")
    ndim = -1
    for tensor in tensors:
        if not isinstance(tensor, rx.Expr):
            raise TypeError(f"Expected relax.Expr, got {type(tensor)}")
        if not isinstance(tensor.struct_info, sinfo.TensorStructInfo):
            raise TypeError(f"Expected TensorStructInfo, but got {tensor.struct_info}")
        t_ndim = tensor.struct_info.ndim
        if t_ndim == -1:
            continue
        if ndim == -1:
            ndim = t_ndim
        elif ndim != t_ndim:
            raise TypeError(f"Expected tensors with equal ndim, but got {ndim} and {t_ndim}")
    return ndim


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
        ndim = _get_equal_ndim(self.of)
        axis = Int()(arg)
        if ndim != -1 and self.normalize:
            if self.is_insertion:
                axis = axis % (ndim + 1)
            else:
                axis = axis % ndim
        return axis


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

    def __call__(self, arg: ty.Axes) -> typing.List[int]:
        ndim = self.of.struct_info_.ndim
        axes = Array(Int(), length=[])(arg)
        if self.normalize:
            if self.is_insertion:
                axes = [x % (ndim + 1) for x in axes]
            else:
                axes = [x % ndim for x in axes]
            axes = sorted(x % (ndim + 1) for x in axes)
        return axes


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
