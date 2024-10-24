# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The expression nodes of Relax."""
import typing
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Union, Mapping

import numpy as _np  # type: ignore

import tvm
import tvm._ffi
import tvm.ir
import tvm.relax
from tvm import DataType
from tvm._ffi import base as _base
from tvm.runtime import Object
from tvm.runtime import ndarray as _nd

from ..ir import BaseFunc, Node, Span
from ..runtime import Scriptable, String
from ..tir import PrimExpr
from . import _ffi_api

# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
Expr = Union[tvm.ir.RelayExpr]
Type = Union[tvm.ir.Type]  # pylint: disable=invalid-name
GlobalVar = Union[tvm.ir.GlobalVar]


@tvm._ffi.register_object("relax.Id")
class Id(Object):
    """Unique identifier(name) used in Var.
    Guaranteed to be stable across all passes.
    """

    name_hint: str

    def __init__(self):
        raise RuntimeError("Cannot directly construct Id")


# NOTE: place base struct info in expr to avoid cyclic dep
# from expr to struct info.
class StructInfo(Node, Scriptable):
    """The base class of all StructInfo.

    StructInfo contains both the static type
    and runtime structural information.
    """

    def __eq__(self, other):
        """Compare two struct info for structural equivalence."""
        return tvm.ir.structural_equal(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def same_as(self, other):
        """Overload with structural equality."""
        return super().__eq__(other)

    def is_base_of(self, derived: "StructInfo") -> bool:
        """Check if self is base of another derived struct info.

        Parameters
        ----------
        derived : StructInfo
            The derived struct info to be checked.

        Returns
        -------
        result : bool
            The check result.
        """
        return _ffi_api.StructInfoIsBaseOf(self, derived)  # type: ignore


# will be registered afterwards in python/tvm/relax/op/init.py
_op_ffi_api = None  # pylint: disable=invalid-name


def _binary_op_helper(lhs: "ExprWithOp", rhs: "ExprWithOp", op: Callable) -> "ExprWithOp":
    if not isinstance(lhs, Expr):  # type: ignore
        raise ValueError("lhs must be Expr")
    if isinstance(rhs, Expr):  # type: ignore
        return op(lhs, rhs)
    elif isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    else:
        raise TypeError(f"type {type(rhs)} not supported")


def _binary_rhs_helper(rhs: "ExprWithOp") -> "ExprWithOp":
    if isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    raise TypeError(f"type {type(rhs)} not supported")


class ExprWithOp(Expr, Scriptable):
    """Basetype of all relax expressions that defines op overloading."""

    def astype(self, dtype: Union[str, DataType]) -> "ExprWithOp":
        """Cast the content type of the current data to dtype.

        Parameters
        ----------
        dtype : str
            The target data type.

        Note
        ----
        This function only works for TensorType Exprs.

        Returns
        -------
        result : ExprWithOp
            The result expression.
        """
        return _op_ffi_api.astype(self, dtype)  # type: ignore

    def __neg__(self) -> "ExprWithOp":
        return _op_ffi_api.negative(self)  # type: ignore

    def __lt__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.less)  # type: ignore

    def __gt__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.greater)  # type: ignore

    def __ge__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.greater_equal)  # type: ignore

    def __le__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.less_equal)  # type: ignore

    # NOTE: Cannot override __eq__ and __ne__, which will influence object equal

    def __add__(self, other: Expr) -> "ExprWithOp":
        if isinstance(self.struct_info_, tvm.relax.TupleStructInfo) and isinstance(other, tuple):
            return tuple([*self, *other])

        return _binary_op_helper(self, other, _op_ffi_api.add)  # type: ignore

    def __radd__(self, other: Expr) -> "ExprWithOp":
        return self.__add__(other)

    def __sub__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.subtract)  # type: ignore

    def __rsub__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __mul__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.multiply)  # type: ignore

    def __rmul__(self, other: Expr) -> "ExprWithOp":
        return self.__mul__(other)

    def __truediv__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.divide)  # type: ignore

    def __rtruediv__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __floordiv__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.floor_divide)  # type: ignore

    def __rfloordiv__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __mod__(self, other: Expr) -> "ExprWithOp":
        # TODO(siyuan): Support it after mod operator is supported in relax
        raise ValueError("relax.mod is not supported yet.")

    def __rmod__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __pow__(self, other: Expr) -> "ExprWithOp":
        return _binary_op_helper(self, other, _op_ffi_api.power)  # type: ignore

    def __rpow__(self, other: Expr) -> "ExprWithOp":
        return _binary_rhs_helper(other)

    def __call__(self, *args: List[Expr], attrs: Optional[Dict[str, Any]] = None) -> "ExprWithOp":
        """Call the variable (if it represents a function).

        Parameters
        ----------
        args: List[Expr]
            The arguments to the call.

        attr: Optional[Dict[str, object]]
            The additional attributes to the call.

        Returns
        -------
        call: ExprWithOp
            A call taking the variable as a function.
        """
        return Call(self, args, attrs=attrs)

    def __getitem__(self, index: int) -> "ExprWithOp":
        """Get the i-th element of the tuple or Expr with TupleType.

        Parameters
        ----------
        index: int
            The index of the element to be retrieved.

        Note
        ----
        This function will be overridden by Tuple and ShapeExpr

        Returns
        -------
        result: ExprWithOp
            The result expression.
        """
        try:
            return TupleGetItem(self, index)
        except tvm.TVMError as err:
            # For Python objects with __getitem__, but without
            # __len__, tuple unpacking is done by iterating over
            # sequential indices until IndexError is raised.
            # Therefore, convert from TVMError to IndexError for
            # compatibility.
            if "Index out of bounds" in err.args[0]:
                raise IndexError from err
            raise

    def _check_for_tensor_struct_info(self):
        """Raise an error if this is something other than a Tensor

        Used for early checks in `expr.dtype` and `expr.shape`
        accessors.  While invalid usage would cause errors to be
        raised during shape inference, an earlier check makes it
        easier to find the invalid usage.
        """
        if self.struct_info_ is None:
            return

        if not isinstance(self.struct_info_, tvm.relax.TensorStructInfo):
            raise TypeError(
                f"Runtime unpacking of DLDataType is only implemented for tensors, "
                f"but was applied to object {self} of type {type(self)}."
            )

    @property
    def dtype(self) -> "_DLTensorDTypeProxy":
        """Returns a proxy object for accessing DLTensor::dtype"""
        self._check_for_tensor_struct_info()
        return _DLTensorDTypeProxy(self)

    @property
    def ndim(self) -> "Expr":
        """Returns the runtime value of DLTensor::ndim"""
        self._check_for_tensor_struct_info()
        op = tvm.ir.Op.get("relax.inspect.tensor_ndim")
        return tvm.relax.Call(op, [self])

    @property
    def shape(self) -> "_DLTensorShapeProxy":
        """Returns a proxy object for accessing DLTensor::shape"""
        self._check_for_tensor_struct_info()
        return _DLTensorShapeProxy(self)

    @property
    def strides(self) -> "_DLTensorStrideProxy":
        """Returns a proxy object for accessing DLTensor::strides"""
        self._check_for_tensor_struct_info()
        return _DLTensorStrideProxy(self)

    @property
    def byte_offset(self) -> "Expr":
        """Returns a proxy object for accessing DLTensor::byte_offset"""
        self._check_for_tensor_struct_info()
        op = tvm.ir.Op.get("relax.inspect.tensor_byte_offset")
        return tvm.relax.Call(op, [self])

    @property
    def elem_offset(self) -> "Expr":
        """Returns a proxy object for accessing a DLTensor's elem_offset

        This parameter is not stored in the DLTensor, but is instead
        derived from the DLTensor's byte offset and datatype.  This is
        exposed in Relax for ease of use, and for translation into the
        `tir::BufferNode::elem_offset` field when interacting with TIR
        buffers.
        """
        self._check_for_tensor_struct_info()
        op = tvm.ir.Op.get("relax.inspect.tensor_elem_offset")
        return tvm.relax.Call(op, [self])


class _DLTensorDTypeProxy(tvm.runtime.ObjectGeneric):
    """A proxy object for unpacking DLDatatype from DLTensor

    Exposes accessors for `DLDataType` fields `type_code`, `lanes`,
    and `bits` within a `DLTensor::dtype`.  Accessing these fields
    will produce `relax.Call` expressions, representing the field's
    runtime value.  If the datatype of the tensor is known at
    compile-time, the `relax.Call` will be normalized into a
    `relax.PrimValue`, with no runtime cost.

    Parameters
    ----------
    tensor: relax.Expr

        The relax tensor (or a variable referring to a relax tensor),
        whose runtime shape is being inspected.

    """

    def __init__(self, tensor):
        self.tensor = tensor

    def asobject(self):
        """Provide expected in error message

        This method is called when `_DLTensorDTypeProxy` is used in a
        context that requires a `relax.Expr`.  This usage is not
        supported, and raising an error here can provide suggested
        fixes that are not present in the default error message from
        `tvm.runtime.convert_to_object`.
        """

        fields = [f"{self.tensor}.dtype.{field}" for field in ["type_code", "bits", "lanes"]]
        raise TypeError(
            f"{self.tensor}.dtype cannot be converted to a relax expression, "
            f"and should be used as a proxy object to access "
            f"fields {fields}"
        )

    @property
    def type_code(self) -> Expr:
        """Accessor for the DLDataType::bits field

        Returns
        -------
        type_code: Expr

            The type code of the DLTensor.  See the `DLDeviceType`
            enum in `dlpack.h` for more information.
        """
        op = tvm.ir.Op.get("relax.inspect.tensor_dtype_code")
        return tvm.relax.Call(op, [self.tensor])

    @property
    def lanes(self) -> Expr:
        """Accessor for the DLDataType::bits field

        Returns
        -------
        lanes: Expr

            The number of lanes in the DLDataType
        """
        op = tvm.ir.Op.get("relax.inspect.tensor_dtype_lanes")
        return tvm.relax.Call(op, [self.tensor])

    @property
    def bits(self) -> Expr:
        """Accessor for the DLDataType::bits field

        Returns
        -------
        bits: Expr

            The number of bits in the DLDataType
        """
        op = tvm.ir.Op.get("relax.inspect.tensor_dtype_bits")
        return tvm.relax.Call(op, [self.tensor])


class _DLTensorShapeProxy(tvm.runtime.ObjectGeneric):
    """A proxy object for unpacking the shape from DLTensor

    Exposes accessors for the `DLTensor::shape` field.  Accessing
    these fields will produce `relax.Call` expressions, representing
    the field's runtime value.  If the datatype of the tensor is known
    at compile-time, the `relax.Call` will be normalized into a
    `relax.PrimValue`, with no runtime cost.

    Parameters
    ----------
    tensor: relax.Expr

        The relax tensor (or a variable referring to a relax tensor),
        whose runtime shape is being inspected.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def asobject(self):
        """Provide expected in error message

        This method is called when `_DLTensorShapeProxy` is used in a
        context that requires a `relax.Expr`.  This usage is not
        supported, and raising an error here can provide suggested
        fixes that are not present in the default error message from
        `tvm.runtime.convert_to_object`.
        """
        raise TypeError(
            f"{self.tensor}.shape cannot be converted to a relax expression, "
            f"and should be used as a proxy object to access the runtime shape of the DLTensor. "
            f"The DLTensor::ndim field can be accessed as len({self.tensor}), "
            f"and the DLTensor::shape array can be accessed as {self.tensor}.shape[i]"
        )

    def __getitem__(self, axis: Union[int, PrimExpr, Expr]) -> Expr:
        """Returns the extent of a tensor axis

        Parameters
        ----------
        axis: Union[int, PrimExpr, Expr]

            The tensor axis whose extent should be returned.  For ease
            of use, any python integers or TIR expressions are
            converted to `relax.Expr`.

        Returns
        -------
        extent: Expr

            The extent of the tensor's axis.
        """

        if not isinstance(axis, tvm.relax.Expr):
            axis = tvm.relax.PrimValue(axis)

        if axis.struct_info_ is not None and not isinstance(
            axis.struct_info_, tvm.relax.PrimStructInfo
        ):
            raise TypeError(
                f"The index used to access {self.tensor}.shape "
                f'must have struct info R.Prim("int64"), '
                f"but index {axis} had struct info {axis.struct_info_}."
            )

        op = tvm.ir.Op.get("relax.inspect.tensor_shape_i")
        return tvm.relax.Call(op, [self.tensor, axis])


class _DLTensorStrideProxy(tvm.runtime.ObjectGeneric):
    """A proxy object for unpacking the strides from DLTensor

    Exposes accessors for the `DLTensor::strides` field.  Accessing
    these fields will produce `relax.Call` expressions, representing
    the field's runtime value.  If the datatype of the tensor is known
    at compile-time, the `relax.Call` will be normalized into a
    `relax.PrimValue`, with no runtime cost.

    Parameters
    ----------
    tensor: relax.Expr

        The relax tensor (or a variable referring to a relax tensor),
        whose runtime strides is being inspected.
    """

    def __init__(self, tensor):
        self.tensor = tensor

    def asobject(self):
        """Provide expected in error message

        This method is called when `_DLTensorStrideProxy` is used in a
        context that requires a `relax.Expr`.  This usage is not
        supported, and raising an error here can provide suggested
        fixes that are not present in the default error message from
        `tvm.runtime.convert_to_object`.
        """
        raise TypeError(
            f"{self.tensor}.strides cannot be converted to a relax expression, "
            f"and should be used as a proxy object to access the runtime strides of the DLTensor. "
            f"The DLTensor::ndim field can be accessed as len({self.tensor}), "
            f"and the DLTensor::strides array can be accessed as {self.tensor}.strides[i]"
        )

    def __getitem__(self, axis: Union[int, PrimExpr, Expr]) -> Expr:
        """Returns the extent of a tensor axis

        Parameters
        ----------
        axis: Union[int, PrimExpr, Expr]

            The tensor axis whose extent should be returned.  For ease
            of use, any python integers or TIR expressions are
            converted to `relax.Expr`.

        Returns
        -------
        extent: Expr

            The extent of the tensor's axis.
        """

        if not isinstance(axis, tvm.relax.Expr):
            axis = tvm.relax.PrimValue(axis)

        if axis.struct_info_ is not None and not isinstance(
            axis.struct_info_, tvm.relax.PrimStructInfo
        ):
            raise TypeError(
                f"The index used to access {self.tensor}.strides "
                f'must have struct info R.Prim("int64"), '
                f"but index {axis} had struct info {axis.struct_info_}."
            )

        op = tvm.ir.Op.get("relax.inspect.tensor_stride_i")
        return tvm.relax.Call(op, [self.tensor, axis])


@tvm._ffi.register_object("relax.expr.Call")
class Call(ExprWithOp):
    """Function call node in Relax.

    Call node corresponds the operator application node
    in computational graph terminology.

    Parameters
    ----------
    op: tvm.ir.Op or any tvm.relax.Expr with function type.
        The operation to be called.

    args: Union[List[Expr], typing.Tuple[Expr, ...]]
        The arguments to the call.

    attrs: Optional[tvm.ir.Attrs]
        Attributes to the call, can be None

    sinfo_args: Optional[Union[List[StructInfo], typing.Tuple[StructInfo, ...]]]
        The structure info arguments of a CallNode.
        sinfo_args is designed to be non-empty only for intrinsic op (e.g.,
        call_tir, call_builtin_with_ctx, etc.) and calls to ExternFuncs, with the main
        usage of structure info inference.

    span: Optional[Span]
        Span that points to original source code
    """

    op: Expr
    args: List[Expr]
    attrs: tvm.ir.Attrs
    sinfo_args: List[StructInfo]
    span: Optional[Span]

    def __init__(
        self,
        op: Union[Expr, tvm.ir.Op],
        args: Union[List[Expr], typing.Tuple[Expr, ...]],
        attrs: Optional[tvm.ir.Attrs] = None,
        sinfo_args: Optional[Union[List[StructInfo], typing.Tuple[StructInfo, ...]]] = None,
        span: Optional[Span] = None,
    ):
        if not sinfo_args:
            sinfo_args = []
        self.__init_handle_by_constructor__(
            _ffi_api.Call, op, args, attrs, sinfo_args, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.If")
class If(ExprWithOp):
    """A conditional expression in Relax.

    Parameters
    ----------
    cond: Expr
        The condition.

    true_branch: Expr
        The expression evaluated when condition is true.

    false_branch: Expr
        The expression evaluated when condition is false.

    span: Optional[Span]
        Span that points to original source code
    """

    cond: Expr
    true_branch: Expr
    false_branch: Expr
    span: Optional[Span]

    def __init__(
        self, cond: Expr, true_branch: Expr, false_branch: Expr, span: Optional[Span] = None
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.If, cond, true_branch, false_branch, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.Tuple")
class Tuple(ExprWithOp):
    """Tuple expression that groups several fields together.

    Parameters
    ----------
    fields : Union[List[Expr], typing.Tuple[Expr, ...]]
        The fields in the tuple.

    span: Optional[Span]
        Span that points to original source code
    """

    fields: List[Expr]
    span: Optional[Span]

    def __init__(
        self, fields: Union[List[Expr], typing.Tuple[Expr, ...]], span: Optional[Span] = None
    ):
        if isinstance(fields, tvm.relax.Tuple):
            fields = fields.fields
        elif isinstance(getattr(fields, "struct_info_", None), tvm.relax.TupleStructInfo):
            fields = [*fields]

        self.__init_handle_by_constructor__(_ffi_api.Tuple, fields, span)  # type: ignore

    def __getitem__(self, index: int) -> Expr:
        if index >= len(self) or index < -len(self):
            raise IndexError("Tuple index out of range")
        return self.fields[index]

    def __len__(self) -> int:
        return len(self.fields)


@tvm._ffi.register_object("relax.expr.TupleGetItem")
class TupleGetItem(ExprWithOp):
    """Get index-th item from a tuple.

    Parameters
    ----------
    tuple_value: Expr
        The input tuple expression.

    index: int
        The index.

    span: Optional[Span]
        Span that points to original source code
    """

    tuple_value: Expr
    index: int
    span: Optional[Span]

    def __init__(self, tuple_value: Expr, index: int, span: Optional[Span] = None):
        self.__init_handle_by_constructor__(
            _ffi_api.TupleGetItem, tuple_value, index, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(ExprWithOp):
    """A shape expression which allows users to construct a shape containing PrimExpr.

    Parameters
    ----------
    values: Union[List[PrimExpr], typing.Tuple[PrimExpr, ...], tvm.ir.Array]
        The values of the shape expression.

    span: Optional[Span]
        Span that points to original source code
    """

    values: List[PrimExpr]
    span: Optional[Span]

    def __init__(
        self,
        values: Union[List[PrimExpr], typing.Tuple[PrimExpr, ...], tvm.ir.Array],
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values, span)  # type: ignore

    def __getitem__(self, index):
        if index >= len(self) or index < -len(self):
            raise IndexError("ShapeExpr index out of range")
        return self.values[index]

    def __len__(self):
        return len(self.values)


def make_shape(shape: Union[List[Any], typing.Tuple[Any, ...]]) -> ShapeExpr:
    if isinstance(shape, (list, tuple)):
        return ShapeExpr(shape)
    raise ValueError("Wrong type")


@tvm._ffi.register_object("relax.expr.Constant")
class Constant(ExprWithOp):
    """Constant Tensor

    Parameters
    ----------
    data: tvm.nd.NDArray
        The data of the constant tensor.

    struct_info: Optional[StructInfo]
        The struct info of the constant tensor. If not specified, infer it from data.

    span: Optional[Span]
        Span that points to original source code

    Note
    ----
    Scalar constants are represented by ndim-0 constant tensors.
    """

    data: tvm.nd.NDArray
    span: Optional[Span]

    def __init__(
        self,
        data: tvm.nd.NDArray,
        struct_info: Optional[StructInfo] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Constant, data, struct_info, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.Var")
class Var(ExprWithOp):
    """The variable class for all Relax bindings.

    Parameters
    ----------
    name_hint: Union[str, Id]
        The name hint of the variable.

    struct_info: Optional[StructInfo]
        The struct info annotation of the variable.

    span: Optional[Span]
        Span that points to original source code
    """

    vid: Id
    span: Optional[Span]

    def __init__(
        self,
        name_hint: Union[str, Id],
        struct_info: Optional[StructInfo] = None,
        span: Optional[Span] = None,
    ) -> None:
        if struct_info is not None:
            struct_info = tvm.runtime.convert_to_object(struct_info)
            if not isinstance(struct_info, StructInfo):
                raise TypeError(
                    "struct_info needs to be an instance of StructInfo. "
                    "If you attempt to pass in shape, "
                    "use relax.TensorStructInfo(shape, dtype)."
                )
        self.__init_handle_by_constructor__(
            _ffi_api.Var if isinstance(name_hint, str) else _ffi_api.VarFromId,  # type: ignore
            name_hint,
            struct_info,
            span,
        )

    @property
    def name_hint(self) -> str:
        """Get name hint of the current var."""
        name = str(self.vid.name_hint)
        return name


@tvm._ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    """A sub-type of the variable node used to mark dataflow variables from
    normal visible "function local" bindings.


    Parameters
    ----------
    name_hint: Union[str, Id]
        The name hint of the variable.

    struct_info: Optional[StructInfo]
        The struct info annotation of the variable.

    span: Optional[Span]
        Span that points to original source code
    """

    vid: Id
    span: Optional[Span]

    def __init__(
        self,
        name_hint: Union[str, Id],
        struct_info: Optional[StructInfo] = None,
        span: Optional[Span] = None,
    ) -> None:
        # pylint: disable=super-init-not-called
        if struct_info is not None:
            struct_info = tvm.runtime.convert_to_object(struct_info)
            if not isinstance(struct_info, StructInfo):
                raise TypeError(
                    "struct_info needs to be an instance of StructInfo. "
                    "If you attempt to pass in shape, "
                    "use relax.TensorStructInfo(shape, dtype)."
                )

        self.__init_handle_by_constructor__(
            _ffi_api.DataflowVar  # type: ignore
            if isinstance(name_hint, str)
            else _ffi_api.DataflowVarFromId,  # type: ignore
            name_hint,
            struct_info,
            span,
        )


@tvm._ffi.register_object("relax.expr.PrimValue")
class PrimValue(Expr, Scriptable):
    """The prim expr representing the value."""

    value: PrimExpr

    def __init__(self, value: Union[PrimExpr, int], span: Optional[Span] = None) -> None:
        if isinstance(value, int):
            value = tvm.tir.IntImm("int64", value)
        self.__init_handle_by_constructor__(_ffi_api.PrimValue, value, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.StringImm")
class StringImm(Expr, Scriptable):
    """Represent a string literal constant."""

    value: str
    span: Optional[Span]

    def __init__(self, value: str, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.StringImm, value, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.DataTypeImm")
class DataTypeImm(Expr, Scriptable):
    """Represent a data type constant."""

    value: DataType
    span: Optional[Span]

    def __init__(self, value: Union[DataType, str], span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.DataTypeImm, value, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.Binding")
class Binding(Node, Scriptable):
    """The base class of a binding in Relax."""

    var: Var
    span: Optional[Span]


@tvm._ffi.register_object("relax.expr.MatchCast")
class MatchCast(Binding):
    """Runtime-match the value to the struct info.

    This operation does runtime check, populates the un-defined symbolic shape vars
    and vars in struct_info in the first occurrence, and insert equality assertions in
    other cases.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    struct_info: tvm.relax.StructInfo
        The struct info to match cast to.
    """

    struct_info: StructInfo
    value: Expr
    span: Optional[Span]

    def __init__(
        self, var: Var, value: Expr, struct_info: StructInfo, span: Optional[Span] = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchCast, var, value, struct_info, span  # type: ignore
        )


@tvm._ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    """Variable binding, bind he variable of the lhs with the rhs.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    """

    var: Var
    value: Expr
    span: Optional[Span]

    def __init__(self, var: Var, value: Expr, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node, Scriptable):
    """base class of binding block, bindings inside can be impure
    (with side effect or control flow)"""

    bindings: List[Binding]
    span: Optional[Span]

    def __init__(self, bindings: List[Binding], span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    """dataflow block, bindings inside are pure (no side effect and no control flow)"""

    bindings: List[Binding]
    span: Optional[Span]

    def __init__(self, bindings: List[Binding], span: Optional[Span] = None) -> None:
        # pylint: disable=super-init-not-called
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(ExprWithOp):
    """A sequence of binding blocks followed by an expression."""

    blocks: List[BindingBlock]
    body: Expr
    span: Optional[Span]

    def __init__(self, blocks: List[BindingBlock], body: Expr, span: Optional[Span] = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, span)  # type: ignore


@tvm._ffi.register_object("relax.expr.Function")
class Function(BaseFunc, Scriptable):
    """A Relax function."""

    params: List[Var]
    body: Expr
    ret_struct_info: StructInfo
    is_pure: bool
    attrs: tvm.ir.DictAttrs
    span: Optional[Span]

    def __init__(
        self,
        params: List[Var],
        body: Expr,
        ret_struct_info: Optional[StructInfo] = None,
        is_pure: Optional[bool] = True,
        attrs: Optional[tvm.ir.DictAttrs] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            params,
            body,
            ret_struct_info,
            is_pure,
            attrs,
            span,
        )  # type: ignore

    @staticmethod
    def create_empty(
        params: List[Var],
        ret_struct_info: StructInfo,
        is_pure: Optional[bool] = True,
        attrs: Optional[tvm.ir.DictAttrs] = None,
        span: Optional[Span] = None,
    ):
        """Construct a relax.Function but without body"""
        return _ffi_api.FunctionCreateEmpty(
            params, ret_struct_info, is_pure, attrs, span
        )  # type: ignore

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relax.Expr]
            Arguments.
        """
        return Call(self, args, None, None)

    def bind_symbolic_vars(
        self, binding_map: Mapping[Union[str, tvm.tir.Var], PrimExpr]
    ) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[Union[str, tvm.tir.Var], PrimExpr]

            The mapping of values to be replaced.  Keys may be either
            a `tir.Var` or a string name of the variable.  If the
            variables are referred to by name, the name must uniquely
            identify a symbolic variable in the function.

        Returns
        -------
        func: Function

            The updated function
        """

        # Relax uses int64 for symbolic variables, but the FFI
        # converts python integers into int32.
        binding_map = {
            key: tvm.tir.const(value, "int64") if isinstance(value, int) else value
            for key, value in binding_map.items()
        }

        return _ffi_api.FunctionBindSymbolicVars(self, binding_map)  # type: ignore

    def bind_params(
        self,
        binding_map: Mapping[
            Union[str, Var],
            Union[int, float, PrimExpr, tvm.runtime.NDArray, _np.ndarray, Expr],
        ],
    ) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[
                Union[str, Var],
                Union[int, float, PrimExpr, tvm.runtime.NDArray, _np.ndarray, Expr],
        ]

            The mapping of values to be replaced.

            Keys may be either a `relax.Var` or a string name of the
            Relax variable.  If the variables are referred to by name,
            the name must uniquely identify a parameter in the
            function.

            Values must be a relax expression, or a value that is
            convertible into a relax expression.  The value must be
            compatible with the variable being replaced.

        Returns
        -------
        func: Function

            The updated function
        """

        def _normalize_value(value):
            # Conversions that must occur prior to the FFI
            # conversions.
            if isinstance(value, int):
                # Relax uses int64 for symbolic variables, but the FFI
                # converts python integers into int32.
                return tvm.tir.const(value, "int64")
            elif isinstance(value, (_np.ndarray, tvm.nd.NDArray)):
                return tvm.relax.const(value)
            else:
                return value

        binding_map = {key: _normalize_value(value) for key, value in binding_map.items()}

        return _ffi_api.FunctionBindParams(self, binding_map)  # type: ignore

    def inline_functions(
        self, function_map: Mapping[Union[str, tvm.ir.GlobalVar], "Function"]
    ) -> "Function":
        return _ffi_api.FunctionInlineFunctions(self, function_map)  # type: ignore


@tvm._ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc, ExprWithOp):
    """extern function, which represents a PackedFunc."""

    global_symbol: String
    span: Optional[Span]

    def __init__(
        self,
        global_symbol: String,
        struct_info: Optional[StructInfo] = None,
        span: Optional[Span] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExternFunc, global_symbol, struct_info, span  # type: ignore
        )


def extern(name: str, struct_info: Optional[StructInfo] = None, span: Optional[Span] = None):
    """Create extern function."""
    return ExternFunc(name, struct_info, span)


def const(
    value: Union[bool, int, float, _np.ndarray, tvm.nd.NDArray], dtype: Optional[str] = None
) -> Constant:
    """Create a constant value.

    Parameters
    ----------
    value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
        The constant value.

    dtype: Optional[str]
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    if isinstance(value, (_base.numeric_types, (bool, list))):
        value = _np.array(value, dtype=dtype)

    if not dtype:
        # when dtype is None: int maps to "int32", float maps to "float32"
        dtype = {  # type: ignore
            _np.dtype("int64"): _np.int32,  # type: ignore
            _np.dtype("float64"): _np.float32,  # type: ignore
        }.get(
            value.dtype, None  # type: ignore
        )

    if isinstance(value, (_np.ndarray, _np.generic)):
        if dtype is not None:
            value = value.astype(dtype)
        value = _nd.array(value)

    if not isinstance(value, _nd.NDArray):
        raise ValueError("value has to be scalar or NDArray")

    return Constant(value)


def te_tensor(
    value: Expr, tir_var_map: Dict[tvm.tir.Var, tvm.tir.PrimExpr], name: str = "rxplaceholder"
):
    """Create a TE tensor from relax expression, with TIR variables in the
    tensor shape substituted by the given mapping

    Parameters
    ----------
    value : Expr
        The relax expression, which is required to have TensorStructInfo.

    tir_var_map : Dict[tvm.tir.Var, tvm.tir.PrimExpr]
        The mapping to substitute the TIR variables appeared in the
        shape of the input Expr.

    name : str
        The name of the created tensor.
    """
    return _ffi_api.TETensor(value, tir_var_map, name)  # type: ignore


def get_shape_of(expr: Expr) -> Expr:
    """Get shape of expr.

    Parameters
    ----------
    expr: Expr
        The input expr.

    Returns
    -------
    shape: Expr
        The shape expression

    Note
    ----
    This function requires expr to be normalized.
    The function will report an error if expr's StructInfo is not TensorStructInfo.
    It will try to return symbolic function when possible. If the tensor do not
    have a compile-time symbolic shape, the function will then choose to return
    `Call(relax.op.shape_of, [expr])`.
    """
    return _ffi_api.GetShapeOf(expr)  # type: ignore


def _update_struct_info(expr: Expr, struct_info: Optional[StructInfo]) -> None:
    _ffi_api.UpdateStructInfo(expr, struct_info)  # type: ignore
