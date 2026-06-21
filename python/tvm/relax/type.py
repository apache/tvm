# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-import
# ruff: noqa: F401
"""The Relax type nodes, including richer dependent type nodes."""

import tvm_ffi
from tvm_ffi import Array

import tvm
from tvm.ir import EnvFunc, Span, TupleType, VDevice
from tvm.runtime import DataType
from tvm.tirx import PrimExpr

from . import _ffi_api
from .expr import Expr, ShapeExpr, Type
from .ty import PackedFuncType


@tvm_ffi.register_object("relax.ObjectType")
class ObjectType(Type):
    """Type of an Object."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ObjectType, span)  # type: ignore


@tvm_ffi.register_object("relax.PrimType")
class PrimType(Type):
    """Type of a primitive POD value.

    Parameters
    ----------
    dtype_or_expr : Union[str, DataType, PrimExpr]

       The data type of the prim value, or a known expression for the prim
       value.
    """

    value: PrimExpr | None
    dtype: str

    def __init__(
        self,
        dtype: str | DataType | None = None,
        value: int | float | PrimExpr | None = None,
        span: Span = None,
    ) -> None:
        # Guard against incorrect usage.  For backwards compatibility,
        # the dtype and value are in the opposite order from most
        # usages.  While PrimType could take a single positional
        # argument and check the type, this would require an API
        # difference from TVMScript's PrimProxy, which cannot.
        # (PrimProxy uses string arguments for datatype, and also for
        # inline variable definitions when used in a function
        # signature, and requires separate arguments to distinguish
        # the two cases.)
        if isinstance(dtype, PrimExpr | int | float):
            raise TypeError(
                f"The first positional argument of PrimType must be the datatype, "
                f", but received {type(dtype)}.  "
                f"The value can be specified as a keyword argument "
                f"without needing specifying the dtype: "
                f"PrimType(value=arg)."
            )

        if dtype is None and value is None:
            raise TypeError(
                "PrimType.__init__ missing required argument.  "
                "Must provide either 'dtype' or 'value'"
            )

        if dtype is not None:
            if isinstance(value, PrimExpr):
                assert value.dtype == dtype, (
                    "When providing both 'value' and 'dtype' to PrimType.__init__, "
                    "they must be consistent with each other.  "
                    "However, the value {value} has dtype {value.dtype}, "
                    "but the specified dtype was {dtype}."
                )
            elif isinstance(value, int | float):
                value = tvm.tirx.const(value, dtype)

        # Use relax's default integer type if not otherwise specified.
        if isinstance(value, int):
            value = tvm.tirx.IntImm("int64", value)

        if value is None:
            self.__init_handle_by_constructor__(_ffi_api.PrimTypeFromDtype, dtype, span)  # type: ignore
        else:
            self.__init_handle_by_constructor__(_ffi_api.PrimTypeFromValue, value, span)  # type: ignore


@tvm_ffi.register_object("relax.ShapeType")
class ShapeType(Type):
    """Type of a shape value.

    Parameters
    ----------
    values : Optional[List[PrimExpr]]
       The symbolic shape values if known.

    ndim : Optional[int]
       The size of the shape.

    Note
    ----
    Do not specify values and ndim at the same time.
    """

    values: list[PrimExpr] | None
    ndim: int
    span: Span

    def __init__(
        self, values: list[PrimExpr] | None = None, ndim: int = -1, span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ShapeType,
            values,
            ndim,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.TensorType")
class TensorType(Type):
    """Type of a Tensor value.

    Parameters
    ----------
    shape : Optional[Expr]
       The shape expression.

    dtype : Optional[str]
        The content data type.

    vdevice : Optional[Vdevice]
        The virtual device.

    ndim : Optional[int]
       The number of dimensions of the tensor.

    Note
    ----
    Do not specify shape and ndim at the same time.
    """

    shape: Expr | None
    dtype: str
    vdevice: VDevice | None
    ndim: int
    span: Span

    def __init__(
        self,
        shape: Expr | None | list[PrimExpr] = None,
        dtype: str = "float32",
        vdevice: VDevice | None | str = None,
        ndim: int = -1,
        span: Span = None,
    ) -> None:
        if isinstance(shape, list | tuple | Array):
            shape = ShapeExpr(shape)
        self.__init_handle_by_constructor__(
            _ffi_api.TensorType,
            shape,
            dtype,
            ndim,
            vdevice,
            span,  # type: ignore
        )


@tvm_ffi.register_object("relax.FuncType")
class FuncType(Type):
    """Type of a function value.

    Parameters
    ----------
    params: List[Type]
        The type of the fields.

    ret: Type
        The type of return value

    purity: bool
        Whether the function is pure (has no visible side effects).
        Note: We consider a function to be pure only if it is pure on all inputs.
        If a function can have visible side effects only in some cases,
        we still consider it impure.
    """

    params: list[Type] | None
    ret: Type
    derive_func: EnvFunc | None
    purity: bool
    span: Span

    def __init__(
        self, params: list[Type], ret: Type, purity: bool = True, span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.FuncType,
            params,
            ret,
            purity,
            span,  # type: ignore
        )

    @staticmethod
    def opaque_func(
        *,
        ret: Type | None = None,
        derive_func: str | EnvFunc | None = None,
        purity: bool = False,
        span: Span = None,
    ) -> "FuncType":
        """
        Create an opaque FuncType.

        The opaque function takes either a ret
        that specificies the type of the return value
        or a derive_func that provides a customized derivation rule.

        Parameters
        ----------
        ret: Optional[Type]
           The type of the function return value.

        derive_func: Optional[Union[str,EnvFunc]]
           The environment function used for derivation

        purity: bool
           Whether the function is pure (false by default, as most opaque functions are not pure)

        span: Optional[Span]
           Optional span information of the ast.

        Returns
        -------
        info: FuncType

        Note
        ----
        We cannot specify ret and derive_func simultaneously.
        """

        if isinstance(derive_func, str):
            derive_func = tvm.ir.EnvFunc.get("tvm.relax.type.infer_view_ty")
        return _ffi_api.FuncTypeOpaqueFunc(ret, derive_func, purity, span)  # type: ignore
