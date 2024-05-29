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
"""The struct info nodes of the Relax language."""
from typing import List, Optional, Union

import tvm._ffi
import tvm

from tvm.ir import Span, EnvFunc, Array, VDevice
from tvm.tir import PrimExpr
from tvm.runtime import DataType
from .expr import StructInfo, Expr, ShapeExpr

from . import _ffi_api, ty, expr


@tvm._ffi.register_object("relax.ObjectStructInfo")
class ObjectStructInfo(StructInfo):
    """StructInfo of an Object."""

    def __init__(self, span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ObjectStructInfo, span)  # type: ignore


@tvm._ffi.register_object("relax.PrimStructInfo")
class PrimStructInfo(StructInfo):
    """StructInfo of a primitive POD value.

    Parameters
    ----------
    dtype_or_expr : Union[str, DataType, PrimExpr]

       The data type of the prim value, or a known expression for the prim
       value.
    """

    value: Optional[PrimExpr]
    dtype: str

    def __init__(
        self,
        dtype: Optional[Union[str, DataType]] = None,
        value: Optional[Union[int, float, PrimExpr]] = None,
        span: Span = None,
    ) -> None:
        # Guard against incorrect usage.  For backwards compatibility,
        # the dtype and value are in the opposite order from most
        # usages.  While PrimStructInfo could take a single positional
        # argument and check the type, this would require an API
        # difference from TVMScript's PrimProxy, which cannot.
        # (PrimProxy uses string arguments for datatype, and also for
        # inline variable definitions when used in a function
        # signature, and requires separate arguments to distinguish
        # the two cases.)
        if isinstance(dtype, (PrimExpr, int, float)):
            raise TypeError(
                f"The first positional argument of PrimStructInfo must be the datatype, "
                f", but received {type(dtype)}.  "
                f"The value can be specified as a keyword argument "
                f"without needing specifying the dtype: "
                f"PrimStructInfo(value=arg)."
            )

        if dtype is None and value is None:
            raise TypeError(
                "PrimStructInfo.__init__ missing required argument.  "
                "Must provide either 'dtype' or 'value'"
            )

        if dtype is not None:
            if isinstance(value, PrimExpr):
                assert value.dtype == dtype, (
                    "When providing both 'value' and 'dtype' to PrimStructInfo.__init__, "
                    "they must be consistent with each other.  "
                    "However, the value {value} has dtype {value.dtype}, "
                    "but the specified dtype was {dtype}."
                )
            elif isinstance(value, (int, float)):
                value = tvm.tir.const(value, dtype)

        # Use relax's default integer type if not otherwise specified.
        if isinstance(value, int):
            value = tvm.tir.IntImm("int64", value)

        if value is None:
            self.__init_handle_by_constructor__(
                _ffi_api.PrimStructInfoFromDtype, dtype, span
            )  # type: ignore
        else:
            self.__init_handle_by_constructor__(
                _ffi_api.PrimStructInfoFromValue, value, span
            )  # type: ignore


@tvm._ffi.register_object("relax.ShapeStructInfo")
class ShapeStructInfo(StructInfo):
    """StructInfo of a shape value.

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

    values: Optional[List[PrimExpr]]
    ndim: int
    span: Span

    def __init__(
        self, values: Optional[List[PrimExpr]] = None, ndim: int = -1, span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ShapeStructInfo, values, ndim, span  # type: ignore
        )


@tvm._ffi.register_object("relax.TensorStructInfo")
class TensorStructInfo(StructInfo):
    """StructInfo of a Tensor value.

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

    shape: Optional[Expr]
    dtype: str
    vdevice: Optional[VDevice]
    ndim: int
    span: Span

    def __init__(
        self,
        shape: Union[Optional[Expr], List[PrimExpr]] = None,
        dtype: str = "float32",
        vdevice: Union[Optional[VDevice], str] = None,
        ndim: int = -1,
        span: Span = None,
    ) -> None:
        if isinstance(shape, (list, tuple, Array)):
            shape = ShapeExpr(shape)
        self.__init_handle_by_constructor__(
            _ffi_api.TensorStructInfo, shape, dtype, ndim, vdevice, span  # type: ignore
        )


@tvm._ffi.register_object("relax.TupleStructInfo")
class TupleStructInfo(StructInfo):
    """StructInfo of a Tuple value.

    Parameters
    ----------
    fields: List[StructInfo]
        The struct info of the fields.
    """

    fields: List[StructInfo]
    span: Span

    def __init__(self, fields: List[StructInfo], span: Span = None) -> None:
        self.__init_handle_by_constructor__(_ffi_api.TupleStructInfo, fields, span)  # type: ignore


@tvm._ffi.register_object("relax.FuncStructInfo")
class FuncStructInfo(StructInfo):
    """StructInfo of a function value.

    Parameters
    ----------
    params: List[StructInfo]
        The struct info of the fields.

    ret: StructInfo
        The struct info of return value

    purity: bool
        Whether the function is pure (has no visible side effects).
        Note: We consider a function to be pure only if it is pure on all inputs.
        If a function can have visible side effects only in some cases,
        we still consider it impure.
    """

    params: Optional[List[StructInfo]]
    ret: StructInfo
    derive_func: Optional[EnvFunc]
    purity: bool
    span: Span

    def __init__(
        self, params: List[StructInfo], ret: StructInfo, purity: bool = True, span: Span = None
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.FuncStructInfo, params, ret, purity, span  # type: ignore
        )

    @staticmethod
    def opaque_func(
        *,
        ret: Optional[StructInfo] = None,
        derive_func: Optional[Union[str, EnvFunc]] = None,
        purity: bool = False,
        span: Span = None,
    ) -> "FuncStructInfo":
        """
        Create an opaque FuncStructInfo.

        The opaque function takes either a ret
        that specificies the struct info of the return value
        or a derive_func that provides a customized derivation rule.

        Parameters
        ----------
        ret: Optional[StructInfo]
           The struct info of the function return value.

        derive_func: Optional[Union[str,EnvFunc]]
           The environment function used for derivation

        purity: bool
           Whether the function is pure (false by default, as most opaque functions are not pure)

        span: Optional[Span]
           Optional span information of the ast.

        Returns
        -------
        info: FuncStructInfo

        Note
        ----
        We cannot specify ret and derive_func simultaneously.
        """

        if isinstance(derive_func, str):
            derive_func = tvm.ir.EnvFunc.get("tvm.relax.struct_info.infer_view_sinfo")
        return _ffi_api.FuncStructInfoOpaqueFunc(ret, derive_func, purity, span)  # type: ignore
