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
"""Common expressions data structures in the IR."""
from numbers import Number
from typing import Optional

import tvm.ffi

from ..runtime import Object, Scriptable
from . import _ffi_api
from .base import Node, Span


@tvm.ffi.register_object("ir.BaseExpr")
class BaseExpr(Node):
    """Base class of all the expressions."""

    span: Optional[Span]


@tvm.ffi.register_object("ir.PrimExpr")
class PrimExpr(BaseExpr):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """

    dtype: str


@tvm.ffi.register_object("ir.RelaxExpr")
class RelaxExpr(BaseExpr):
    """Base class of all non-primitive expressions."""

    @property
    def struct_info(self) -> Optional["tvm.relax.StructInfo"]:
        """Get the struct info field

        Returns
        -------
        struct_info : tvm.relax.StructInfo
            The struct info if available.
        """
        return _ffi_api.ExprStructInfo(self)


@tvm.ffi.register_object("ir.GlobalVar")
class GlobalVar(RelaxExpr):
    """A global variable in the IR.

    GlobalVar is used to refer to the global functions
    stored in the IRModule.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """

    name_hint: str

    def __init__(self, name_hint: str):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, name_hint)

    def __call__(self, *args: RelaxExpr) -> BaseExpr:
        """Call the global variable.

        Parameters
        ----------
        args: List[RelaxExpr]
            The arguments to the call.

        Returns
        -------
        call: BaseExpr
            A call taking the variable as a function.
        """
        # pylint: disable=import-outside-toplevel

        # TODO(@relax-team): replace with Relax base class after it's introduced
        if all(isinstance(x, RelaxExpr) for x in args):
            from tvm import relax

            return relax.Call(self, args)

        elif all(isinstance(x, (Number, PrimExpr)) for x in args):
            return tvm.tir.call_tir(self, *args)

        arg_types = [type(x) for x in args]
        raise RuntimeError(f"Do not know how to handle GlobalVar.__call__ for types {arg_types}")


@tvm.ffi.register_object("ir.Range")
class Range(Node, Scriptable):
    """Represent a range in TVM.

    You do not need to create a Range explicitly.
    Python lists and tuples will be converted automatically to a Range in API functions.

    Parameters
    ----------
    begin : PrimExpr
        The begin value of the range when end is None.
        Otherwise it is the length of the range.

    end : Optional[PrimExpr]
        The end value of the range.

    span : Optional[Span]
        The location of this node in the source code.

    Note
    ----
    The constructor creates the range `[begin, end)`
    if the end argument is not None. Otherwise, it creates `[0, begin)`.
    """

    min: PrimExpr
    extent: PrimExpr
    span: Optional[Span]

    def __init__(
        self, begin: PrimExpr, end: Optional[PrimExpr] = None, span: Optional[Span] = None
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, span)

    @staticmethod
    def from_min_extent(
        min_value: PrimExpr, extent: PrimExpr, span: Optional[Span] = None
    ) -> "Range":
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : PrimExpr
            The minimum value of the range.

        extent : PrimExpr
            The extent of the range.

        span : Optional[Span]
            The location of this node in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.Range_from_min_extent(min_value, extent, span)

    def __eq__(self, other: Object) -> bool:
        return tvm.ir.structural_equal(self, other)

    def __ne__(self, other: Object) -> bool:
        return not self.__eq__(other)
