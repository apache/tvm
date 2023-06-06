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

import tvm._ffi

from ..runtime import Scriptable, const, convert
from . import _ffi_api
from .base import Node


class BaseExpr(Node):
    """Base class of all the expressions."""


class PrimExpr(BaseExpr):
    """Base class of all primitive expressions.

    PrimExpr is used in the low-level code
    optimizations and integer analysis.
    """


class RelayExpr(BaseExpr):
    """Base class of all non-primitive expressions."""

    @property
    def checked_type(self):
        """Get the checked type of tvm.relay.Expr.

        Returns
        -------
        checked_type : tvm.relay.Type
            The checked type.
        """
        ret = self._checked_type_
        if ret is None:
            raise ValueError("The type checker has not populated the checked_type for this node")
        return ret


@tvm._ffi.register_object("GlobalVar")
class GlobalVar(RelayExpr):
    """A global variable in the IR.

    GlobalVar is used to refer to the global functions
    stored in the IRModule.

    Parameters
    ----------
    name_hint: str
        The name of the variable.
    """

    def __init__(self, name_hint, type_annot=None):
        self.__init_handle_by_constructor__(_ffi_api.GlobalVar, name_hint, type_annot)

    def __call__(self, *args):
        """Call the global variable.

        Parameters
        ----------
        args: List[RelayExpr]
            The arguments to the call.

        Returns
        -------
        call: BaseExpr
            A call taking the variable as a function.
        """
        # pylint: disable=import-outside-toplevel
        if all(isinstance(x, RelayExpr) for x in args):
            from tvm import relay

            return relay.Call(self, args)
        elif all(isinstance(x, (Number, PrimExpr)) for x in args):
            return tvm.tir.call_tir(self, *args)

        arg_types = [type(x) for x in args]
        raise RuntimeError(f"Do not know how to handle GlobalVar.__call__ for types {arg_types}")

    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[Object->str]
            Optionally annotate function to provide additional
            information in the comment block.

        Returns
        -------
        text : str
            The text format of the expression.

        Notes
        -----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.
        """
        from tvm.relay import astext  # pylint: disable=import-outside-toplevel

        return astext(self, show_meta_data, annotate)


@tvm._ffi.register_object
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
        The location of this itervar in the source code.

    Note
    ----
    The constructor creates the range `[begin, end)`
    if the end argument is not None. Otherwise, it creates `[0, begin)`.
    """

    def __init__(self, begin, end=None, span=None):
        if end is None:
            end = convert(begin)
            begin = const(0, dtype=end.dtype, span=span)
        self.__init_handle_by_constructor__(_ffi_api.Range, begin, end, span)

    @staticmethod
    def from_min_extent(min_value, extent, span=None):
        """Construct a Range by min and extent.

        This constructs a range in [min_value, min_value + extent)

        Parameters
        ----------
        min_value : PrimExpr
            The minimum value of the range.

        extent : PrimExpr
            The extent of the range.

        span : Optional[Span]
            The location of this itervar in the source code.

        Returns
        -------
        rng : Range
            The constructed range.
        """
        return _ffi_api.Range_from_min_extent(min_value, extent, span)
