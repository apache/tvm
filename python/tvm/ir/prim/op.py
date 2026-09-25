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
"""Construction helpers for shared primitive expressions."""

from threading import local
from typing import Any

from ..base import Span
from ..expr import Call, Expr
from . import _ffi_api


class OpConstFoldScope:
    """Control eager primitive-operator folding on the current thread.

    Parameters
    ----------
    enabled : bool
        Whether primitive construction folds constants and identities. The default
        outside a scope is True. Explicit symbolic simplification is unaffected.

    Examples
    --------
    .. code-block:: python

        with tvm.ir.prim.OpConstFoldScope(enabled=False):
            expr = tvm.ir.prim.const(1) + tvm.ir.prim.const(2)
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._local = local()

    def __enter__(self):
        if not hasattr(self._local, "previous"):
            self._local.previous = []
        self._local.previous.append(_ffi_api._OpConstFoldSwapEnabled(self.enabled))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _ffi_api._OpConstFoldSwapEnabled(self._local.previous.pop())


def op_const_fold_enabled() -> bool:
    """Return whether primitive construction eagerly folds on the current thread."""
    return _ffi_api.op_const_fold_enabled()


def convert(expr) -> Expr:
    """Convert a scalar or sequence to primitive expressions."""
    return _ffi_api.convert(expr)


def min_value(dtype, span=None):
    """minimum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The minimum value of dtype.
    """
    return _ffi_api.min_value(dtype, span)  # type: ignore


def max_value(dtype: str, span: Span | None = None) -> Any:
    """maximum value of dtype

    Parameters
    ----------
    dtype : str
        The data type.

    span : Optional[Span]
        The location of this operator in the source code.

    Returns
    -------
    value : tvm.ir.Expr
        The maximum value of dtype.
    """
    return _ffi_api.max_value(dtype, span)  # type: ignore


def clz(x):
    """Count leading zero bits of an integer x.

    Parameters
    ----------
    x : Expr
        Input 32 or 64 bit integer.
        The result is undefined if the input is 0.

    Returns
    -------
    y : Expr
        The result.
    """
    return Call("prim.clz", [x], ret_ty="int32")
