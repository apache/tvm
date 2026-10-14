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
"""Lightweight builder receipts and optional native emission hooks.

This module imports no concrete builder. TVM loads its native libraries in
``tvm.base`` before importing the IR package; the optional hook is resolved
once here, after those libraries have registered their packed functions.
Both presence and absence are cached. Loading a builder extension later does
not replace this hook, and choosing an active builder never changes it.
"""

from typing import TYPE_CHECKING, Generic, TypeVar

import tvm_ffi

if TYPE_CHECKING:
    from .expr import Call
    from .stmt import Stmt

_T = TypeVar("_T")


class AlreadyEmitted(Generic[_T]):
    """Hold the exact native object already emitted by a builder.

    Parameters
    ----------
    value : T
        Emitted native object. It is retained without copying or emitting it.

    Attributes
    ----------
    value : T
        The same object, available for identity checks and source-location
        attachment. Builder consumers must check for this receipt before FFI
        conversion, so they do not emit the object again.

    Notes
    -----
    FFI conversion exposes only the wrapped object, not its emission history.
    The historical ``tvm.script.ir_builder.base.AlreadyEmitted`` import is an
    alias of this class.
    """

    __slots__ = ("value",)
    value: _T

    def __init__(self, value: _T) -> None:
        self.value = value

    def __tvm_ffi_object__(self) -> _T:
        """Return the exact wrapped native object without copying or emitting."""
        return self.value


_builder_maybe_emit_void_call = tvm_ffi.get_global_func(
    "ir.builder_maybe_emit_void_call", allow_missing=True
)


def builder_maybe_emit_void_call(call: "Call") -> "Call | AlreadyEmitted[Stmt]":
    """Emit a void operator call once when an IRBuilder is active.

    Parameters
    ----------
    call : Call
        Already constructed call, retaining its explicit or inferred result
        type and source location. Construction and inference themselves have
        no builder side effects.

    Returns
    -------
    Call or AlreadyEmitted[Stmt]
        The original call when the optional native hook is absent, the result
        is non-void, or no builder is active. Otherwise, a receipt containing
        the exact Evaluate statement stored by the active builder. Both
        primitive void and an empty tuple result count as void.

    Notes
    -----
    An active builder with no frames constructs a standalone statement.
    Errors from an invalid active frame propagate normally. Native libraries
    must register the optional hook before this module is imported; lookup
    is performed once, including when the hook is absent.
    """
    if _builder_maybe_emit_void_call is None:
        return call
    statement = _builder_maybe_emit_void_call(call)
    return call if statement is None else AlreadyEmitted(statement)
