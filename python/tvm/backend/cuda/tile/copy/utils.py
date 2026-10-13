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
"""Shared helpers for copy operator dispatches on CUDA targets."""

from collections.abc import Iterable

from tvm.runtime import MemoryScope
from tvm.tirx.tile.dispatch import DispatchContext
from tvm.tirx.tile.instruction import TensorCall

from ..common import validate_copy_op

_SCOPE_GLOBAL = MemoryScope("global")
_SCOPE_LOCAL = MemoryScope("local")
_SCOPE_SHARED = MemoryScope("shared")
_SCOPE_SHARED_DYN = MemoryScope("shared.dyn")


def _single_thread_exec(op_call: TensorCall, sctx: DispatchContext):
    """Predicate: exec scope must be single-thread."""
    exec_scope = sctx.scope_kind
    ok = exec_scope == "thread"
    return (ok, None if ok else f"expected thread exec_scope, got {exec_scope}")


DEFAULT_ALLOWED_PAIRS = (
    (_SCOPE_GLOBAL, _SCOPE_SHARED),
    (_SCOPE_GLOBAL, _SCOPE_SHARED_DYN),
    (_SCOPE_SHARED, _SCOPE_GLOBAL),
    (_SCOPE_SHARED_DYN, _SCOPE_GLOBAL),
    (_SCOPE_GLOBAL, _SCOPE_LOCAL),
    (_SCOPE_LOCAL, _SCOPE_GLOBAL),
    (_SCOPE_SHARED, _SCOPE_LOCAL),
    (_SCOPE_SHARED_DYN, _SCOPE_LOCAL),
    (_SCOPE_LOCAL, _SCOPE_SHARED),
    (_SCOPE_LOCAL, _SCOPE_SHARED_DYN),
)


def _scope_allowed(
    op_call: TensorCall,
    sctx: DispatchContext,
    allowed_pairs: Iterable[tuple[MemoryScope, MemoryScope]] = DEFAULT_ALLOWED_PAIRS,
):
    op_call = TensorCall.decode(op_call)
    dst_buffer_region, src_buffer_region = (op_call.dst, op_call.src)
    src_scope = src_buffer_region.source.scope()
    dst_scope = dst_buffer_region.source.scope()
    ok = (src_scope, dst_scope) in allowed_pairs
    if not ok:
        allowed_str = ", ".join((f"{a.name}->{b.name}" for a, b in allowed_pairs))
        return (
            False,
            f"unsupported memory scopes src={src_scope} dst={dst_scope}; allowed: {allowed_str}",
        )
    return (True, None)


def _is_valid_copy(op_call: TensorCall, sctx: DispatchContext):
    return (validate_copy_op(op_call, sctx), "validate_copy_op failed")
