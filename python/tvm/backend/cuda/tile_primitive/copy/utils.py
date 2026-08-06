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

from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import match_scope, validate_copy_op


def _single_thread_exec(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Predicate: exec scope must be single-thread."""
    exec_scope = sctx.scope_kind
    ok = exec_scope == "thread"
    return (ok, None if ok else f"expected thread exec_scope, got {exec_scope}")


DEFAULT_ALLOWED_PAIRS: tuple[tuple[str, str], ...] = (
    ("global", "shared*"),
    ("shared*", "global"),
    ("global", "local"),
    ("local", "global"),
    ("shared*", "local"),
    ("local", "shared*"),
)


def _scope_allowed(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    allowed_pairs: Iterable[tuple[str, str]] = DEFAULT_ALLOWED_PAIRS,
):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = (op_call.dst, op_call.src)
    src_scope = src_buffer_region.buffer.scope()
    dst_scope = dst_buffer_region.buffer.scope()
    ok = any(
        (
            match_scope(src_scope, src_pat) and match_scope(dst_scope, dst_pat)
            for src_pat, dst_pat in allowed_pairs
        )
    )
    if not ok:
        allowed_str = ", ".join((f"{a}->{b}" for a, b in allowed_pairs))
        return (
            False,
            f"unsupported memory scopes src={src_scope} dst={dst_scope}; allowed: {allowed_str}",
        )
    return (True, None)


def _is_valid_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    return (validate_copy_op(op_call, sctx), "validate_copy_op failed")
