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

from tvm.tirx import Buffer
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import match_scope, validate_copy_op


def _is_valid_smem_tmem_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Validate smem->tmem copy operation.

    The new tcgen05.cp.32x128b.warpx4 dispatch requires the destination tmem
    buffer to declare warpx4 broadcast as ``R[4 : 32@TLane]``. The legacy
    128-row dispatch path (no replica) goes through a separate code path and
    is not handled here.
    """
    dst_region, src_region = op_call.args[:2]
    src: Buffer = src_region.buffer
    dst: Buffer = dst_region.buffer
    if not (src.scope().startswith("shared") and dst.scope() == "tmem"):
        return (False, f"expected shared->tmem, got {src.scope()}->{dst.scope()}")
    if not (src.layout and dst.layout):
        return (False, "both buffers must have layouts")
    if dst.allocated_addr is None:
        return (False, "tmem buffer must have allocated_addr")
    # Require warpx4 router on TMEM side so this dispatch only handles the
    # 32x128b.warpx4 case; other shapes (128x256b/128x128b etc.) fall back
    # to the legacy dispatch.
    rep = dst.layout.replica
    if not (
        len(rep) == 1
        and int(rep[0].extent) == 4
        and int(rep[0].stride) == 32
        and "TLane" in str(rep[0].axis)
    ):
        return (False, f"requires R[4:32@TLane] on tmem, got replica={list(rep)}")
    return (True, None)


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
