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

"""Common utilities for CUDA operator scheduling (basic helpers and copy ops)."""

import functools
import operator
import re
from enum import Enum

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.stmt import TilePrimitiveCall


def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 greater than or equal to x."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_st_extent(buffer_region: BufferRegion):
    """Get the start and extent of a buffer region."""
    region = buffer_region.region
    return [r.min for r in region], [r.extent for r in region]


def get_indices(nth, start, extent):
    """Convert a fused index into multi-dimensional indices."""
    assert len(start) == len(extent)
    if len(start) == 1:
        return [start[0] + nth]
    relative = []
    for e in reversed(extent):
        relative.append(nth % e)
        nth //= e
    return [r + s for r, s in zip(reversed(relative), start)]


def smem_desc_add_16B_offset(desc_val, offset):
    """Add a 16B-aligned byte offset to the lower 32 bits of a SMEM descriptor.

    Uses the SmemDescriptor union defined in the CUDA header (header.py).
    All callers must share a single implementation to avoid codegen conflicts.
    """
    func_name = "tvm_builtin_smem_desc_add_16B_offset"
    source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
    return Tx.cuda.func_call(
        func_name, desc_val, offset, source_code=source_code, return_type="uint64"
    )


class CopyInstType(Enum):
    """Enumeration of instruction types for memory operations."""

    NORMAL = 0
    CP_ASYNC = 1


def validate_copy_op(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,  # pylint: disable=unused-argument
) -> bool:
    """Sanity check for copy op"""
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (src.layout and dst.layout and src.dtype == dst.dtype):
        return False
    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_region, dst_region = src_buffer_region.region, dst_buffer_region.region
    # Extract extents and validate non-unit dimensions match
    src_extent_ = [r.extent for r in src_region if r.extent != 1]
    dst_extent_ = [r.extent for r in dst_region if r.extent != 1]
    if len(src_extent_) != len(dst_extent_) or not all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_, dst_extent_)
    ):
        return False
    return True


def get_vec_len(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    vec_candidates: list[int],
    thread_cnt=1,
) -> int | None:
    """Get the vector length for the copy operation."""

    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer
    # layout=None (flat local buffer) is treated as trivial for vectorization purposes
    if not (
        (dst.layout is None or dst.layout.is_trivial())
        and (src.layout is None or src.layout.is_trivial())
    ):
        return None

    # Extract regions and validate dimensions
    analyzer = Analyzer()
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    # Thread and vectorization setup
    DataType(src.dtype).bits  # in bits
    n_elements = functools.reduce(operator.mul, src_extent, 1)
    if n_elements % thread_cnt != 0:
        return None

    # Find valid vector length
    for vec_len in vec_candidates:
        if vec_len > 0 and all(
            analyzer.can_prove_equal(x % vec_len, 0)
            for x in [
                src_st[-1],
                dst_st[-1],
                src.shape[-1] if len(src.shape) > 1 else 0,
                dst.shape[-1] if len(dst.shape) > 1 else 0,
                src_extent[-1],
                dst_extent[-1],
                n_elements // thread_cnt,
            ]
        ):
            return vec_len
    else:
        return None


def copy_vec_load_impl(
    op_call: TilePrimitiveCall, sctx: DispatchContext, inst_type: CopyInstType
) -> PrimFunc | None:
    """Schedule copy operation between global and local/shared memory on CUDA across a CTA/thread.
    The implementation tries to vectorize the copy operation and parallelize over
    threads in a CTA/using a single thread.
    """
    dst_buffer_region, src_buffer_region = op_call.args[:2]
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    if not (
        (src.scope() == "global" and dst.scope().startswith("shared"))
        or (src.scope().startswith("shared") and dst.scope() == "global")
        or (src.scope() == "global" and dst.scope() == "local")
        or (src.scope() == "local" and dst.scope() == "global")
        or (src.scope().startswith("shared") and dst.scope() == "local")
        or (dst.scope().startswith("shared") and src.scope() == "local")
    ):
        fail(f"unsupported memory scopes src={src.scope()} dst={dst.scope()}")

    # Thread and vectorization setup
    if sctx.is_cta:
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params
    elif sctx.is_thread:
        tx = 1
    else:
        fail(f"unsupported exec_scope {sctx.scope_kind}")

    elem_size = DataType(src.dtype).bits  # in bits
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
        vec_len = get_vec_len(
            dst_buffer_region,
            src_buffer_region,
            [128 // elem_size, 64 // elem_size, 32 // elem_size, 1],
            thread_cnt=tx,
        )
    if vec_len is None:
        fail("no valid vector length; check alignment/extents/thread-count")

    # cp-size (the size of data in bytes) can only be 4, 8 and 16 for cp.async
    if inst_type == CopyInstType.CP_ASYNC:
        cp_size = vec_len * elem_size // 8  # in bytes
        if cp_size not in [4, 8, 16]:
            fail("invalid cp.async cp_size; expected 4, 8 or 16 bytes")

    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)
    n_elements = functools.reduce(operator.mul, src_extent, 1)

    if sctx.is_cta:
        # fmt: off
        @Tx.prim_func
        def impl():
            """Implement copy operation with vectorized loads/stores."""
            for s in Tx.serial(0, n_elements // (tx * vec_len)):
                for tid_x in Tx.thread_binding(tx, "threadIdx.x"):
                    if inst_type == CopyInstType.NORMAL:
                        for vec in Tx.vectorized(vec_len):
                            fused = Tx.meta_var((s * tx + tid_x) * vec_len + vec)
                            dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                            src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                            dst[tuple(dst_indices)] = src[tuple(src_indices)]
                    elif inst_type == CopyInstType.CP_ASYNC:
                        fused = Tx.meta_var((s * tx + tid_x) * vec_len)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                        Tx.evaluate(Tx.ptx.cp_async(dst.ptr_to(dst_indices), src.ptr_to(src_indices), cp_size))  # noqa: E501
            if dst.scope().startswith("shared") and inst_type == CopyInstType.NORMAL:
                Tx.tvm_storage_sync("shared")
        # fmt: on
    elif sctx.is_thread:
        # fmt: off
        @Tx.prim_func(check_well_formed=False)
        def impl():
            for s in Tx.serial(0, n_elements // (vec_len)):
                if inst_type == CopyInstType.NORMAL:
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                        src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                        dst[tuple(dst_indices)] = src[tuple(src_indices)]
                elif inst_type == CopyInstType.CP_ASYNC:
                    fused = Tx.meta_var(s * vec_len)
                    dst_indices = Tx.meta_var(get_indices(fused, dst_st, dst_extent))
                    src_indices = Tx.meta_var(get_indices(fused, src_st, src_extent))
                    Tx.evaluate(Tx.ptx.cp_async(dst.ptr_to(dst_indices), src.ptr_to(src_indices), cp_size))  # noqa: E501
        # fmt: on
    else:
        fail(f"unsupported exec_scope {sctx.scope_kind}")
    return impl


def match_scope(scope: str | None, pattern: str) -> bool:
    """Glob-lite scope matching: 'shared*' => prefix match; otherwise exact.

    Returns True when scope is None (meaning "any scope is fine").
    """
    if scope is None:
        return True
    if pattern.endswith("*"):
        return scope.startswith(pattern[:-1])
    return scope == pattern


def get_thread_cnt(sctx: DispatchContext) -> int | None:
    """Get thread count for the current execution scope."""
    scope_name = sctx.scope_kind
    if scope_name == "cta":
        return sctx.launch_params["threadIdx.x"].dom.extent
    if scope_name == "warpgroup":
        return 128
    if scope_name == "warp":
        return 32
    if scope_name == "thread":
        return 1
    return None


def sm_version_ok(
    op: TilePrimitiveCall, sctx: DispatchContext, min_version: int
) -> tuple[bool, str | None]:
    """Check if SM version >= min_version. Usable as a dispatch predicate."""
    target_arch = sctx.target.arch if hasattr(sctx.target, "arch") else ""
    sm_match = re.match(r"sm_(\d+)", target_arch)
    sm_version = int(sm_match.group(1)) if sm_match else 0
    ok = sm_version >= min_version
    return (ok, None if ok else f"sm_version {sm_version} < {min_version}")
