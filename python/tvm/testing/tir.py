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
# pylint: disable=invalid-name, import-outside-toplevel, unused-variable
"""Common utility functions in TVM tir"""
import inspect
import re
import tvm
from tvm.ir.diagnostics import override_renderer


CHECK_ERROR_RE = re.compile(r"^.*# check_error: (.+)$")


def check_error(func, rel_lineno):
    """check if TIR script throws error"""
    # Override the default renderer to accumulate errors
    errors = []

    def render(e):
        for d in e.diagnostics:
            errors.append(d)

    override_renderer(render)
    # The diagnostic context throws an exception when it gets an error
    try:
        source_code = inspect.getsource(func)
        source_code = "@T.prim_func\n" + source_code
        from tvm.script import from_source

        # to avoid cyclic import
        from_source(source_code)
    except tvm.error.DiagnosticError as e:
        pass
    assert len(errors) == 1, errors
    for d in errors:
        assert (
            d.span.line - 1 == rel_lineno
        ), f"Expected error to be on line {rel_lineno}, but it was on {d.span.line - 1}"

    error_line = source_code.split("\n")[rel_lineno]
    m = CHECK_ERROR_RE.match(error_line)
    if m:
        expected_error_text = m.group(1)
        errors = [e.message for e in errors]
        assert (
            expected_error_text in errors
        ), f'check_error expects "{expected_error_text} in str(errors): {errors}'


def mma_schedule(
    workload,
    k_inner,
    in_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
    shared_scope="shared",
):
    """Create a tensorized schedule for 4k GEMM with MMA intrinsics."""
    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)

    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, k_inner])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)

    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    num_ty = i_factors[2] * j_factors[2]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, shared_scope)
        sch.compute_at(block_read, k0)
        vector_size = 16 if in_dtype == "int8" else 8
        warp_size = 32
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        offset = 8 if in_dtype == "float16" else 16
        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    A_warp = sch.cache_read(block_outer, 0, "warp")
    B_warp = sch.cache_read(block_outer, 1, "warp")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    C_warp = sch.cache_write(block_outer, 0, "warp")
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_block("C_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)

    sch.transform_layout(A_warp, ("write", 0), index_map_A)
    sch.transform_layout(B_warp, ("write", 0), index_map_B)
    sch.transform_layout(C_warp, ("read", 0), index_map_C)

    sch.tensorize(loop_a, ldmatrix_a_intrin)
    sch.tensorize(loop_b, ldmatrix_b_intrin)
    sch.tensorize(sch.get_loops(block_inner)[-3], mma_intrin)
    sch.tensorize(sch.get_loops(block_init_c)[-2], mma_fill_intrin)
    sch.tensorize(sch.get_loops(C_warp)[-2], mma_store_intrin)

    return sch
