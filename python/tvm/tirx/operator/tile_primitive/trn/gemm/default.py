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

"""Implementation of copy operator dispatchs."""

import functools
import operator

from tvm.arith.analyzer import Analyzer
from tvm.ir import assert_structural_equal
from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import init_analyzer
from ..dim_utils import normalize_and_group
from ..instruction_generator import InstructionGenerator
from ..workspace_utils import check_workspace_buffer, largest_psum_per_bank, max_psum_banks


class OperatorKind:
    A = 0
    B = 1
    C = 2


def get_pf_dim_from_buffer_region(
    buffer_region: BufferRegion,
    analyzer: Analyzer,
    operator_kind: OperatorKind,
    transposed: bool = False,
):
    """Extract partition and free dimensions from buffer region."""
    # Find non-unit dimensions
    non_unit_dims = [
        i
        for i in range(len(buffer_region.buffer.shape))
        if not analyzer.can_prove_equal(buffer_region.region[i].extent, 1)
    ]
    assert len(non_unit_dims) == 2, "Only 2D matrix is supported for gemm"

    layout, seps = normalize_and_group(buffer_region.buffer.layout, buffer_region.buffer.shape)
    # Determine partition and free dimensions based on operator kind
    if operator_kind == OperatorKind.A:
        p_dim, f_dim = non_unit_dims[1], non_unit_dims[0]
    elif operator_kind == OperatorKind.B:
        p_dim, f_dim = non_unit_dims[0], non_unit_dims[1]
    else:
        assert not transposed, (
            "Transposed C is implemented by swapping lhs and rhs. No need to specify by user."
        )
        # For C, determine dimensions based on layout
        has_partition = any(
            layout.shard[i].axis.name == "P"
            for i in range(seps[non_unit_dims[0]], seps[non_unit_dims[0] + 1])
        )
        p_dim, f_dim = (
            (non_unit_dims[0], non_unit_dims[1])
            if has_partition
            else (non_unit_dims[1], non_unit_dims[0])
        )

    # Swap dimensions if transposed
    if transposed:
        p_dim, f_dim = f_dim, p_dim

    # Validate partition dimension
    p_exts = [
        layout.shard[i].extent
        for i in range(seps[p_dim], seps[p_dim + 1])
        if layout.shard[i].axis.name == "P"
    ]

    assert functools.reduce(operator.mul, p_exts, 1) == layout.size("P"), (
        f"Accumulation dimension and output non-streaming dimension must contain whole P dimension. "  # noqa: E501
        f"However, the {p_dim} dimension of {buffer_region} does not."
    )

    # Validate free dimension
    assert all(
        layout.shard[i].axis.name in ["F", "Bank"] or layout.shard[i].extent == 1
        for i in range(seps[f_dim], seps[f_dim + 1])
    ), (
        f"Spatial dimension must not contain P. However, the {f_dim} dimension of {buffer_region} does."  # noqa: E501
    )

    return p_dim, f_dim


def matmul_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Schedule GEMM operation on Trainium."""
    # Basic validation checks
    if not (sctx.is_trn() and sctx.scope_kind == "thread"):
        fail("requires Trainium target and thread exec_scope")

    # Extract arguments
    (
        D_buffer_region,
        A_buffer_region,
        B_buffer_region,
        C_buffer_region,
        transpose_A,
        transpose_B,
        alpha,
        beta,
    ) = op.args
    analyzer = init_analyzer(sctx)
    A, B, C, _D = (
        A_buffer_region.buffer,
        B_buffer_region.buffer,
        C_buffer_region.buffer,
        D_buffer_region.buffer,
    )

    # Validate alpha, beta
    assert analyzer.can_prove_equal(alpha, 1) and analyzer.can_prove_equal(beta, 0), (
        "Only alpha=1 and beta=0 are supported"
    )

    # D and C must be the same buffer region
    assert_structural_equal(D_buffer_region, C_buffer_region)

    # Validate buffer properties
    assert all(
        [
            A.layout and B.layout and C.layout,
            A.dtype == B.dtype,
            A.scope() == "trn.sbuf" and B.scope() == "trn.sbuf",
            C.scope() == "trn.psum" or C.scope() == "trn.sbuf",
            A.layout.is_trainium(),
            B.layout.is_trainium(),
            C.layout.is_trainium(),
            A.layout.size("P") == B.layout.size("P"),
        ]
    ), "Invalid buffer layout and scope"

    p_size = A.layout.size("P")
    assert p_size == B.layout.size("P"), "Partition size mismatch"

    # Get partition and free dimensions
    lhs_p_dim, lhs_f_dim = get_pf_dim_from_buffer_region(
        A_buffer_region, analyzer, OperatorKind.A, transpose_A
    )
    rhs_p_dim, rhs_f_dim = get_pf_dim_from_buffer_region(
        B_buffer_region, analyzer, OperatorKind.B, transpose_B
    )
    acc_p_dim, acc_f_dim = get_pf_dim_from_buffer_region(C_buffer_region, analyzer, OperatorKind.C)
    # Swap LHS and RHS if needed based on accumulator dimensions
    swap_lhs_rhs = acc_p_dim > acc_f_dim
    if swap_lhs_rhs:
        lhs_p_dim, rhs_p_dim = rhs_p_dim, lhs_p_dim
        lhs_f_dim, rhs_f_dim = rhs_f_dim, lhs_f_dim
        A, B = B, A
        A_buffer_region, B_buffer_region = B_buffer_region, A_buffer_region

    # Validate dimension compatibility
    assert analyzer.can_prove(
        A_buffer_region.region[lhs_p_dim].extent == B_buffer_region.region[rhs_p_dim].extent
    ), (
        f"Reduction dimension must match, but the {lhs_p_dim} dimension of {A_buffer_region} != the {rhs_p_dim} dimension of {B_buffer_region}"  # noqa: E501
    )

    assert analyzer.can_prove(
        A_buffer_region.region[lhs_f_dim].extent == C_buffer_region.region[acc_p_dim].extent
    ), (
        f"Spatial dimension must match, but the {lhs_f_dim} dimension of {A_buffer_region} != the {acc_p_dim} dimension of {C_buffer_region}"  # noqa: E501
    )

    assert analyzer.can_prove(
        B_buffer_region.region[rhs_f_dim].extent == C_buffer_region.region[acc_f_dim].extent
    ), (
        f"Spatial dimension must match, but the {rhs_f_dim} dimension of {B_buffer_region} != the {acc_f_dim} dimension of {C_buffer_region}"  # noqa: E501
    )

    inst_gen = InstructionGenerator([A_buffer_region, B_buffer_region, C_buffer_region], analyzer)
    inst_gen.link_buffer_regions(A_buffer_region, B_buffer_region, {lhs_p_dim: rhs_p_dim})
    inst_gen.link_buffer_regions(B_buffer_region, C_buffer_region, {rhs_f_dim: acc_f_dim})
    inst_gen.link_buffer_regions(A_buffer_region, C_buffer_region, {lhs_f_dim: acc_p_dim})
    inst_repr = inst_gen.find_max_inst_size_from_one_region(B_buffer_region, [rhs_f_dim])
    inst_repr = inst_gen.fit_inst_tile_to_region(inst_repr, C_buffer_region, [acc_f_dim])
    inst_repr.bound_inst_size(512, analyzer)
    rhs_f = T.Var("rhs_f", "int32")
    lhs_f = T.Var("lhs_f", "int32")
    p = T.Var("p", "int32")
    reduction_b = T.Var("reduction_b", "int32")
    lhs_b = T.Var("lhs_b", "int32")
    rhs_b = T.Var("rhs_b", "int32")
    lhs_f_size = C.layout.size("P")
    inst_gen.bind_inst_iter(
        B_buffer_region, rhs_f, inst_repr.size, inst_repr.stride, is_free_dim=True
    )
    inst_gen.bind_inst_iter(C_buffer_region, lhs_f, lhs_f_size, 1, is_free_dim=False)
    inst_gen.bind_inst_iter(A_buffer_region, p, A.layout.size("P"), 1, is_free_dim=False)
    reduction_b_extent = inst_gen.fill_in_block_dim(A_buffer_region, reduction_b, [lhs_p_dim])
    lhs_b_extent = inst_gen.fill_in_block_dim(A_buffer_region, lhs_b, [lhs_f_dim])
    rhs_b_extent = inst_gen.fill_in_block_dim(B_buffer_region, rhs_b, [rhs_f_dim])

    # FIXME: we need to lower the guard to things like matmul(lhs[...][lhs_guard], rhs[...][rhs_guard], mask=p_guard)  # noqa: E501
    # so we need to separate the guard for lhs_f, rhs_f and p
    # fmt: off
    @T.inline
    def matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, acc, C_as_output, max_psum_slots):  # noqa: E501
        with T.attr(0, "tensorized_nki_instruction", 1):
            for p_loop in T.serial(0, p_size, annotations={"nki_dim": "P"}):
                for lhs_f_loop in T.serial(0, lhs_f_size, annotations={"nki_dim": "lhs_F"}):
                    for rhs_f_loop in T.serial(0, inst_repr.size, annotations={"nki_dim": "rhs_F"}):
                        b_idx = T.meta_var(lhs_b_loop * rhs_b_extent + rhs_b_loop)
                        inst_gen.set_bind_map(A_buffer_region, {lhs_b: lhs_b_loop, lhs_f: lhs_f_loop, p: p_loop, reduction_b: reduction_b_loop})  # noqa: E501
                        inst_gen.set_bind_map(B_buffer_region, {rhs_b: rhs_b_loop, rhs_f: rhs_f_loop, p: p_loop, reduction_b: reduction_b_loop})  # noqa: E501
                        inst_gen.set_bind_map(C_buffer_region, {lhs_f: lhs_f_loop, rhs_f: rhs_f_loop, lhs_b: lhs_b_loop, rhs_b: rhs_b_loop})  # noqa: E501
                        lhs_indices = T.meta_var(inst_gen.generate_indices(A_buffer_region))
                        rhs_indices = T.meta_var(inst_gen.generate_indices(B_buffer_region))
                        C_indices = T.meta_var(inst_gen.generate_indices(C_buffer_region))
                        if inst_gen.make_guard(A_buffer_region) and inst_gen.make_guard(B_buffer_region):  # noqa: E501
                            if C_as_output:
                                T.evaluate(T.nki.matmul(acc[C_indices], A[lhs_indices], B[rhs_indices]))  # noqa: E501
                            else:
                                T.evaluate(T.nki.matmul(acc[b_idx % max_psum_slots, lhs_f_loop, rhs_f_loop], A[lhs_indices], B[rhs_indices]))  # noqa: E501

    if C.scope() == "trn.psum":
        @T.prim_func
        def impl_C_psum():
            for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(lhs_b_extent, rhs_b_extent, reduction_b_extent):  # noqa: E501
                matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, C, True, None)
        return impl_C_psum

    # todo: generalize the process of generating composite matmul + another_op pattern
    # by generating TIR op and reusing existing dispatch rule

    # we will support matmul + epilogue as a user-specified pattern
    # and a matmul fusion pass can help infer the pattern

    acc_psum_shape = (max_psum_banks, p_size, largest_psum_per_bank)
    if "acc_psum" not in op.workspace:
        assert sctx.alloc_only, "Accumulation psum buffer must be specified in workspace. Run tvm.tirx.transform.trn.TrnPrivateBufferAlloc first."  # noqa: E501
        acc_psum = T.buffer(
                acc_psum_shape,
                "float32",
                scope="trn.psum",
                allocated_addr=(0, 0),
                buffer_name="acc_psum"
            )
        sctx.add_alloc_buffer(acc_psum)
        max_psum_slots = max_psum_banks
    else:
        acc_psum = op.workspace["acc_psum"]
        check_workspace_buffer(acc_psum, (p_size, largest_psum_per_bank), "trn.psum")
        max_psum_slots = acc_psum.shape[0]

    @T.prim_func
    def impl_C_sbuf():
        for lhs_b_loop, rhs_b_loop in T.grid(lhs_b_extent, rhs_b_extent):
            for reduction_b_loop in T.serial(0, reduction_b_extent):
                matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, acc_psum, False, max_psum_slots)  # noqa: E501
            with T.attr(0, "tensorized_nki_instruction", 1):
                for lhs_f_loop in T.serial(0, lhs_f_size, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in T.serial(0, inst_repr.size, annotations={"nki_dim": "F"}):
                        b_idx = T.meta_var(lhs_b_loop * rhs_b_extent + rhs_b_loop)
                        inst_gen.set_bind_map(C_buffer_region, {lhs_f: lhs_f_loop, rhs_f: rhs_f_loop, lhs_b: lhs_b_loop, rhs_b: rhs_b_loop})  # noqa: E501
                        if inst_gen.make_guard(C_buffer_region):
                            acc_indices = T.meta_var(inst_gen.generate_indices(C_buffer_region))
                            T.evaluate(T.nki.tensor_copy(C[acc_indices], acc_psum[b_idx % max_psum_slots, lhs_f_loop, rhs_f_loop]))  # noqa: E501
    # fmt: on
    return impl_C_sbuf


# Rich dispatcher variant for TRN gemm
@register_dispatch(
    "gemm",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.scope_kind == "thread",
                f"unsupported exec_scope {sctx.scope_kind}",
            ),
        )
    ],
)
def gemm_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return matmul_trn(op, sctx)
