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

from tvm.backend.trn.layout import is_trainium_layout
from tvm.ir import TensorRegion, assert_structural_equal
from tvm.script import tirx as T
from tvm.sym.analyzer import Analyzer
from tvm.tirx import Function
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.tensor_instruction import TensorCall

from ..common import init_analyzer
from ..dim_utils import normalize_and_group
from ..instruction_generator import InstructionGenerator
from ..workspace_utils import check_workspace_tensor, largest_psum_per_bank, max_psum_banks


class OperatorKind:
    A = 0
    B = 1
    C = 2


def get_pf_dim_from_tensor_region(
    tensor_region: TensorRegion,
    analyzer: Analyzer,
    operator_kind: OperatorKind,
    transposed: bool = False,
):
    """Extract partition and free dimensions from tensor region."""
    # Find non-unit dimensions
    non_unit_dims = [
        i
        for i in range(len(tensor_region.source.ty.shape))
        if not analyzer.can_prove_equal(tensor_region.region[i].extent, 1)
    ]
    assert len(non_unit_dims) == 2, "Only 2D matrix is supported for gemm"

    layout, seps = normalize_and_group(
        tensor_region.source.ty.layout, tensor_region.source.ty.shape
    )
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
        f"However, the {p_dim} dimension of {tensor_region} does not."
    )

    # Validate free dimension
    assert all(
        layout.shard[i].axis.name in ["F", "Bank"] or layout.shard[i].extent == 1
        for i in range(seps[f_dim], seps[f_dim + 1])
    ), (
        f"Spatial dimension must not contain P. However, the {f_dim} dimension of {tensor_region} does."  # noqa: E501
    )

    return p_dim, f_dim


def matmul_trn(op: TensorCall, sctx: DispatchContext) -> Function | None:
    """Schedule GEMM operation on Trainium."""
    # Basic validation checks
    if not (sctx.is_target("trn") and sctx.scope_kind == "thread"):
        fail("requires Trainium target and thread exec_scope")

    # Extract arguments
    (
        D_tensor_region,
        A_tensor_region,
        B_tensor_region,
        C_tensor_region,
        transpose_A,
        transpose_B,
        alpha,
        beta,
    ) = op.args
    analyzer = init_analyzer(sctx)
    A, B, C, _D = (
        A_tensor_region.source,
        B_tensor_region.source,
        C_tensor_region.source,
        D_tensor_region.source,
    )

    # Validate alpha, beta
    assert analyzer.can_prove_equal(alpha, 1) and analyzer.can_prove_equal(beta, 0), (
        "Only alpha=1 and beta=0 are supported"
    )

    # D and C must be the same tensor region
    assert_structural_equal(D_tensor_region, C_tensor_region)

    # Validate tensor properties
    assert all(
        [
            A.ty.layout and B.ty.layout and C.ty.layout,
            A.ty.dtype == B.ty.dtype,
            A.scope() == "trn.sbuf" and B.scope() == "trn.sbuf",
            C.scope() == "trn.psum" or C.scope() == "trn.sbuf",
            is_trainium_layout(A.ty.layout),
            is_trainium_layout(B.ty.layout),
            is_trainium_layout(C.ty.layout),
            A.ty.layout.size("P") == B.ty.layout.size("P"),
        ]
    ), "Invalid tensor layout and scope"

    p_size = A.ty.layout.size("P")
    assert p_size == B.ty.layout.size("P"), "Partition size mismatch"

    # Get partition and free dimensions
    lhs_p_dim, lhs_f_dim = get_pf_dim_from_tensor_region(
        A_tensor_region, analyzer, OperatorKind.A, transpose_A
    )
    rhs_p_dim, rhs_f_dim = get_pf_dim_from_tensor_region(
        B_tensor_region, analyzer, OperatorKind.B, transpose_B
    )
    acc_p_dim, acc_f_dim = get_pf_dim_from_tensor_region(C_tensor_region, analyzer, OperatorKind.C)
    # Swap LHS and RHS if needed based on accumulator dimensions
    swap_lhs_rhs = acc_p_dim > acc_f_dim
    if swap_lhs_rhs:
        lhs_p_dim, rhs_p_dim = rhs_p_dim, lhs_p_dim
        lhs_f_dim, rhs_f_dim = rhs_f_dim, lhs_f_dim
        A, B = B, A
        A_tensor_region, B_tensor_region = B_tensor_region, A_tensor_region

    # Validate dimension compatibility
    assert analyzer.can_prove(
        A_tensor_region.region[lhs_p_dim].extent == B_tensor_region.region[rhs_p_dim].extent
    ), (
        f"Reduction dimension must match, but the {lhs_p_dim} dimension of {A_tensor_region} != the {rhs_p_dim} dimension of {B_tensor_region}"  # noqa: E501
    )

    assert analyzer.can_prove(
        A_tensor_region.region[lhs_f_dim].extent == C_tensor_region.region[acc_p_dim].extent
    ), (
        f"Spatial dimension must match, but the {lhs_f_dim} dimension of {A_tensor_region} != the {acc_p_dim} dimension of {C_tensor_region}"  # noqa: E501
    )

    assert analyzer.can_prove(
        B_tensor_region.region[rhs_f_dim].extent == C_tensor_region.region[acc_f_dim].extent
    ), (
        f"Spatial dimension must match, but the {rhs_f_dim} dimension of {B_tensor_region} != the {acc_f_dim} dimension of {C_tensor_region}"  # noqa: E501
    )

    inst_gen = InstructionGenerator([A_tensor_region, B_tensor_region, C_tensor_region], analyzer)
    inst_gen.link_tensor_regions(A_tensor_region, B_tensor_region, {lhs_p_dim: rhs_p_dim})
    inst_gen.link_tensor_regions(B_tensor_region, C_tensor_region, {rhs_f_dim: acc_f_dim})
    inst_gen.link_tensor_regions(A_tensor_region, C_tensor_region, {lhs_f_dim: acc_p_dim})
    inst_repr = inst_gen.find_max_inst_size_from_one_region(B_tensor_region, [rhs_f_dim])
    inst_repr = inst_gen.fit_inst_tile_to_region(inst_repr, C_tensor_region, [acc_f_dim])
    inst_repr.bound_inst_size(512, analyzer)
    rhs_f = T.Var("rhs_f", "int32")
    lhs_f = T.Var("lhs_f", "int32")
    p = T.Var("p", "int32")
    reduction_b = T.Var("reduction_b", "int32")
    lhs_b = T.Var("lhs_b", "int32")
    rhs_b = T.Var("rhs_b", "int32")
    lhs_f_size = C.ty.layout.size("P")
    inst_gen.bind_inst_iter(
        B_tensor_region, rhs_f, inst_repr.size, inst_repr.stride, is_free_dim=True
    )
    inst_gen.bind_inst_iter(C_tensor_region, lhs_f, lhs_f_size, 1, is_free_dim=False)
    inst_gen.bind_inst_iter(A_tensor_region, p, A.ty.layout.size("P"), 1, is_free_dim=False)
    reduction_b_extent = inst_gen.fill_in_block_dim(A_tensor_region, reduction_b, [lhs_p_dim])
    lhs_b_extent = inst_gen.fill_in_block_dim(A_tensor_region, lhs_b, [lhs_f_dim])
    rhs_b_extent = inst_gen.fill_in_block_dim(B_tensor_region, rhs_b, [rhs_f_dim])

    # FIXME: we need to lower the guard to things like matmul(lhs[...][lhs_guard], rhs[...][rhs_guard], mask=p_guard)  # noqa: E501
    # so we need to separate the guard for lhs_f, rhs_f and p
    # fmt: off
    @T.inline
    def matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, acc, C_as_output, max_psum_slots):  # noqa: E501
        with T.nki.tensorized_instruction():
            for p_loop in T.serial(0, p_size, annotations={"nki_dim": "P"}):
                for lhs_f_loop in T.serial(0, lhs_f_size, annotations={"nki_dim": "lhs_F"}):
                    for rhs_f_loop in T.serial(0, inst_repr.size, annotations={"nki_dim": "rhs_F"}):
                        inst_gen.set_bind_map(A_tensor_region, {lhs_b: lhs_b_loop, lhs_f: lhs_f_loop, p: p_loop, reduction_b: reduction_b_loop})  # noqa: E501
                        inst_gen.set_bind_map(B_tensor_region, {rhs_b: rhs_b_loop, rhs_f: rhs_f_loop, p: p_loop, reduction_b: reduction_b_loop})  # noqa: E501
                        inst_gen.set_bind_map(C_tensor_region, {lhs_f: lhs_f_loop, rhs_f: rhs_f_loop, lhs_b: lhs_b_loop, rhs_b: rhs_b_loop})  # noqa: E501
                        lhs_indices = T.meta_var(inst_gen.generate_indices(A_tensor_region))
                        rhs_indices = T.meta_var(inst_gen.generate_indices(B_tensor_region))
                        C_indices = T.meta_var(inst_gen.generate_indices(C_tensor_region))
                        if inst_gen.make_guard(A_tensor_region) and inst_gen.make_guard(B_tensor_region):  # noqa: E501
                            if T.constexpr(C_as_output):
                                T.evaluate(T.nki.matmul(acc[C_indices], A[lhs_indices], B[rhs_indices]))  # noqa: E501
                            else:
                                T.evaluate(T.nki.matmul(acc[(lhs_b_loop * rhs_b_extent + rhs_b_loop) % max_psum_slots, lhs_f_loop, rhs_f_loop], A[lhs_indices], B[rhs_indices]))  # noqa: E501

    if C.scope() == "trn.psum":
        # This fragment captures tensors and indices from its insertion scope.
        @T.function(check_well_formed=False)
        def impl_C_psum():
            for lhs_b_loop, rhs_b_loop, reduction_b_loop in T.grid(lhs_b_extent, rhs_b_extent, reduction_b_extent):  # noqa: E501
                matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, C, True, None)
        return impl_C_psum

    # todo: generalize the process of generating composite matmul + another_op pattern
    # by generating TIR op and reusing existing dispatch rule

    # we will support matmul + epilogue as a user-specified pattern
    # and a matmul fusion pass can help infer the pattern

    acc_psum_shape = (max_psum_banks, p_size, largest_psum_per_bank)
    if "acc_psum" not in op.workspaces:
        assert sctx.alloc_only, "Accumulation psum tensor must be specified in workspace. Run tvm.tirx.trn.transform.TrnPrivateTensorAlloc first."  # noqa: E501
        acc_psum = T.Var(
            "acc_psum",
            T.Tensor(acc_psum_shape, "float32", scope="trn.psum"),
        )
        sctx.add_alloc_tensor(
            acc_psum, allocated_addr=[T.int32(0), T.int32(0)]
        )
        max_psum_slots = max_psum_banks
    else:
        acc_psum = op.workspaces["acc_psum"]
        check_workspace_tensor(acc_psum, (p_size, largest_psum_per_bank), "trn.psum")
        max_psum_slots = acc_psum.ty.shape[0]

    # This fragment captures tensors and indices from its insertion scope.
    @T.function(check_well_formed=False)
    def impl_C_sbuf():
        for lhs_b_loop, rhs_b_loop in T.grid(lhs_b_extent, rhs_b_extent):
            for reduction_b_loop in T.serial(0, reduction_b_extent):
                matmul_inst_macro(lhs_b_loop, rhs_b_loop, reduction_b_loop, acc_psum, False, max_psum_slots)  # noqa: E501
            with T.nki.tensorized_instruction():
                for lhs_f_loop in T.serial(0, lhs_f_size, annotations={"nki_dim": "P"}):
                    for rhs_f_loop in T.serial(0, inst_repr.size, annotations={"nki_dim": "F"}):
                        inst_gen.set_bind_map(C_tensor_region, {lhs_f: lhs_f_loop, rhs_f: rhs_f_loop, lhs_b: lhs_b_loop, rhs_b: rhs_b_loop})  # noqa: E501
                        if inst_gen.make_guard(C_tensor_region):
                            acc_indices = T.meta_var(inst_gen.generate_indices(C_tensor_region))
                            T.evaluate(T.nki.tensor_copy(C[acc_indices], acc_psum[(lhs_b_loop * rhs_b_extent + rhs_b_loop) % max_psum_slots, lhs_f_loop, rhs_f_loop]))  # noqa: E501
    # fmt: on
    return impl_C_sbuf
