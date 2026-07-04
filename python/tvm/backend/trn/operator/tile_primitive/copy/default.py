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

from tvm.backend.trn.layout import is_trainium_layout
from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import init_analyzer, nki_dim
from ..dim_utils import get_ewise_dim_map
from ..instruction_generator import InstructionGenerator
from ..workspace_utils import check_workspace_buffer, largest_psum_per_bank, max_psum_banks


def transpose_schedule(
    op: TilePrimitiveCall, inst_gen: InstructionGenerator, sctx: DispatchContext
) -> PrimFunc | None:
    dst_region, src_region = op.args
    assert src_region.buffer.scope() != "trn.psum", "Transpose on psum buffer is not supported"

    inst_repr_dst, inst_repr_src = inst_gen.find_max_inst_size_transpose(dst_region, src_region)

    lhs_f = T.Var("lhs_F", "int32")
    lhs_p = T.Var("lhs_P", "int32")
    dst_f = T.Var("dst_F", "int32")
    b_var = T.Var("B", "int32")
    extend_b = T.Var("extend_B", "int32")
    p_size = src_region.buffer.layout.size("P")
    lhs_f_size = dst_region.buffer.layout.size("P")
    rhs_f_size = p_size
    inst_gen.bind_inst_iter(
        src_region, lhs_f, inst_repr_src.size, inst_repr_src.stride, is_free_dim=True
    )
    inst_gen.bind_inst_iter(
        dst_region,
        dst_f,
        inst_repr_dst.size,
        inst_repr_dst.stride,
        is_free_dim=True,
        no_propagate=True,
    )
    inst_gen.bind_inst_iter(src_region, lhs_p, p_size, 1, is_free_dim=False, no_propagate=True)
    if dst_region.buffer.scope() == "trn.sbuf":
        max_extend_num = (
            inst_gen.find_max_inst_size_from_one_region(
                dst_region, min_stride=inst_repr_dst.stride
            ).size
            // rhs_f_size
        )
        max_elem_in_a_bank = largest_psum_per_bank // rhs_f_size
        if max_extend_num < max_elem_in_a_bank:
            extend_len = max_extend_num
        elif max_extend_num % max_elem_in_a_bank == 0:
            extend_len = max_elem_in_a_bank
        else:
            extend_len = 1
        inst_gen.bind_inst_iter(
            dst_region,
            extend_b,
            extend_len,
            inst_repr_dst.stride * inst_repr_dst.size,
            is_free_dim=True,
        )
    b_extent = inst_gen.fill_in_block_dim(dst_region, b_var)

    if "identity" not in op.workspace:
        assert sctx.alloc_only, (
            "Identity tensor must be specified in workspace. Run tvm.tirx.trn.transform.TrnPrivateBufferAlloc first."  # noqa: E501
        )
        identity_tensor = T.buffer(
            (p_size, rhs_f_size), src_region.buffer.dtype, scope="trn.sbuf", buffer_name="identity"
        )
        sctx.add_alloc_buffer(identity_tensor)

        @T.prim_func
        def identity_init():
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for rhs_f_loop in T.serial(0, rhs_f_size, annotations={nki_dim: "F"}):
                        T.evaluate(T.nki.identity(identity_tensor[p_loop, rhs_f_loop], p_size))
            T.tvm_kernel_replace_point()

        sctx.add_init_stmt(identity_init.body)
    else:
        identity_tensor = op.workspace["identity"]
        check_workspace_buffer(identity_tensor, (p_size, rhs_f_size), "trn.sbuf")

    dst_buffer = dst_region.buffer
    src_buffer = src_region.buffer
    if dst_buffer.scope() == "trn.psum":

        @T.prim_func
        def transpose_psum_output():
            for b_loop in T.serial(0, b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for lhs_f_loop in T.serial(0, lhs_f_size, annotations={nki_dim: "lhs_F"}):
                            for rhs_f_loop in T.serial(
                                0, rhs_f_size, annotations={nki_dim: "rhs_F"}
                            ):
                                inst_gen.set_bind_map(
                                    dst_region,
                                    {b_var: b_loop, lhs_f: lhs_f_loop, dst_f: rhs_f_loop},
                                )
                                inst_gen.set_bind_map(
                                    src_region, {b_var: b_loop, lhs_f: lhs_f_loop, lhs_p: p_loop}
                                )
                                src_indices = T.meta_var(inst_gen.generate_indices(src_region))
                                dst_indices = T.meta_var(inst_gen.generate_indices(dst_region))
                                src_guard = T.meta_var(inst_gen.make_guard(src_region))
                                dst_guard = T.meta_var(inst_gen.make_guard(dst_region))
                                if src_guard and dst_guard:
                                    T.evaluate(
                                        T.nki.matmul(
                                            dst_buffer[tuple(dst_indices)],
                                            src_buffer[tuple(src_indices)],
                                            identity_tensor[p_loop, rhs_f_loop],
                                        )
                                    )

        return transpose_psum_output

    if "acc_psum" not in op.workspace:
        assert sctx.alloc_only, (
            "Accumulation psum buffer must be specified in workspace. Run tvm.tirx.trn.transform.TrnPrivateBufferAlloc first."  # noqa: E501
        )
        acc_psum = T.buffer(
            (max_psum_banks, p_size, largest_psum_per_bank),
            "float32",
            scope="trn.psum",
            allocated_addr=(0, 0),
            buffer_name="acc_psum",
        )
        sctx.add_alloc_buffer(acc_psum)
        max_psum_slots = max_psum_banks
    else:
        acc_psum = op.workspace["acc_psum"]
        check_workspace_buffer(acc_psum, (p_size, largest_psum_per_bank), "trn.psum")
        max_psum_slots = acc_psum.shape[0]

    # fmt: off
    @T.prim_func
    def transpose_sbuf_output():
        for b_loop in T.serial(0, b_extent):
            for extend_b_loop in T.serial(0, extend_len):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for lhs_f_loop in T.serial(0, lhs_f_size, annotations={nki_dim: "lhs_F"}):
                            for rhs_f_loop in T.serial(0, rhs_f_size, annotations={nki_dim: "rhs_F"}):  # noqa: E501
                                inst_gen.set_bind_map(src_region, {b_var: b_loop, lhs_f: lhs_f_loop, lhs_p: p_loop, extend_b: extend_b_loop})  # noqa: E501
                                src_indices = T.meta_var(inst_gen.generate_indices(src_region))
                                src_guard = T.meta_var(inst_gen.make_guard(src_region))
                                if src_guard:
                                    T.evaluate(T.nki.matmul(acc_psum[b_loop % max_psum_slots, lhs_f_loop,extend_b_loop * rhs_f_size + rhs_f_loop], src_buffer[tuple(src_indices)], identity_tensor[p_loop, rhs_f_loop]))  # noqa: E501
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, rhs_f_size * extend_len, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map(dst_region, {b_var: b_loop, lhs_f: p_loop, dst_f: f_loop % rhs_f_size, extend_b: f_loop // rhs_f_size})  # noqa: E501
                        dst_guard = T.meta_var(inst_gen.make_guard(dst_region))
                        dst_indices = T.meta_var(inst_gen.generate_indices(dst_region))
                        if dst_guard:
                            T.evaluate(T.nki.tensor_copy(dst_buffer[tuple(dst_indices)], acc_psum[b_loop % max_psum_slots, p_loop, f_loop]))  # noqa: E501
    # fmt: on
    return transpose_sbuf_output


def copy_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if sctx.scope_kind != "thread":
        fail("requires thread exec_scope for TRN copy")

    dst_region, src_region = op.args
    src, dst = src_region.buffer, dst_region.buffer

    # Check for valid buffer configurations
    valid_config = all(
        [
            src.layout and dst.layout,
            src.scope() in ["global", "trn.sbuf", "trn.psum"],
            dst.scope() in ["global", "trn.sbuf", "trn.psum"],
            src.scope() != "global" or dst.scope() != "global",
            (src.scope() == "global" and isinstance(src.layout, T.TileLayout))
            or (src.scope() in ["trn.sbuf", "trn.psum"] and is_trainium_layout(src.layout)),
            (dst.scope() == "global" and isinstance(dst.layout, T.TileLayout))
            or (dst.scope() in ["trn.sbuf", "trn.psum"] and is_trainium_layout(dst.layout)),
        ]
    )

    if not valid_config:
        raise ValueError("Invalid buffer layout/scope for copy operation.")

    analyzer = init_analyzer(sctx)
    src_extent = [r.extent for r in src_region.region]
    dst_extent = [r.extent for r in dst_region.region]

    # Validate non-unit dimensions match
    src_non_unit = [e for e in src_extent if e != 1]
    dst_non_unit = [e for e in dst_extent if e != 1]
    dims_match = len(src_non_unit) == len(dst_non_unit) and all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_non_unit, dst_non_unit)
    )

    if not dims_match:
        fail("shape mismatch between src and dst for TRN copy")

    dim_map = get_ewise_dim_map(src_region, dst_region, analyzer)
    inst_gen = InstructionGenerator([src_region, dst_region], analyzer)
    inst_gen.link_buffer_regions(src_region, dst_region, dim_map)

    if not inst_gen.check_partition_dim_match(src_region, dst_region):
        return transpose_schedule(op, inst_gen, sctx)

    if is_trainium_layout(src.layout):
        inst = inst_gen.find_max_inst_size_from_one_region(src_region)
        inst = inst_gen.fit_inst_tile_to_region(inst, dst_region)
        src_to_dst = True
    else:
        inst = inst_gen.find_max_inst_size_from_one_region(dst_region)
        inst = inst_gen.fit_inst_tile_to_region(inst, src_region)
        src_to_dst = False

    if src.scope() == "global":
        func = T.nki.load
    elif dst.scope() == "global":
        func = T.nki.store
    else:
        func = T.nki.tensor_copy

    if func == T.nki.tensor_copy:
        inst_size_limit = op.config.get("max_inst_size", 512)
        inst.bound_inst_size(inst_size_limit, analyzer)
    else:
        assert "max_inst_size" not in op.config, "max_inst_size is not supported for load/store"

    p_var = T.Var("P", "int32")
    f_var = T.Var("F", "int32")
    b_var = T.Var("B", "int32")
    if src_to_dst:
        from_region, _to_region = src_region, dst_region
    else:
        from_region, _to_region = dst_region, src_region
    p_size = from_region.buffer.layout.size("P")
    inst_gen.bind_inst_iter(from_region, p_var, p_size, 1, is_free_dim=False)
    inst_gen.bind_inst_iter(from_region, f_var, inst.size, inst.stride, is_free_dim=True)
    b_extent = inst_gen.fill_in_block_dim(from_region, b_var)

    # fmt: off
    @T.prim_func
    def impl():
        # the additional b loop is to satisfy hardware instuction size limit
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({b_var: b_loop, p_var: p_loop, f_var: f_loop})
                        if inst_gen.make_guard(dst_region):
                            src_indices = T.meta_var(inst_gen.generate_indices(src_region))
                            dst_indices = T.meta_var(inst_gen.generate_indices(dst_region))
                            func(dst[tuple(dst_indices)], src[tuple(src_indices)])
    # fmt: on
    return impl


# Rich dispatcher variant for TRN copy
@register_dispatch(
    "copy",
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
def copy_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_trn(op, sctx)
