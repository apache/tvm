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

"""Implementation of UnaryReduce dispatch."""

from tvm.script import tirx as T
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.ops import UnaryReduce

from ..binary.utils import try_find_inst_nary
from ..common import init_analyzer, nki_dim
from ..dim_utils import get_reduction_dim_map
from ..instruction_generator import InstructionGenerator
from ..reduction.utils import generate_intermediate_buffer
from ..unary.utils import get_const_bias_tensor, try_find_inst_unary
from .utils import opcode_table


def unary_reduce_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Generate a TRN schedule for unary reduction operations."""
    op = TilePrimitiveCall.downcast(op)
    assert isinstance(op, UnaryReduce), f"invalid operator downcast: {op}"

    # Extract operation components
    unary_output, reduce_output = op.dsts
    unary_input, bias, scale = op.srcs
    analyzer = init_analyzer(sctx)

    # Normalize axes and default values
    reduce_axes = [i if i >= 0 else len(unary_output.buffer.shape) + i for i in op.reduce_axes]
    scale = 1.0 if scale is None else scale
    bias = 0.0 if bias is None else bias

    inst_gen = InstructionGenerator([unary_output, unary_input, bias, reduce_output], analyzer)
    reduce_dim_map = get_reduction_dim_map(unary_output, reduce_output, reduce_axes, analyzer)
    inst_gen.link_buffer_regions(unary_output, reduce_output, reduce_dim_map)
    # Find instruction patterns based on bias type
    if isinstance(bias, BufferRegion):
        inst_repr, _, _ = try_find_inst_nary(
            unary_output,
            [unary_input, bias],
            analyzer,
            inst_gen,
            allow_first_op_tensortensor=False,
            allowed_f_dim_dst=reduce_axes,
        )
    else:
        inst_repr = try_find_inst_unary(
            unary_output, unary_input, analyzer, inst_gen, allowed_f_dim_dst=reduce_axes
        )

    # Apply instruction size limits
    inst_size_limit = op.config.get("max_inst_size", None)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    p_var = T.Var("P", "int32")
    f_var = T.Var("F", "int32")
    reduction_b_var = T.Var("rB", "int32")
    spatial_b_var = T.Var("sB", "int32")
    p_size = unary_output.buffer.layout.size("P")
    inst_gen.bind_inst_iter(unary_output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(unary_output, f_var, inst_repr.size, inst_repr.stride, True)
    reduction_b_extent = inst_gen.fill_in_block_dim(unary_output, reduction_b_var, reduce_axes)
    spatial_b_extent = inst_gen.fill_in_block_dim(unary_output, spatial_b_var)
    if reduction_b_extent != 1:
        intermediate_buffer = generate_intermediate_buffer(
            reduce_output, reduction_b_extent, op.workspace, sctx
        )
    # Extract buffers and opcodes
    src, dst1, dst2 = unary_input.buffer, unary_output.buffer, reduce_output.buffer
    unary_opcode = opcode_table[op.unary_op]
    reduce_opcode = opcode_table[op.reduce_op]

    # Handle bias buffer
    bias_buffer = (
        bias.buffer
        if isinstance(bias, BufferRegion)
        else get_const_bias_tensor(bias, (p_size, inst_repr.size), dst1.dtype, op.workspace, sctx)
    )

    # Create appropriate implementation based on intermediate buffer requirement
    if reduction_b_extent == 1:
        # Direct implementation without intermediate buffer
        # fmt: off
        @T.prim_func
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop})  # noqa: E501
                            src_1_indices = T.meta_var(inst_gen.generate_indices(unary_input))
                            dst_1_indices = T.meta_var(inst_gen.generate_indices(unary_output))
                            dst_2_indices = T.meta_var(inst_gen.generate_indices(reduce_output))
                            if inst_gen.make_guard(unary_output):
                                if isinstance(bias, BufferRegion):
                                    src_bias_indices = T.meta_var(inst_gen.generate_indices(bias))
                                    T.evaluate(T.nki.activation_reduce(dst2[tuple(dst_2_indices)], dst1[tuple(dst_1_indices)], src[tuple(src_1_indices)], unary_opcode, reduce_opcode, bias_buffer[tuple(src_bias_indices)], scale))  # noqa: E501
                                else:
                                    T.evaluate(T.nki.activation_reduce(dst2[tuple(dst_2_indices)], dst1[tuple(dst_1_indices)], src[tuple(src_1_indices)], unary_opcode, reduce_opcode, bias_buffer[p_loop, f_loop], scale))  # noqa: E501
        # fmt: on

        import tvm

        mod = tvm.IRModule({"main": impl})
        mod = tvm.tirx.transform.StmtSimplify()(mod)
        return mod["main"]
    else:
        # fmt: off
        @T.prim_func
        def impl():
            for b_loop in T.serial(0, spatial_b_extent):
                for reduction_b_loop in T.serial(0, reduction_b_extent):
                    with T.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                                inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop, reduction_b_var: reduction_b_loop})  # noqa: E501
                                src_1_indices = T.meta_var(inst_gen.generate_indices(unary_input))
                                dst_1_indices = T.meta_var(inst_gen.generate_indices(unary_output))
                                if inst_gen.make_guard(unary_output):
                                    if isinstance(bias, BufferRegion):
                                        src_bias_indices = T.meta_var(inst_gen.generate_indices(bias))  # noqa: E501
                                        T.evaluate(T.nki.activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[tuple(dst_1_indices)], src[tuple(src_1_indices)], unary_opcode, reduce_opcode, bias_buffer[tuple(src_bias_indices)], scale))  # noqa: E501
                                    else:
                                        T.evaluate(T.nki.activation_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[tuple(dst_1_indices)], src[tuple(src_1_indices)], unary_opcode, reduce_opcode, bias_buffer[p_loop, f_loop], scale))  # noqa: E501
                with T.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in T.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(reduce_output):
                                dst_2_indices = T.meta_var(inst_gen.generate_indices(reduce_output))
                                # TODO: we should use nki.activation_reduce as second stage reduction  # noqa: E501
                                T.evaluate(T.nki.tensorreduce(dst2[tuple(dst_2_indices)], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1))  # noqa: E501
        # fmt: on

        return impl


@register_dispatch(
    "unary_reduce",
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
def unary_reduce_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return unary_reduce_trn(op, sctx)
