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

"""Implementation of BinaryReduce dispatch."""

from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.ops import BinaryReduce

from ..binary.utils import InstType, try_find_inst_nary
from ..common import init_analyzer, nki_dim
from ..dim_utils import get_reduction_dim_map
from ..instruction_generator import InstructionGenerator
from ..reduction.utils import generate_intermediate_buffer
from .utils import opcode_table


def binary_reduce_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Generate a TRN schedule for binary reduction operations."""
    op = TilePrimitiveCall.downcast(op)
    assert isinstance(op, BinaryReduce), f"invalid operator downcast: {op}"

    # Extract operation components
    binary_output, reduce_output = op.dsts
    binary_input1, binary_input2 = op.srcs
    reduce_axes = op.reduce_axes
    analyzer = init_analyzer(sctx)

    # Normalize negative axes
    reduce_axes = [i if i >= 0 else len(binary_output.buffer.shape) + i for i in reduce_axes]

    # Find instruction patterns
    inst_gen = InstructionGenerator(
        [binary_output, binary_input1, binary_input2, reduce_output], analyzer
    )
    reduce_dim_map = get_reduction_dim_map(binary_output, reduce_output, reduce_axes, analyzer)
    inst_gen.link_buffer_regions(binary_output, reduce_output, reduce_dim_map)
    inst_repr, inst_type, reverse = try_find_inst_nary(
        binary_output,
        [binary_input1, binary_input2],
        analyzer,
        inst_gen,
        allowed_f_dim_dst=reduce_axes,
        allow_first_op_tensortensor=False,
    )

    # Apply instruction size limits
    inst_size_limit = op.config.get("max_inst_size", None)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    # Generate axes and validate
    assert inst_type[0] == InstType.TENSOR_SCALAR, (
        f"TensorTensor is not supported for vector reduce: {op}"
    )

    # Handle input reversal if needed
    if reverse[0]:
        binary_input1, binary_input2 = binary_input2, binary_input1

    # Generate intermediate buffer for reduction if needed
    p_var = Tx.Var("P", "int32")
    f_var = Tx.Var("F", "int32")
    reduction_b_var = Tx.Var("rB", "int32")
    spatial_b_var = Tx.Var("sB", "int32")
    p_size = binary_output.buffer.layout.size("P")
    inst_gen.bind_inst_iter(binary_output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(binary_output, f_var, inst_repr.size, inst_repr.stride, True)
    reduction_b_extent = inst_gen.fill_in_block_dim(binary_output, reduction_b_var, reduce_axes)
    spatial_b_extent = inst_gen.fill_in_block_dim(binary_output, spatial_b_var)
    if reduction_b_extent != 1:
        intermediate_buffer = generate_intermediate_buffer(
            reduce_output, reduction_b_extent, op.workspace, sctx
        )

    # Handle source 2 (either buffer region or constant)
    CONST = binary_input2 if not isinstance(binary_input2, BufferRegion) else None
    # Extract buffers and opcodes
    src1, src2 = (
        binary_input1.buffer,
        (binary_input2.buffer if isinstance(binary_input2, BufferRegion) else None),
    )
    dst1, dst2 = binary_output.buffer, reduce_output.buffer
    binary_opcode, reduce_opcode = opcode_table[op.binary_op], opcode_table[op.reduce_op]
    # Create appropriate implementation based on intermediate buffer requirement
    if reduction_b_extent == 1:
        # Direct implementation without intermediate buffer
        # fmt: off
        @Tx.prim_func
        def impl():
            for b_loop in Tx.serial(0, spatial_b_extent):
                with Tx.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in Tx.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in Tx.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop})  # noqa: E501
                            src_1_indices = Tx.meta_var(inst_gen.generate_indices(binary_input1))
                            vec_dst_idx = Tx.meta_var(inst_gen.generate_indices(binary_output))
                            reduce_dst_idx = Tx.meta_var(inst_gen.generate_indices(reduce_output))
                            if inst_gen.make_guard(binary_output):
                                if CONST is None:
                                    src_2_indices = Tx.meta_var(inst_gen.generate_indices(binary_input2))  # noqa: E501
                                    Tx.nki.tensorscalar_reduce(dst2[tuple(reduce_dst_idx)], dst1[tuple(vec_dst_idx)], src1[tuple(src_1_indices)], src2[tuple(src_2_indices)], binary_opcode, reduce_opcode, reverse[0])  # noqa: E501
                                else:
                                    Tx.nki.tensorscalar_reduce(dst2[tuple(reduce_dst_idx)], dst1[tuple(vec_dst_idx)], src1[tuple(src_1_indices)], CONST, binary_opcode, reduce_opcode, reverse[0])  # noqa: E501
        # fmt: on
    else:
        # Implementation with intermediate buffer
        # fmt: off
        @Tx.prim_func
        def impl():
            for b_loop in Tx.serial(0, spatial_b_extent):
                for reduction_b_loop in Tx.serial(0, reduction_b_extent):
                    with Tx.attr(0, "tensorized_nki_instruction", 1):
                        for p_loop in Tx.serial(0, p_size, annotations={nki_dim: "P"}):
                            for f_loop in Tx.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                                inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, spatial_b_var: b_loop, reduction_b_var: reduction_b_loop})  # noqa: E501
                                if inst_gen.make_guard(binary_output):
                                    src_1_indices = Tx.meta_var(inst_gen.generate_indices(binary_input1))  # noqa: E501
                                    vec_dst_idx = Tx.meta_var(inst_gen.generate_indices(binary_output))  # noqa: E501
                                    if CONST is None:
                                        src_2_indices = Tx.meta_var(inst_gen.generate_indices(binary_input2))  # noqa: E501
                                        Tx.nki.tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[tuple(vec_dst_idx)], src1[tuple(src_1_indices)], src2[tuple(src_2_indices)], binary_opcode, reduce_opcode, reverse[0])  # noqa: E501
                                    else:
                                        Tx.nki.tensorscalar_reduce(intermediate_buffer[p_loop, reduction_b_loop], dst1[tuple(vec_dst_idx)], src1[tuple(src_1_indices)], CONST, binary_opcode, reduce_opcode, reverse[0])  # noqa: E501
                with Tx.attr(0, "tensorized_nki_instruction", 1):
                    for p_loop in Tx.serial(0, p_size, annotations={nki_dim: "P"}):
                        for f_loop in Tx.serial(0, reduction_b_extent, annotations={nki_dim: "F"}):
                            inst_gen.set_bind_map_all({p_var: p_loop, spatial_b_var: b_loop})
                            if inst_gen.make_guard(reduce_output):
                                dst_2_indices = Tx.meta_var(inst_gen.generate_indices(reduce_output))  # noqa: E501
                                Tx.nki.tensorreduce(dst2[tuple(dst_2_indices)], intermediate_buffer[p_loop, f_loop], reduce_opcode, False, -1)  # noqa: E501
        # fmt: on

    return impl


# Rich dispatcher variants for TRN compose ops
@register_dispatch(
    "binary_reduce",
    "trn",
    variant="default",
    priority=10,
    when=[
        predicate(
            "exec_scope",
            lambda op, sctx: (
                sctx.scope_kind == "kernel",
                f"unsupported exec_scope {sctx.scope_kind}",
            ),
        )
    ],
)
def binary_reduce_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return binary_reduce_trn(op, sctx)
