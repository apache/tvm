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

"""Implementation of BinaryChain dispatch."""

from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.ops import BinaryChain

from ..binary.utils import InstType, try_find_inst_nary
from ..common import init_analyzer, nki_dim
from ..instruction_generator import InstructionGenerator
from .utils import opcode_table


def binary_chain_trn(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Generate a TRN schedule for binary chain operations."""
    op = TilePrimitiveCall.downcast(op)
    assert isinstance(op, BinaryChain), f"invalid operator downcast: {op}"

    # Extract operation components
    output = op.dsts[0]
    srcs = op.srcs
    reverse = [False, op.reverse1]
    analyzer = init_analyzer(sctx)

    # Find instruction patterns
    inst_gen = InstructionGenerator([output, *srcs], analyzer)
    inst_result = try_find_inst_nary(
        output, srcs, analyzer, inst_gen, allow_first_op_tensortensor=False
    )
    inst_repr, inst_types, _reverse = inst_result

    # Generate axes and validate
    assert inst_types[0] == InstType.TENSOR_SCALAR, (
        "The first operator must be a tensor scalar operator"
    )

    # Handle input reversal if needed
    reverse[0] = _reverse[0]
    if reverse[0]:
        srcs[0], srcs[1] = srcs[1], srcs[0]

    p_var = Tx.Var("P", "int32")
    b_var = Tx.Var("B", "int32")
    f_var = Tx.Var("F", "int32")
    p_size = output.buffer.layout.size("P")
    inst_size_limit = op.config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)
    inst_gen.bind_inst_iter(output, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(output, f_var, inst_repr.size, inst_repr.stride, True)
    b_extent = inst_gen.fill_in_block_dim(output, b_var)

    # Extract buffers and opcodes
    _src, dst = srcs[0].buffer, output.buffer
    opcode0, opcode1 = opcode_table[op.op0], opcode_table[op.op1]

    # Determine operation function based on instruction type
    func = (
        Tx.nki.scalar_tensor_scalar
        if inst_types[1] == InstType.TENSOR_SCALAR
        else Tx.nki.scalar_tensor_tensor
    )

    # Helper function to get source indices
    def get_srcs(inst_gen):
        return [
            (
                srcs[i].buffer[inst_gen.generate_indices(srcs[i])]
                if isinstance(srcs[i], BufferRegion)
                else srcs[i]
            )
            for i in range(len(srcs))
        ]

    # Create implementation
    # fmt: off
    @Tx.prim_func
    def impl():
        for b_loop in Tx.serial(0, b_extent):
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in Tx.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})
                        dst_indices = Tx.meta_var(inst_gen.generate_indices(output))
                        srcs = Tx.meta_var(get_srcs(inst_gen))
                        if inst_gen.make_guard(output):
                            Tx.evaluate(func(dst[tuple(dst_indices)], *srcs, opcode0, opcode1, reverse[0], reverse[1]))  # noqa: E501
    # fmt: on

    return impl


@register_dispatch(
    "binary_chain",
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
def binary_chain_trn_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return binary_chain_trn(op, sctx)
