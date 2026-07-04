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

"""Implementation of binary operator dispatches."""

from tvm.script import tirx as T
from tvm.tirx import FloatImm, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.operator.tile_primitive.common import MapOpType
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import init_analyzer, nki_dim
from ..instruction_generator import InstructionGenerator
from .utils import InstType, binary_map_ops, try_find_inst_nary


def binary_trn(
    op: TilePrimitiveCall, binary_op: MapOpType, sctx: DispatchContext
) -> PrimFunc | None:
    """Generate a binary operation schedule for Trainium."""
    if not (sctx.is_target("trn") and sctx.scope_kind == "thread"):
        fail("requires Trainium target and thread exec_scope")

    assert binary_op in binary_map_ops, f"Unsupported binary operation {binary_op}"

    # Initialize analyzer and buffer regions
    analyzer = init_analyzer(sctx)
    _dst, _src1, _src2 = op.args

    # Find instruction parameters
    inst_gen = InstructionGenerator([_dst, _src1, _src2], analyzer)
    inst_repr, inst_types, reverse = try_find_inst_nary(_dst, [_src1, _src2], analyzer, inst_gen)
    # Handle operand swapping if needed
    if reverse[0]:
        _src1, _src2 = _src2, _src1

    # Extract buffers and constants
    CONST = _src2 if isinstance(_src2, FloatImm) else None
    dst, src1 = _dst.buffer, _src1.buffer
    src2 = None if CONST is not None else _src2.buffer

    p_var = T.Var("P", "int32")
    b_var = T.Var("B", "int32")
    f_var = T.Var("F", "int32")
    p_size = dst.layout.size("P")
    inst_size_limit = op.config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)
    inst_gen.bind_inst_iter(_dst, p_var, p_size, 1, False)
    inst_gen.bind_inst_iter(_dst, f_var, inst_repr.size, inst_repr.stride, True)
    b_extent = inst_gen.fill_in_block_dim(_dst, b_var)
    # Setup execution parameters
    opcode = binary_map_ops[binary_op]

    # Select appropriate NKI function based on instruction type
    _func = T.nki.tensortensor if inst_types[0] == InstType.TENSOR_TENSOR else T.nki.tensorscalar

    def func(*args):
        return _func(*args, reverse[0]) if inst_types[0] == InstType.TENSOR_SCALAR else _func(*args)

    # Define the implementation function
    @T.prim_func
    def impl():
        for b_loop in T.serial(0, b_extent):
            with T.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})

                        if inst_gen.make_guard(_dst):
                            dst_indices = T.meta_var(inst_gen.generate_indices(_dst))
                            src1_indices = T.meta_var(inst_gen.generate_indices(_src1))
                            if CONST is None:
                                src2_indices = T.meta_var(inst_gen.generate_indices(_src2))
                                T.evaluate(
                                    func(
                                        dst[tuple(dst_indices)],
                                        src1[tuple(src1_indices)],
                                        src2[tuple(src2_indices)],
                                        opcode,
                                    )
                                )
                            else:
                                T.evaluate(
                                    func(
                                        dst[tuple(dst_indices)],
                                        src1[tuple(src1_indices)],
                                        CONST,
                                        opcode,
                                    )
                                )

    return impl


# ---------------------------------------------------------------------------
# Registration: bind each binary op name to its TRN schedule candidates.
# ---------------------------------------------------------------------------
from tvm.tirx.operator.tile_primitive import register_dispatch  # noqa: E402

for _op_name, _op_type in {
    "add": MapOpType.ADD,
    "sub": MapOpType.SUB,
    "mul": MapOpType.MUL,
    "maximum": MapOpType.MAX,
    "minimum": MapOpType.MIN,
}.items():

    @register_dispatch(_op_name, "trn", variant="binary", priority=0)
    def _binary_dispatch(op, sctx, _ty=_op_type):
        return binary_trn(op, _ty, sctx)
