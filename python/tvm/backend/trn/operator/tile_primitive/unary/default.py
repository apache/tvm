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

"""Implementation of default unary operator dispatches."""

from tvm.tirx import FloatImm, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.operator.tile_primitive.common import MapOpType
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import init_analyzer
from ..instruction_generator import InstructionGenerator
from .utils import (
    const_input_ops,
    generate_unary_func,
    non_activation_unary_map_ops,
    try_find_inst_unary,
)


def unary_trn(op: TilePrimitiveCall, unary_op: MapOpType, sctx: DispatchContext) -> PrimFunc | None:
    """Schedule unary operation on Trainium."""
    # Check execution environment
    if not (sctx.is_target("trn") and sctx.scope_kind == "thread"):
        fail("requires Trainium target and thread exec_scope")

    # Extract operation arguments
    dst_buffer_region, _src = op.args

    # Handle constant or buffer source
    if isinstance(_src, FloatImm):
        if unary_op not in const_input_ops:
            assert False, f"Unsupported unary operation {unary_op} taking const as input"
        CONST = _src
        src_buffer_region = None
    else:
        CONST = None
        src_buffer_region = _src

    # Initialize analyzer and validate operation type
    analyzer = init_analyzer(sctx)
    assert unary_op in non_activation_unary_map_ops, f"Unsupported unary operation {unary_op}"

    inst_gen = InstructionGenerator([dst_buffer_region, _src], analyzer)
    # Find instruction parameters
    if CONST is None:
        inst_repr = try_find_inst_unary(dst_buffer_region, src_buffer_region, analyzer, inst_gen)
    else:
        inst_repr = try_find_inst_unary(dst_buffer_region, dst_buffer_region, analyzer, inst_gen)
    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        _src,
        inst_gen,
        inst_repr,
        unary_op,
        None,  # No bias
        None,  # No scale
        analyzer,
        op.workspace,
        op.config,
        sctx,
    )


# ---------------------------------------------------------------------------
# Registration: bind each default unary op name to its TRN schedule candidates.
# ---------------------------------------------------------------------------
from tvm.tirx.operator.tile_primitive import register_dispatch  # noqa: E402

for _op_name, _op_type in {"reciprocal": MapOpType.RECIPROCAL, "memset": MapOpType.FILL}.items():

    @register_dispatch(_op_name, "trn", variant="unary", priority=0)
    def _unary_dispatch(op, sctx, _ty=_op_type):
        return unary_trn(op, _ty, sctx)
