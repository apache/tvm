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

"""Implementation of unary with bias and scale operator dispatches."""

from tvm.tirx import BufferRegion, PrimFunc
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.stmt import TilePrimitiveCall

from ...common import MapOpType
from ..binary import try_find_inst_nary
from ..common import init_analyzer
from ..instruction_generator import InstructionGenerator
from .utils import activation_map_ops, generate_unary_func, try_find_inst_unary


def unary_with_bias_scale_trn(
    op: TilePrimitiveCall, unary_op: MapOpType = MapOpType.SQRT, sctx: DispatchContext = None
) -> PrimFunc | None:
    """Schedule unary operation with bias and scale on Trainium."""
    # Check execution environment
    if not (sctx.is_trn() and sctx.scope_kind == "kernel"):
        fail("requires Trainium target and kernel exec_scope")

    # Extract operation arguments with defaults
    dst_buffer_region, src_buffer_region, _bias, scale = op.args
    scale = 1.0 if scale is None else scale
    _bias = 0.0 if _bias is None else _bias

    # Initialize analyzer and validate operation type
    analyzer = init_analyzer(sctx)
    assert unary_op in activation_map_ops, f"Unsupported activation operation {unary_op}"

    # Find instruction parameters
    inst_gen = InstructionGenerator([dst_buffer_region, src_buffer_region, _bias], analyzer)
    if isinstance(_bias, BufferRegion):
        inst_repr, _, _ = try_find_inst_nary(
            dst_buffer_region,
            [src_buffer_region, _bias],
            analyzer,
            inst_gen,
            allow_first_op_tensortensor=False,
        )
    else:
        # Handle scalar bias
        inst_repr = try_find_inst_unary(dst_buffer_region, src_buffer_region, analyzer, inst_gen)

    # Generate and return the implementation function
    return generate_unary_func(
        dst_buffer_region,
        src_buffer_region,
        inst_gen,
        inst_repr,
        unary_op,
        _bias,
        scale,
        analyzer,
        op.workspace,
        op.config,
        sctx,
    )


# ---------------------------------------------------------------------------
# Registration: bind each unary_with_bias_scale op name to its TRN schedule candidates.
# ---------------------------------------------------------------------------
from tvm.tirx.operator.tile_primitive import register_dispatch  # noqa: E402

for _op_name, _op_type in {"sqrt": MapOpType.SQRT, "exp": MapOpType.EXP}.items():

    @register_dispatch(_op_name, "trn", variant="unary_with_bias_scale", priority=0)
    def _unary_bs_dispatch(op, sctx, _ty=_op_type):
        return unary_with_bias_scale_trn(op, _ty, sctx)
