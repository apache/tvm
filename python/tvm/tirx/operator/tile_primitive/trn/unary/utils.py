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

"""Shared helpers, op tables, and validation functions for unary operator dispatches."""

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import BufferRegion, FloatImm

from ...common import MapOpType
from ..common import nki_dim
from ..dim_utils import get_ewise_dim_map
from ..instruction_generator import InstructionGenerator
from ..workspace_utils import check_workspace_buffer

# Operation type classifications
non_activation_unary_map_ops = [MapOpType.RECIPROCAL, MapOpType.FILL]
activation_map_ops = [MapOpType.SQRT, MapOpType.EXP]

# Operation code table for instructions
opcode_table = {MapOpType.SQRT: "sqrt", MapOpType.EXP: "exp"}

# Operations that take constants as input
const_input_ops = [MapOpType.FILL]


def try_find_inst_unary(
    dst_buffer_region: BufferRegion,
    src_buffer_region: BufferRegion,
    analyzer: Analyzer,
    inst_gen: InstructionGenerator,
    allowed_f_dim_dst: tuple[int] | None = None,
    allowed_f_dim_src: tuple[int] | None = None,
):
    """Find instruction parameters for a unary operation."""
    dst = dst_buffer_region.buffer
    src = src_buffer_region.buffer

    # Validate buffer layouts and scopes
    valid_layout_scope = all(
        [
            src.layout and dst.layout,
            src.scope() in ("trn.sbuf", "trn.psum"),
            dst.scope() == "trn.sbuf",
            src.layout.is_trainium(),
            dst.layout.is_trainium(),
        ]
    )

    if not valid_layout_scope:
        assert False, (
            f"scope or layout mismatch, src: {src_buffer_region}, dst: {dst_buffer_region}"
        )

    # Extract and validate dimensions
    dst_region = dst_buffer_region.region
    src_region = src_buffer_region.region

    dst_extent = [r.extent for r in dst_region]
    src_extent = [r.extent for r in src_region]

    dst_extent_nonunit = [e for e in dst_extent if e != 1]
    src_extent_nonunit = [e for e in src_extent if e != 1]

    # Verify dimensions match
    dims_match = len(src_extent_nonunit) == len(dst_extent_nonunit) and all(
        analyzer.can_prove_equal(s, d) for s, d in zip(src_extent_nonunit, dst_extent_nonunit)
    )

    if not dims_match:
        assert False, (
            f"shape or dimension mismatch, src: {src_buffer_region}, dst: {dst_buffer_region}"
        )
    dim_map = get_ewise_dim_map(src_buffer_region, dst_buffer_region, analyzer)
    inst_gen.link_buffer_regions(src_buffer_region, dst_buffer_region, dim_map)
    # Find optimal instruction parameters
    inst_repr = inst_gen.find_max_inst_size_from_one_region(dst_buffer_region, allowed_f_dim_dst)
    inst_repr = inst_gen.fit_inst_tile_to_region(inst_repr, src_buffer_region, allowed_f_dim_src)
    return inst_repr


def get_const_bias_tensor(bias, shape, dtype, workspace, sctx):
    """Create or retrieve a constant bias tensor."""
    if "const_bias" not in workspace:
        assert sctx.alloc_only, (
            "Constant bias tensor must be specified in workspace. Run tvm.tirx.transform.trn.TrnPrivateBufferAlloc first."  # noqa: E501
        )
        # Create new bias buffer
        bias_buffer = Tx.buffer(shape, dtype, scope="trn.sbuf", buffer_name="const_bias")
        sctx.add_alloc_buffer(bias_buffer)

        @Tx.prim_func
        def const_bias_init():
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(0, shape[0], annotations={nki_dim: "P"}):
                    for f_loop in Tx.serial(0, shape[1], annotations={nki_dim: "F"}):
                        Tx.evaluate(Tx.nki.memset(bias_buffer[p_loop, f_loop], bias))
            Tx.tvm_kernel_replace_point()

        sctx.add_init_stmt(const_bias_init.body)
    else:
        # Use existing bias buffer
        bias_buffer = workspace["const_bias"]
        check_workspace_buffer(bias_buffer, shape, "trn.sbuf")

    return bias_buffer


def generate_unary_func(
    dst_buffer_region,
    _src,
    inst_gen: InstructionGenerator,
    inst_repr,
    unary_op,
    bias,
    scale,
    analyzer,
    workspace,
    config,
    sctx,
):
    """Generate a function that implements a unary operation."""
    # Prepare parameters
    p_size = dst_buffer_region.buffer.layout.size("P")

    # Apply instruction size limits if specified
    inst_size_limit = config.get("max_inst_size", 512)
    inst_repr.bound_inst_size(inst_size_limit, analyzer)

    f_var = Tx.Var("F", "int32")
    p_var = Tx.Var("P", "int32")
    b_var = Tx.Var("B", "int32")
    inst_gen.bind_inst_iter(dst_buffer_region, f_var, inst_repr.size, inst_repr.stride, True)
    inst_gen.bind_inst_iter(dst_buffer_region, p_var, p_size, 1, False)
    b_extent = inst_gen.fill_in_block_dim(dst_buffer_region, b_var)

    # Get operation code if available
    opcode = opcode_table.get(unary_op, None)

    # Extract buffers
    dst = dst_buffer_region.buffer
    src = _src.buffer if isinstance(_src, BufferRegion) else None

    # Handle bias tensor
    if isinstance(bias, FloatImm | float):
        bias_buffer = get_const_bias_tensor(
            bias, (p_size, inst_repr.size), dst.dtype, workspace, sctx
        )
    elif isinstance(bias, BufferRegion):
        bias_buffer = bias.buffer

    # fmt: off
    @Tx.prim_func
    def impl():
        for b_loop in Tx.serial(0, b_extent):
            with Tx.attr(0, "tensorized_nki_instruction", 1):
                for p_loop in Tx.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in Tx.serial(0, inst_repr.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({p_var: p_loop, f_var: f_loop, b_var: b_loop})
                        dst_indices = Tx.meta_var(inst_gen.generate_indices(dst_buffer_region))
                        if inst_gen.make_guard(dst_buffer_region):
                            if unary_op == MapOpType.FILL:
                                Tx.evaluate(Tx.nki.memset(dst[tuple(dst_indices)], _src))
                            else:
                                src_indices = Tx.meta_var(inst_gen.generate_indices(_src))
                                if unary_op == MapOpType.RECIPROCAL:
                                    Tx.evaluate(Tx.nki.reciprocal(dst[tuple(dst_indices)], src[tuple(src_indices)]))  # noqa: E501
                                elif isinstance(bias, BufferRegion):
                                    bias_indices = Tx.meta_var(inst_gen.generate_indices(bias))
                                    Tx.evaluate(Tx.nki.activation(dst[tuple(dst_indices)], src[tuple(src_indices)], opcode, scale=scale, bias=bias_buffer[tuple(bias_indices)]))  # noqa: E501
                                else:
                                    Tx.evaluate(Tx.nki.activation(dst[tuple(dst_indices)], src[tuple(src_indices)], opcode, scale=scale, bias=bias_buffer[p_loop, f_loop]))  # noqa: E501
    # fmt: on

    return impl
