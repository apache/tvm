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
from tvm.tirx import Function
from tvm.tirx.operator.tile_primitive import DispatchContext, fail
from tvm.tirx.tensor_instruction import TensorCall

from ..common import init_analyzer, nki_dim
from ..dim_utils import get_ewise_dim_map
from ..instruction_generator import InstructionGenerator


def copy_trn(op: TensorCall, sctx: DispatchContext) -> Function | None:
    """Schedule copy operation between global and shared memory on CUDA."""
    # Basic validation checks
    if sctx.scope_kind != "thread":
        fail("requires thread exec_scope for TRN copy")

    dst_region, src_region = op.args
    src, dst = src_region.source, dst_region.source

    # Check for valid buffer configurations
    valid_config = all(
        [
            src.ty.layout and dst.ty.layout,
            src.scope() in ["global", "trn.sbuf", "trn.psum"],
            dst.scope() in ["global", "trn.sbuf", "trn.psum"],
            src.scope() != "global" or dst.scope() != "global",
            (src.scope() == "global" and isinstance(src.ty.layout, T.TileLayout))
            or (src.scope() in ["trn.sbuf", "trn.psum"] and is_trainium_layout(src.ty.layout)),
            (dst.scope() == "global" and isinstance(dst.ty.layout, T.TileLayout))
            or (dst.scope() in ["trn.sbuf", "trn.psum"] and is_trainium_layout(dst.ty.layout)),
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
        raise ValueError(
            "tensor_copy cannot transpose the partition dimension; "
            "prepare identity and use matmul explicitly"
        )

    if is_trainium_layout(src.ty.layout):
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
        inst_size_limit = op.options.get("max_inst_size", 512)
        inst.bound_inst_size(inst_size_limit, analyzer)
    else:
        assert "max_inst_size" not in op.options, "max_inst_size is not supported for load/store"

    p_var = T.Var("P", "int32")
    f_var = T.Var("F", "int32")
    b_var = T.Var("B", "int32")
    if src_to_dst:
        from_region, _to_region = src_region, dst_region
    else:
        from_region, _to_region = dst_region, src_region
    p_size = from_region.source.ty.layout.size("P")
    inst_gen.bind_inst_iter(from_region, p_var, p_size, 1, is_free_dim=False)
    inst_gen.bind_inst_iter(from_region, f_var, inst.size, inst.stride, is_free_dim=True)
    b_extent = inst_gen.fill_in_block_dim(from_region, b_var)

    # fmt: off
    # This fragment captures buffers and indices from its insertion scope.
    @T.function(check_well_formed=False)
    def impl():
        # the additional b loop is to satisfy hardware instuction size limit
        for b_loop in T.serial(0, b_extent):
            with T.nki.tensorized_instruction():
                for p_loop in T.serial(0, p_size, annotations={nki_dim: "P"}):
                    for f_loop in T.serial(0, inst.size, annotations={nki_dim: "F"}):
                        inst_gen.set_bind_map_all({b_var: b_loop, p_var: p_loop, f_var: f_loop})
                        if inst_gen.make_guard(dst_region):
                            src_indices = T.meta_var(inst_gen.generate_indices(src_region))
                            dst_indices = T.meta_var(inst_gen.generate_indices(dst_region))
                            func(dst[tuple(dst_indices)], src[tuple(src_indices)])
    # fmt: on
    return impl
