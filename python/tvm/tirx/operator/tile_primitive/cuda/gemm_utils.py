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

"""GEMM-related utilities for CUDA op dispatches."""

from tvm.arith.analyzer import Analyzer
from tvm.tirx import Buffer
from tvm.tirx.operator.tile_primitive import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall


def validate_gemm_op(op_call: TilePrimitiveCall, sctx: DispatchContext) -> bool:
    """Sanity check for gemm op"""
    C_buffer_region, A_buffer_region, B_buffer_region = op_call.args[:3]
    C: Buffer = C_buffer_region.buffer
    A: Buffer = A_buffer_region.buffer
    B: Buffer = B_buffer_region.buffer
    if not (C.layout and A.layout and B.layout and A.dtype == B.dtype):
        return False
    # Extract regions and validate dimensions
    analyzer = Analyzer()
    C_region, A_region, B_region = (
        C_buffer_region.region,
        A_buffer_region.region,
        B_buffer_region.region,
    )
    # Extract extents and validate non-unit dimensions match
    transA, transB = op_call.args[3:5]
    C_extent_ = [r.extent for r in C_region if r.extent != 1]
    A_extent_ = [r.extent for r in A_region if r.extent != 1]
    B_extent_ = [r.extent for r in B_region if r.extent != 1]
    assert len(C_extent_) == len(A_extent_) == len(B_extent_) == 2, (
        "Only 2D C, A, B are supported for gemm"
    )
    if transA:
        A_extent_ = [A_extent_[1], A_extent_[0]]
    if transB:
        B_extent_ = [B_extent_[1], B_extent_[0]]
    # C: MxN, A: MxK, B: NxK
    if not all(
        [
            analyzer.can_prove_equal(C_extent_[0], A_extent_[0]),
            analyzer.can_prove_equal(C_extent_[1], B_extent_[0]),
            analyzer.can_prove_equal(A_extent_[1], B_extent_[1]),
        ]
    ):
        return False
    return True
