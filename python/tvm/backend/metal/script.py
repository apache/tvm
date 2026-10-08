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
"""Metal TVMScript namespace."""

from __future__ import annotations

from tvm.backend.metal import op as _metal_op
from tvm.ir import Op
from tvm.ir.op import _make_op_api
from tvm.tirx import is_tensor_var

_simd_shuffle = _make_op_api(Op.get("tirx.metal.simd_shuffle"), __name__)
_simd_shuffle_up = _make_op_api(Op.get("tirx.metal.simd_shuffle_up"), __name__)
_simd_shuffle_down = _make_op_api(Op.get("tirx.metal.simd_shuffle_down"), __name__)


class MetalNamespace:
    """The Metal intrinsics submodule."""

    def __init__(self):
        self.make_filled_simdgroup_matrix = _metal_op.make_filled_simdgroup_matrix
        self.simdgroup_load = _metal_op.simdgroup_load
        self.simdgroup_store = _metal_op.simdgroup_store
        self.simdgroup_multiply_accumulate = _metal_op.simdgroup_multiply_accumulate

    @staticmethod
    def simd_shuffle(var, lane, *, ty=None, span=None):
        if is_tensor_var(var):
            var = var[0]
        return _simd_shuffle(var, lane, ty=ty, span=span)

    @staticmethod
    def simd_shuffle_up(var, delta, *, ty=None, span=None):
        if is_tensor_var(var):
            var = var[0]
        return _simd_shuffle_up(var, delta, ty=ty, span=span)

    @staticmethod
    def simd_shuffle_down(var, delta, *, ty=None, span=None):
        if is_tensor_var(var):
            var = var[0]
        return _simd_shuffle_down(var, delta, ty=ty, span=span)


__all__ = ["MetalNamespace"]
