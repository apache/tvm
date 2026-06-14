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
from tvm.tirx import Buffer
from tvm.tirx import op as _tir_op
from tvm.tirx.script.builder.ir import _op_wrapper


class MetalNamespace:
    """The Metal intrinsics submodule."""

    def __init__(self):
        self.make_filled_simdgroup_matrix = _op_wrapper(_metal_op.make_filled_simdgroup_matrix)
        self.simdgroup_load = _op_wrapper(_metal_op.simdgroup_load)
        self.simdgroup_store = _op_wrapper(_metal_op.simdgroup_store)
        self.simdgroup_multiply_accumulate = _op_wrapper(_metal_op.simdgroup_multiply_accumulate)

    @staticmethod
    def simd_shuffle(var, lane):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle", var, lane)

    @staticmethod
    def simd_shuffle_up(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle_up", var, delta)

    @staticmethod
    def simd_shuffle_down(var, delta):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.dtype, "tirx.metal.simd_shuffle_down", var, delta)


__all__ = ["MetalNamespace"]
