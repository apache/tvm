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
import tvm
from tvm.tir import Schedule
from tvm.tir.schedule.transform import tile_with_tensor_intrin
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN, AVX512_DOT_16x4_INTRIN

from .tir_schedule_test_utils import (
    DenseTIRModule,
    DenseTIRModuleTiled,
    Conv2dNCHWcTIRModule,
    Conv2dNCHWcTIRModuleTiled,
)


def test_tile_with_tensor_intrin_dense(intrin=VNNI_DOT_16x4_INTRIN):
    s = Schedule(DenseTIRModule)
    block = s.get_block("compute")

    tiled_loop = tile_with_tensor_intrin(s, block, intrin)

    _, _, _, i1_1, _ = s.get_loops(block)

    assert s.get(tiled_loop) == s.get(i1_1)
    tvm.ir.assert_structural_equal(s.mod, DenseTIRModuleTiled)


def test_tile_with_tensor_intrin_conv2d_nchwc(intrin=VNNI_DOT_16x4_INTRIN):
    s = Schedule(Conv2dNCHWcTIRModule)
    block = s.get_block("conv2d_NCHWc_int8")
    tiled_loop = tile_with_tensor_intrin(s, block, intrin)
    tiled_loops = s.get_loops(block)
    assert len(tiled_loops) == 12
    assert s.get(tiled_loop) == s.get(tiled_loops[-2])
    tvm.ir.assert_structural_equal(s.mod, Conv2dNCHWcTIRModuleTiled)


if __name__ == "__main__":
    test_tile_with_tensor_intrin_dense()
    test_tile_with_tensor_intrin_dense(AVX512_DOT_16x4_INTRIN)
    test_tile_with_tensor_intrin_conv2d_nchwc()
    test_tile_with_tensor_intrin_conv2d_nchwc(AVX512_DOT_16x4_INTRIN)
