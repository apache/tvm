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
import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm.script import tir as T
from tvm.relay.backend.contrib.ethosu.tir.passes import CopyComputeReordering

# fmt: off
@tvm.script.ir_module
class AllOperatorsWithWeights:
    @T.prim_func
    def main() -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer1 = T.buffer_decl([8192], "int8")
        buffer2 = T.buffer_decl([128], "uint8")
        buffer3 = T.buffer_decl([32], "uint8")
        buffer4 = T.buffer_decl([112], "uint8")
        buffer5 = T.buffer_decl([32], "uint8")
        buffer6 = T.buffer_decl([112], "uint8")
        buffer7 = T.buffer_decl([32], "uint8")
        buffer8 = T.buffer_decl([112], "uint8")
        buffer9 = T.buffer_decl([32], "uint8")
        buffer10 = T.buffer_decl([2048], "int8")
        # body
        p1 = T.allocate([128], "uint8", "global")
        p2 = T.allocate([112], "uint8", "global")
        p3 = T.allocate([112], "uint8", "global")
        p4 = T.allocate([32], "uint8", "global")
        p5 = T.allocate([32], "uint8", "global")
        p6 = T.allocate([32], "uint8", "global")
        p7 = T.allocate([112], "uint8", "global")
        p8 = T.allocate([32], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 112, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p5[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 112, 12, p5[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 112, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 32, p6[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, 12, p6[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 112, p7[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer9[0], 32, p8[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p8[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
# fmt: on


def test_all_operators_with_weights_max_copy_movements_0():
    test_mod = CopyComputeReordering(0)(AllOperatorsWithWeights)
    reference_mod = AllOperatorsWithWeights
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_all_operators_with_weights_max_copy_movements_1():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer2 = T.buffer_decl([128], "uint8")
            buffer3 = T.buffer_decl([32], "uint8")
            buffer4 = T.buffer_decl([112], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            buffer6 = T.buffer_decl([112], "uint8")
            buffer7 = T.buffer_decl([32], "uint8")
            buffer8 = T.buffer_decl([112], "uint8")
            buffer9 = T.buffer_decl([32], "uint8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1 = T.allocate([128], "uint8", "global")
            p2 = T.allocate([112], "uint8", "global")
            p3 = T.allocate([112], "uint8", "global")
            p4 = T.allocate([32], "uint8", "global")
            p5 = T.allocate([32], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            p7 = T.allocate([112], "uint8", "global")
            p8 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 112, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 112, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 112, 12, p5[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 112, p7[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer9[0], 32, p8[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, 12, p6[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p8[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(1)(AllOperatorsWithWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_all_operators_with_weights_max_copy_movements_2():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer2 = T.buffer_decl([128], "uint8")
            buffer3 = T.buffer_decl([32], "uint8")
            buffer4 = T.buffer_decl([112], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            buffer6 = T.buffer_decl([112], "uint8")
            buffer7 = T.buffer_decl([32], "uint8")
            buffer8 = T.buffer_decl([112], "uint8")
            buffer9 = T.buffer_decl([32], "uint8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1 = T.allocate([128], "uint8", "global")
            p2 = T.allocate([112], "uint8", "global")
            p3 = T.allocate([112], "uint8", "global")
            p4 = T.allocate([32], "uint8", "global")
            p5 = T.allocate([32], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            p7 = T.allocate([112], "uint8", "global")
            p8 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 112, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 112, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 112, p7[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer9[0], 32, p8[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 112, 12, p5[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, 12, p6[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p8[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(2)(AllOperatorsWithWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


# fmt: off
@tvm.script.ir_module
class AllOperatorsWithoutWeights:
    @T.prim_func
    def main() -> None:
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})  
        buffer1 = T.buffer_decl([36], "int8")
        buffer2 = T.buffer_decl([9], "int8")
        # body
        p1 = T.allocate([96], "int8", "global")
        T.evaluate(T.call_extern("ethosu_pooling", "int8", 3, 4, 3, 3, 0, 4, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 12, 3, 1, "int8", 3, 2, 3, 3, 0, 2, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 32, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_pooling", "int8", 3, 2, 3, 3, 0, 2, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 32, 16, 1, "int8", 3, 1, 3, 3, 0, 1, buffer2[0], 0, 0, 0, T.float32(1), 0, "NHWC", 3, 1, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
# fmt: on


@pytest.mark.parametrize("max_copy_movements", [0, 1, 2])
def test_all_operators_without_weights(max_copy_movements):
    test_mod = CopyComputeReordering(max_copy_movements)(AllOperatorsWithoutWeights)
    reference_mod = AllOperatorsWithoutWeights
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


# fmt: off
@tvm.script.ir_module
class OperatorsWithAndWithoutWeights:
    @T.prim_func
    def main() -> None:
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})  
        buffer1 = T.buffer_decl([97156], "int8")
        buffer2 = T.buffer_decl([80], "uint8")
        buffer3 = T.buffer_decl([64], "uint8")
        buffer4 = T.buffer_decl([96], "uint8")
        buffer5 = T.buffer_decl([32], "uint8")
        # body
        p1 = T.allocate([390336], "int8", "global")
        p2 = T.allocate([80], "uint8", "global")
        p3 = T.allocate([64], "uint8", "global")
        p4 = T.allocate([390336], "int8", "global")
        p5 = T.allocate([96], "uint8", "global")
        p6 = T.allocate([32], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p5[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p6[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 3, 214, 0, 114, buffer3[0], 0, 0, 0, T.float32(0.104816), -128, "NHWC", 342, 3, 1, 3, 1, 1, 1, 1, 2, p5[0], 96, 0, p6[0], 32, 0, 1, 0, 1, "CLIP", -128, 127, "TFL", "NONE", 0, 0, 0, dtype="handle"))
# fmt: on


def test_operators_with_and_without_weights_max_copy_movements_0():
    test_mod = CopyComputeReordering(0)(OperatorsWithAndWithoutWeights)
    reference_mod = OperatorsWithAndWithoutWeights
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_operators_with_and_without_weights_max_copy_movements_1():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([97156], "int8")
            buffer2 = T.buffer_decl([80], "uint8")
            buffer3 = T.buffer_decl([64], "uint8")
            buffer4 = T.buffer_decl([96], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            # body
            p1 = T.allocate([390336], "int8", "global")
            p2 = T.allocate([80], "uint8", "global")
            p3 = T.allocate([64], "uint8", "global")
            p4 = T.allocate([390336], "int8", "global")
            p5 = T.allocate([96], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 3, 214, 0, 114, buffer3[0], 0, 0, 0, T.float32(0.104816), -128, "NHWC", 342, 3, 1, 3, 1, 1, 1, 1, 2, p5[0], 96, 0, p6[0], 32, 0, 1, 0, 1, "CLIP", -128, 127, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(1)(OperatorsWithAndWithoutWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_operators_with_and_without_weights_max_copy_movements_2():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})  
            buffer1 = T.buffer_decl([97156], "int8")
            buffer2 = T.buffer_decl([80], "uint8")
            buffer3 = T.buffer_decl([64], "uint8")
            buffer4 = T.buffer_decl([96], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            # body
            p1 = T.allocate([390336], "int8", "global")
            p2 = T.allocate([80], "uint8", "global")
            p3 = T.allocate([64], "uint8", "global")
            p4 = T.allocate([390336], "int8", "global")
            p5 = T.allocate([96], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 3, 214, 0, 114, buffer3[0], 0, 0, 0, T.float32(0.104816), -128, "NHWC", 342, 3, 1, 3, 1, 1, 1, 1, 2, p5[0], 96, 0, p6[0], 32, 0, 1, 0, 1, "CLIP", -128, 127, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(2)(OperatorsWithAndWithoutWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


# fmt: off
@tvm.script.ir_module
class CopyToBufferWithLocalScope:
    @T.prim_func
    def main() -> None:
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})  
        buffer1 = T.buffer_decl([64], "uint8")
        buffer2 = T.buffer_decl([48], "uint8")
        buffer3 = T.buffer_decl([48], "uint8")
        buffer4 = T.buffer_decl([256], "uint8")
        buffer5 = T.buffer_decl([16], "uint8")
        buffer6 = T.buffer_decl([48], "uint8")
        buffer7 = T.buffer_decl([256], "uint8")
        buffer8 = T.buffer_decl([64], "uint8")
        # body
        p1 = T.allocate([48], "uint8", "global")
        p2 = T.allocate([48], "uint8", "global")
        p3 = T.allocate([256], "int8", "local")
        p4 = T.allocate([256], "int8", "global")
        p5 = T.allocate([16], "uint8", "global")
        p6 = T.allocate([48], "uint8", "global")
        p7 = T.allocate([256], "int8", "local")
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 48, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 48, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 256, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer1[0], 0, 0, 0, T.float32(0.00392081), -128, "NHWC", 16, 4, 1, "int8", 4, 4, 4, 4, 0, 4, p4[0], 0, 0, 0, T.float32(0.00839574), -128, "NHCWB16", 64, 16, 1, 1, 1, 1, 1, 1, 1, p1[0], 48, 0, p2[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 16, p5[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 48, p6[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 256, p7[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 4, 4, 4, 4, 0, 4, p4[0], 0, 0, 0, T.float32(0.0078125), 0, "NHCWB16", 64, 16, 1, "int8", 4, 4, 4, 4, 0, 4, buffer8[0], 0, 0, 0, T.float32(0.00372155), -128, "NHWC", 16, 4, 1, 1, 1, 1, 1, 1, 1, p5[0], 16, 0, p6[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
# fmt: on


def test_copy_to_buffer_with_local_scope_max_copy_movements_0():
    test_mod = CopyComputeReordering(0)(CopyToBufferWithLocalScope)
    reference_mod = CopyToBufferWithLocalScope
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


@pytest.mark.parametrize("max_copy_movements", [1, 2])
def test_copy_to_buffer_with_local_scope_max_copy_movements_n(max_copy_movements):
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([64], "uint8")
            buffer2 = T.buffer_decl([48], "uint8")
            buffer3 = T.buffer_decl([48], "uint8")
            buffer4 = T.buffer_decl([256], "uint8")
            buffer5 = T.buffer_decl([16], "uint8")
            buffer6 = T.buffer_decl([48], "uint8")
            buffer7 = T.buffer_decl([256], "uint8")
            buffer8 = T.buffer_decl([64], "uint8")
            # body
            p1 = T.allocate([48], "uint8", "global")
            p2 = T.allocate([48], "uint8", "global")
            p3 = T.allocate([256], "int8", "local")
            p4 = T.allocate([256], "int8", "global")
            p5 = T.allocate([16], "uint8", "global")
            p6 = T.allocate([48], "uint8", "global")
            p7 = T.allocate([256], "int8", "local")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 48, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 48, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 256, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 16, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 48, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer1[0], 0, 0, 0, T.float32(0.00392081), -128, "NHWC", 16, 4, 1, "int8", 4, 4, 4, 4, 0, 4, p4[0], 0, 0, 0, T.float32(0.00839574), -128, "NHCWB16", 64, 16, 1, 1, 1, 1, 1, 1, 1, p1[0], 48, 0, p2[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 256, p7[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 4, 4, 4, 4, 0, 4, p4[0], 0, 0, 0, T.float32(0.0078125), 0, "NHCWB16", 64, 16, 1, "int8", 4, 4, 4, 4, 0, 4, buffer8[0], 0, 0, 0, T.float32(0.00372155), -128, "NHWC", 16, 4, 1, 1, 1, 1, 1, 1, 1, p5[0], 16, 0, p6[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(max_copy_movements)(CopyToBufferWithLocalScope)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_multiple_prim_funcs():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main():
            T.evaluate(0)

        @T.prim_func
        def abc():
            T.evaluate(0)
    # fmt: on

    err_rgx = (
        r"Expected a single primitive function called 'main'. "
        r"Please run the CopyComputeReordering pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        CopyComputeReordering(1)(InputModule)


def test_no_main_prim_func():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def abs():
            T.evaluate(0)
    # fmt: on

    err_rgx = (
        r"Expected a single primitive function called 'main'. "
        r"Please run the CopyComputeReordering pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        CopyComputeReordering(1)(InputModule)


def test_default_max_copy_movements():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([97156], "int8")
            buffer2 = T.buffer_decl([80], "uint8")
            buffer3 = T.buffer_decl([64], "uint8")
            buffer4 = T.buffer_decl([96], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            # body
            p1 = T.allocate([390336], "int8", "global")
            p2 = T.allocate([80], "uint8", "global")
            p3 = T.allocate([64], "uint8", "global")
            p4 = T.allocate([390336], "int8", "global")
            p5 = T.allocate([96], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 3, 214, 0, 114, buffer3[0], 0, 0, 0, T.float32(0.104816), -128, "NHWC", 342, 3, 1, 3, 1, 1, 1, 1, 2, p5[0], 96, 0, p6[0], 32, 0, 1, 0, 1, "CLIP", -128, 127, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering()(OperatorsWithAndWithoutWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_pass_context_option_max_copy_movements():
    # fmt: off
    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})  
            buffer1 = T.buffer_decl([97156], "int8")
            buffer2 = T.buffer_decl([80], "uint8")
            buffer3 = T.buffer_decl([64], "uint8")
            buffer4 = T.buffer_decl([96], "uint8")
            buffer5 = T.buffer_decl([32], "uint8")
            # body
            p1 = T.allocate([390336], "int8", "global")
            p2 = T.allocate([80], "uint8", "global")
            p3 = T.allocate([64], "uint8", "global")
            p4 = T.allocate([390336], "int8", "global")
            p5 = T.allocate([96], "uint8", "global")
            p6 = T.allocate([32], "uint8", "global")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, p1[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 5, 214, 0, 114, p4[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 3, 214, 0, 114, buffer3[0], 0, 0, 0, T.float32(0.104816), -128, "NHWC", 342, 3, 1, 3, 1, 1, 1, 1, 2, p5[0], 96, 0, p6[0], 32, 0, 1, 0, 1, "CLIP", -128, 127, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    with tvm.transform.PassContext(
        config={"tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements": 2}
    ):
        test_mod = CopyComputeReordering()(OperatorsWithAndWithoutWeights)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


if __name__ == "__main__":
    pytest.main([__file__])
