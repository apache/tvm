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


def test_reordering_based_on_cycles():
    # fmt: off
    @tvm.script.ir_module
    class ModuleBefore:
        @T.prim_func
        def main(placeholder: T.Buffer[(256,), "int8"], placeholder_encoded: T.Buffer[(288,), "uint8"], placeholder_encoded_2: T.Buffer[(128,), "uint8"], placeholder_encoded_4: T.Buffer[(288,), "uint8"], placeholder_encoded_6: T.Buffer[(128,), "uint8"], placeholder_encoded_8: T.Buffer[(144,), "uint8"], ethosu_write: T.Buffer[(572,), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            ax0_ax1_fused_ax2_fused_ax3_fused = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_1 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_2 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_3 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_4 = T.var("int32")
            nn = T.var("int32")
            nn_1 = T.var("int32")
            nn_2 = T.var("int32")
            nn_3 = T.var("int32")
            nn_4 = T.var("int32")
            nn_5 = T.var("int32")
            nn_6 = T.var("int32")
            nn_7 = T.var("int32")
            nn_8 = T.var("int32")
            nn_9 = T.var("int32")
            T.preflattened_buffer(placeholder, [1, 8, 8, 4], dtype="int8", data=placeholder.data)
            T.preflattened_buffer(placeholder_encoded, [4, 3, 3, 4], dtype="int8")
            T.preflattened_buffer(placeholder_encoded_2, [4, 3, 3, 1], dtype="int8")
            T.preflattened_buffer(placeholder_encoded_4, [4, 3, 3, 4], dtype="int8")
            T.preflattened_buffer(placeholder_encoded_6, [4, 3, 3, 1], dtype="int8")
            T.preflattened_buffer(placeholder_encoded_8, [4, 1, 3, 4], dtype="int8")
            T.preflattened_buffer(ethosu_write, [1, 13, 11, 4], dtype="int8", data=ethosu_write.data)
            # body
            placeholder_d_d_global = T.allocate([288], "uint8", "global")
            ethosu_write_2 = T.allocate([256], "int8", "global")
            placeholder_d_d_global_2 = T.allocate([128], "uint8", "global")
            ethosu_write_3 = T.allocate([256], "int8", "global")
            placeholder_d_d_global_4 = T.allocate([288], "uint8", "global")
            ethosu_write_4 = T.allocate([256], "int8", "global")
            ethosu_write_5 = T.allocate([256], "int8", "global")
            ethosu_write_6 = T.allocate([324], "int8", "global")
            placeholder_d_global = T.allocate([128], "uint8", "global")
            ethosu_write_7 = T.allocate([324], "int8", "global")
            ethosu_write_8 = T.allocate([484], "int8", "global")
            ethosu_write_9 = T.allocate([484], "int8", "global")
            ethosu_write_10 = T.allocate([484], "int8", "global")
            placeholder_global = T.allocate([144], "uint8", "global")
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused, None, "DataPar", ""), "pragma_compute_cycles_hint", 2304):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded[0], 288, placeholder_d_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 4, 8, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global[0], 240, T.int8(-1), T.int8(-1), 12, placeholder_d_d_global[240], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 576):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_2[0], 128, placeholder_d_d_global_2[0], dtype="handle"))
            with T.attr(T.iter_var(nn_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_2[0], 0, 0, 0, T.float32(0.59999999999999998), 11, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_3[0], 0, 0, 0, T.float32(0.26000000000000001), 15, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global_2[0], 80, 13, placeholder_d_d_global_2[80], 48, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 2304):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_4[0], 288, placeholder_d_d_global_4[0], dtype="handle"))
            with T.attr(T.iter_var(nn_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_4[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global_4[0], 240, T.int8(-1), T.int8(-1), 12, placeholder_d_d_global_4[240], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 192):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "MAX", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 300):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "int8", 9, 9, 4, 9, 0, 9, ethosu_write_6[0], 0, 0, 0, T.float32(1), 0, "NHWC", 36, 4, 1, "AVG", 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 10, 10, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 576):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_6[0], 128, placeholder_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn_5, None, "DataPar", ""), "pragma_compute_cycles_hint", 500):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 9, 9, 4, 9, 0, 9, ethosu_write_6[0], 0, 0, 0, T.float32(0.59999999999999998), 11, "NHWC", 36, 4, 1, "int8", 9, 9, 4, 9, 0, 9, ethosu_write_7[0], 0, 0, 0, T.float32(0.26000000000000001), 15, "NHWC", 36, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_global[0], 80, 13, placeholder_d_global[80], 48, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 10, 10, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_6, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 9, 9, 4, 9, 0, 9, ethosu_write_7[0], 0, 0, 0, T.float32(1), 0, "NHWC", 36, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_8[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "MAX", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_7, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_8[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_9[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "AVG", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_8, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_9[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_10[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "AVG", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 768):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_8[0], 144, placeholder_global[0], dtype="handle"))
            T.attr(T.iter_var(nn_9, None, "DataPar", ""), "pragma_compute_cycles_hint", 504)
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_10[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 44, 4, 1, "int8", 13, 11, 4, 13, 0, 11, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 44, 4, 1, 3, 1, 1, 1, 1, 1, placeholder_global[0], 96, T.int8(-1), T.int8(-1), 12, placeholder_global[96], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 14, 12, 8, dtype="handle"))



    @tvm.script.ir_module
    class ModuleAfter:
        @T.prim_func
        def main(placeholder: T.Buffer[(256,), "int8"], placeholder_encoded: T.Buffer[(288,), "uint8"], placeholder_encoded_2: T.Buffer[(128,), "uint8"], placeholder_encoded_4: T.Buffer[(288,), "uint8"], placeholder_encoded_6: T.Buffer[(128,), "uint8"], placeholder_encoded_8: T.Buffer[(144,), "uint8"], ethosu_write: T.Buffer[(572,), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            ax0_ax1_fused_ax2_fused_ax3_fused = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_1 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_2 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_3 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_4 = T.var("int32")
            nn = T.var("int32")
            nn_1 = T.var("int32")
            nn_2 = T.var("int32")
            nn_3 = T.var("int32")
            nn_4 = T.var("int32")
            nn_5 = T.var("int32")
            nn_6 = T.var("int32")
            nn_7 = T.var("int32")
            nn_8 = T.var("int32")
            nn_9 = T.var("int32")
            T.preflattened_buffer(placeholder, [1, 8, 8, 4], dtype="int8", data=placeholder.data)
            T.preflattened_buffer(placeholder_encoded, [4, 3, 3, 4], dtype="int8", data=placeholder_encoded.data)
            T.preflattened_buffer(placeholder_encoded_2, [4, 3, 3, 1], dtype="int8", data=placeholder_encoded_2.data)
            T.preflattened_buffer(placeholder_encoded_4, [4, 3, 3, 4], dtype="int8", data=placeholder_encoded_4.data)
            T.preflattened_buffer(placeholder_encoded_6, [4, 3, 3, 1], dtype="int8", data=placeholder_encoded_6.data)
            T.preflattened_buffer(placeholder_encoded_8, [4, 1, 3, 4], dtype="int8", data=placeholder_encoded_8.data)
            T.preflattened_buffer(ethosu_write, [1, 13, 11, 4], dtype="int8", data=ethosu_write.data)
            # body
            placeholder_d_d_global = T.allocate([288], "uint8", "global")
            ethosu_write_2 = T.allocate([256], "int8", "global")
            placeholder_d_d_global_2 = T.allocate([128], "uint8", "global")
            ethosu_write_3 = T.allocate([256], "int8", "global")
            placeholder_d_d_global_4 = T.allocate([288], "uint8", "global")
            ethosu_write_4 = T.allocate([256], "int8", "global")
            ethosu_write_5 = T.allocate([256], "int8", "global")
            ethosu_write_6 = T.allocate([324], "int8", "global")
            placeholder_d_global = T.allocate([128], "uint8", "global")
            ethosu_write_7 = T.allocate([324], "int8", "global")
            ethosu_write_8 = T.allocate([484], "int8", "global")
            ethosu_write_9 = T.allocate([484], "int8", "global")
            ethosu_write_10 = T.allocate([484], "int8", "global")
            placeholder_global = T.allocate([144], "uint8", "global")
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused, None, "DataPar", ""), "pragma_compute_cycles_hint", 2304):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded[0], 288, placeholder_d_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 576):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_2[0], 128, placeholder_d_d_global_2[0], dtype="handle"))
            with T.attr(T.iter_var(nn, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 4, 8, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global[0], 240, T.int8(-1), T.int8(-1), 12, placeholder_d_d_global[240], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 2304):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_4[0], 288, placeholder_d_d_global_4[0], dtype="handle"))
            with T.attr(T.iter_var(nn_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_2[0], 0, 0, 0, T.float32(0.59999999999999998), 11, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_3[0], 0, 0, 0, T.float32(0.26000000000000001), 15, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global_2[0], 80, 13, placeholder_d_d_global_2[80], 48, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 576):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_6[0], 128, placeholder_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 320):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_4[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 32, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_d_global_4[0], 240, T.int8(-1), T.int8(-1), 12, placeholder_d_d_global_4[240], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 192):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "int8", 8, 8, 4, 8, 0, 8, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "MAX", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 8, 8, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 300):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 8, 8, 4, 8, 0, 8, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHWC", 32, 4, 1, "int8", 9, 9, 4, 9, 0, 9, ethosu_write_6[0], 0, 0, 0, T.float32(1), 0, "NHWC", 36, 4, 1, "AVG", 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 10, 10, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_5, None, "DataPar", ""), "pragma_compute_cycles_hint", 500):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 9, 9, 4, 9, 0, 9, ethosu_write_6[0], 0, 0, 0, T.float32(0.59999999999999998), 11, "NHWC", 36, 4, 1, "int8", 9, 9, 4, 9, 0, 9, ethosu_write_7[0], 0, 0, 0, T.float32(0.26000000000000001), 15, "NHWC", 36, 4, 1, 3, 3, 1, 1, 1, 1, placeholder_d_global[0], 80, 13, placeholder_d_global[80], 48, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 10, 10, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_6, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 9, 9, 4, 9, 0, 9, ethosu_write_7[0], 0, 0, 0, T.float32(1), 0, "NHWC", 36, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_8[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "MAX", 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 768):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_8[0], 144, placeholder_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn_7, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_8[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_9[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "AVG", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            with T.attr(T.iter_var(nn_8, None, "DataPar", ""), "pragma_compute_cycles_hint", 432):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_9[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "int8", 11, 11, 4, 11, 0, 11, ethosu_write_10[0], 0, 0, 0, T.float32(1), 0, "NHWC", 44, 4, 1, "AVG", 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 12, 12, 8, dtype="handle"))
            T.attr(T.iter_var(nn_9, None, "DataPar", ""), "pragma_compute_cycles_hint", 504)
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 11, 11, 4, 11, 0, 11, ethosu_write_10[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 44, 4, 1, "int8", 13, 11, 4, 13, 0, 11, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 44, 4, 1, 3, 1, 1, 1, 1, 1, placeholder_global[0], 96, T.int8(-1), T.int8(-1), 12, placeholder_global[96], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 14, 12, 8, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(reorder_by_cycles=True)(ModuleBefore)
    reference_mod = ModuleAfter
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


def test_reordering_based_on_cycles_luts_present():
    # fmt: off
    @tvm.script.ir_module
    class ModuleBefore:
        @T.prim_func
        def main(placeholder: T.Buffer[9075, "int8"], placeholder_encoded: T.Buffer[256, "uint8"], placeholder_encoded_2: T.Buffer[112, "uint8"], placeholder_1: T.Buffer[256, "int8"], placeholder_encoded_4: T.Buffer[112, "uint8"], placeholder_2: T.Buffer[256, "int8"], placeholder_3: T.Buffer[256, "int8"], ethosu_write: T.Buffer[2496, "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            ax0_ax1_fused_ax2_fused_ax3_fused = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_1 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_2 = T.var("int32")
            nn = T.var("int32")
            nn_1 = T.var("int32")
            nn_2 = T.var("int32")
            nn_3 = T.var("int32")
            nn_4 = T.var("int32")
            nn_5 = T.var("int32")
            T.preflattened_buffer(placeholder, [1, 55, 55, 3], dtype="int8", data=placeholder.data)
            T.preflattened_buffer(placeholder_encoded, [4, 3, 3, 3], dtype="int8")
            T.preflattened_buffer(placeholder_encoded_2, [4, 2, 3, 1], dtype="int8")
            T.preflattened_buffer(placeholder_1, [256], dtype="int8", data=placeholder_1.data)
            T.preflattened_buffer(placeholder_encoded_4, [4, 2, 3, 1], dtype="int8")
            T.preflattened_buffer(placeholder_2, [256], dtype="int8", data=placeholder_2.data)
            T.preflattened_buffer(placeholder_3, [256], dtype="int8", data=placeholder_3.data)
            T.preflattened_buffer(ethosu_write, [1, 26, 24, 4], dtype="int8", data=ethosu_write.data)
            # body
            placeholder_d_d_global = T.allocate([256], "uint8", "global")
            ethosu_write_2 = T.allocate([12544], "int8", "global")
            placeholder_local = T.allocate([256], "int8", "local")
            placeholder_d_global = T.allocate([112], "uint8", "global")
            ethosu_write_3 = T.allocate([9984], "int8", "global")
            ethosu_write_4 = T.allocate([9984], "int8", "global")
            ethosu_write_5 = T.allocate([9984], "int8", "global")
            placeholder_d_local = T.allocate([256], "int8", "local")
            placeholder_global = T.allocate([112], "uint8", "global")
            ethosu_write_6 = T.allocate([9984], "int8", "global")
            placeholder_d_local_1 = T.allocate([256], "int8", "local")
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused, None, "DataPar", ""), "pragma_compute_cycles_hint", 1728):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded[0], 256, placeholder_d_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn, None, "DataPar", ""), "pragma_compute_cycles_hint", 9920):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 55, 55, 3, 55, 0, 55, placeholder[0], 0, 0, 0, T.float32(0.0027450970374047756), -128, "NHWC", 165, 3, 1, "int8", 28, 28, 4, 28, 0, 28, ethosu_write_2[0], 0, 0, 0, T.float32(0.0095788920298218727), -128, "NHCWB16", 448, 16, 1, 3, 3, 2, 2, 1, 1, placeholder_d_d_global[0], 208, T.int8(-1), T.int8(-1), 0, placeholder_d_d_global[208], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 4, 16, 16, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_1[0], 256, placeholder_local[0], dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 384):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_2[0], 112, placeholder_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 330):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 28, 28, 4, 28, 0, 28, ethosu_write_2[0], 0, 0, 0, T.float32(0.0095788920298218727), -128, "NHCWB16", 448, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_3[0], 0, 0, 0, T.float32(0.0078157493844628334), -128, "NHCWB16", 384, 16, 1, 3, 2, 1, 1, 2, 2, placeholder_d_global[0], 64, 0, placeholder_d_global[64], 48, 0, 0, 0, 0, "SIGMOID", 0, 0, "TFL", "NONE", 5, 12, 16, dtype="handle"))
            with T.attr(T.iter_var(nn_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 411):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_3[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "MAX", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 2, 24, 16, dtype="handle"))
            with T.attr(T.iter_var(nn_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 458):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 2, 24, 16, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_2[0], 256, placeholder_d_local[0], dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 1500):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_4[0], 112, placeholder_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 10464):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_5[0], 0, 0, 0, T.float32(0.00390625), -128, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_6[0], 0, 0, 0, T.float32(0.002753810491412878), -128, "NHCWB16", 384, 16, 1, 3, 2, 1, 1, 2, 2, placeholder_global[0], 64, 0, placeholder_global[64], 48, 1, 2, 1, 2, "TANH", 0, 0, "TFL", "NONE", 5, 12, 16, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_3[0], 256, placeholder_d_local_1[0], dtype="handle"))
            T.attr(T.iter_var(nn_5, None, "DataPar", ""), "pragma_compute_cycles_hint", 5232)
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_6[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 96, 4, 1, "MAX", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 5, 24, 8, dtype="handle"))




    @tvm.script.ir_module
    class ModuleAfter:
        @T.prim_func
        def main(placeholder: T.Buffer[9075, "int8"], placeholder_encoded: T.Buffer[256, "uint8"], placeholder_encoded_2: T.Buffer[112, "uint8"], placeholder_1: T.Buffer[256, "int8"], placeholder_encoded_4: T.Buffer[112, "uint8"], placeholder_2: T.Buffer[256, "int8"], placeholder_3: T.Buffer[256, "int8"], ethosu_write: T.Buffer[2496, "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            ax0_ax1_fused_ax2_fused_ax3_fused = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_1 = T.var("int32")
            ax0_ax1_fused_ax2_fused_ax3_fused_2 = T.var("int32")
            nn = T.var("int32")
            nn_1 = T.var("int32")
            nn_2 = T.var("int32")
            nn_3 = T.var("int32")
            nn_4 = T.var("int32")
            nn_5 = T.var("int32")
            T.preflattened_buffer(placeholder, [1, 55, 55, 3], dtype="int8", data=placeholder.data)
            T.preflattened_buffer(placeholder_encoded, [4, 3, 3, 3], dtype="int8", data=placeholder_encoded.data)
            T.preflattened_buffer(placeholder_encoded_2, [4, 2, 3, 1], dtype="int8", data=placeholder_encoded_2.data)
            T.preflattened_buffer(placeholder_1, [256], dtype="int8", data=placeholder_1.data)
            T.preflattened_buffer(placeholder_encoded_4, [4, 2, 3, 1], dtype="int8", data=placeholder_encoded_4.data)
            T.preflattened_buffer(placeholder_2, [256], dtype="int8", data=placeholder_2.data)
            T.preflattened_buffer(placeholder_3, [256], dtype="int8", data=placeholder_3.data)
            T.preflattened_buffer(ethosu_write, [1, 26, 24, 4], dtype="int8", data=ethosu_write.data)
            # body
            placeholder_d_d_global = T.allocate([256], "uint8", "global")
            ethosu_write_2 = T.allocate([12544], "int8", "global")
            placeholder_local = T.allocate([256], "int8", "local")
            placeholder_d_global = T.allocate([112], "uint8", "global")
            ethosu_write_3 = T.allocate([9984], "int8", "global")
            ethosu_write_4 = T.allocate([9984], "int8", "global")
            ethosu_write_5 = T.allocate([9984], "int8", "global")
            placeholder_d_local = T.allocate([256], "int8", "local")
            placeholder_global = T.allocate([112], "uint8", "global")
            ethosu_write_6 = T.allocate([9984], "int8", "global")
            placeholder_d_local_1 = T.allocate([256], "int8", "local")
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused, None, "DataPar", ""), "pragma_compute_cycles_hint", 1728):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded[0], 256, placeholder_d_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 384):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_2[0], 112, placeholder_d_global[0], dtype="handle"))
            with T.attr(T.iter_var(nn, None, "DataPar", ""), "pragma_compute_cycles_hint", 9920):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 55, 55, 3, 55, 0, 55, placeholder[0], 0, 0, 0, T.float32(0.0027450970374047756), -128, "NHWC", 165, 3, 1, "int8", 28, 28, 4, 28, 0, 28, ethosu_write_2[0], 0, 0, 0, T.float32(0.0095788920298218727), -128, "NHCWB16", 448, 16, 1, 3, 3, 2, 2, 1, 1, placeholder_d_d_global[0], 208, T.int8(-1), T.int8(-1), 0, placeholder_d_d_global[208], 48, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 4, 16, 16, dtype="handle"))
            with T.attr(T.iter_var(ax0_ax1_fused_ax2_fused_ax3_fused_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 1500):
                T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_4[0], 112, placeholder_global[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_1[0], 256, placeholder_local[0], dtype="handle"))
            with T.attr(T.iter_var(nn_1, None, "DataPar", ""), "pragma_compute_cycles_hint", 330):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 28, 28, 4, 28, 0, 28, ethosu_write_2[0], 0, 0, 0, T.float32(0.0095788920298218727), -128, "NHCWB16", 448, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_3[0], 0, 0, 0, T.float32(0.0078157493844628334), -128, "NHCWB16", 384, 16, 1, 3, 2, 1, 1, 2, 2, placeholder_d_global[0], 64, 0, placeholder_d_global[64], 48, 0, 0, 0, 0, "SIGMOID", 0, 0, "TFL", "NONE", 5, 12, 16, dtype="handle"))
            with T.attr(T.iter_var(nn_2, None, "DataPar", ""), "pragma_compute_cycles_hint", 411):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_3[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "MAX", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 2, 24, 16, dtype="handle"))
            with T.attr(T.iter_var(nn_3, None, "DataPar", ""), "pragma_compute_cycles_hint", 458):
                T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_5[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 2, 24, 16, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_2[0], 256, placeholder_d_local[0], dtype="handle"))
            with T.attr(T.iter_var(nn_4, None, "DataPar", ""), "pragma_compute_cycles_hint", 10464):
                T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_5[0], 0, 0, 0, T.float32(0.00390625), -128, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write_6[0], 0, 0, 0, T.float32(0.002753810491412878), -128, "NHCWB16", 384, 16, 1, 3, 2, 1, 1, 2, 2, placeholder_global[0], 64, 0, placeholder_global[64], 48, 1, 2, 1, 2, "TANH", 0, 0, "TFL", "NONE", 5, 12, 16, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", placeholder_3[0], 256, placeholder_d_local_1[0], dtype="handle"))
            T.attr(T.iter_var(nn_5, None, "DataPar", ""), "pragma_compute_cycles_hint", 5232)
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 26, 24, 4, 26, 0, 24, ethosu_write_6[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 384, 16, 1, "int8", 26, 24, 4, 26, 0, 24, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 96, 4, 1, "MAX", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 5, 24, 8, dtype="handle"))
    # fmt: on

    test_mod = CopyComputeReordering(reorder_by_cycles=True)(ModuleBefore)
    reference_mod = ModuleAfter
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)


if __name__ == "__main__":
    pytest.main([__file__])
