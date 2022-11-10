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
from tvm.relay.backend.contrib.ethosu.tir.passes import MergeConstants
import numpy as np


def check_const_dictionaries(const_dict, new_const_dict):
    assert list(const_dict) == list(new_const_dict)
    for key, value in const_dict.items():
        new_value = new_const_dict[key]
        assert len(value) == len(new_value)
        for i in range(len(value)):
            assert value[i] == new_value[i]


def test_only_one_operator():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(128,), "uint8"], buffer3: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1_data = T.allocate([128], "uint8", "global")
            p1 = T.buffer_decl([128], "uint8", data=p1_data)
            p4_data = T.allocate([32], "uint8", "global")
            p4 = T.buffer_decl([32], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(160,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p4_data = T.allocate([160], "uint8", "global")
            p4 = T.buffer_decl([160], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 160, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p4[0], 128, 12, p4[128], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on
    const_dict = {
        0: np.array([0, 10], dtype=np.uint8),
        1: np.array([1, 11], dtype=np.uint8),
    }
    new_const_dict = {0: np.concatenate((const_dict[0], const_dict[1]))}
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_all_operators_with_weights():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(128,), "uint8"], buffer3: T.Buffer[(32,), "uint8"], buffer4: T.Buffer[(112,), "uint8"], buffer5: T.Buffer[(32,), "uint8"], buffer6: T.Buffer[(112,), "uint8"], buffer7: T.Buffer[(32,), "uint8"], buffer8: T.Buffer[(112,), "uint8"], buffer9: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1_data = T.allocate([128], "uint8", "global")
            p1 = T.buffer_decl([128], "uint8", data=p1_data)
            p2_data = T.allocate([112], "uint8", "global")
            p2 = T.buffer_decl([112], "uint8", data=p2_data)
            p3_data = T.allocate([112], "uint8", "global")
            p3 = T.buffer_decl([112], "uint8", data=p3_data)
            p4_data = T.allocate([32], "uint8", "global")
            p4 = T.buffer_decl([32], "uint8", data=p4_data)
            p5_data = T.allocate([32], "uint8", "global")
            p5 = T.buffer_decl([32], "uint8", data=p5_data)
            p6_data = T.allocate([32], "uint8", "global")
            p6 = T.buffer_decl([32], "uint8", data=p6_data)
            p7_data = T.allocate([112], "uint8", "global")
            p7 = T.buffer_decl([112], "uint8", data=p7_data)
            p8_data = T.allocate([3], "uint8", "global")
            p8 = T.buffer_decl([3], "uint8", data=p8_data)
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


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(160,), "uint8"], buffer4: T.Buffer[(144,), "uint8"], buffer6: T.Buffer[(144,), "uint8"], buffer8: T.Buffer[(144,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p4_data = T.allocate([160], "uint8", "global")
            p4 = T.buffer_decl([160], "uint8", data=p4_data)
            p7_data = T.allocate([144], "uint8", "global")
            p7 = T.buffer_decl([144], "uint8", data=p7_data)
            p10_data = T.allocate([144], "uint8", "global")
            p10 = T.buffer_decl([144], "uint8", data=p10_data)
            p11_data = T.allocate([144], "uint8", "global")
            p11 = T.buffer_decl([144], "uint8", data=p11_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 160, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 144, p7[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p4[0], 128, 12, p4[128], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 144, p10[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p7[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 144, p11[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p10[0], 112, 12, p10[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p11[0], 112, 12, p11[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {
        0: np.array([0], dtype=np.uint8),
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
        3: np.array([3], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
        6: np.array([6], dtype=np.uint8),
        7: np.array([7], dtype=np.uint8),
    }
    new_const_dict = {
        0: np.concatenate((const_dict[0], const_dict[1])),
        1: np.concatenate((const_dict[2], const_dict[3])),
        2: np.concatenate((const_dict[4], const_dict[5])),
        3: np.concatenate((const_dict[6], const_dict[7])),
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_operators_with_and_without_weights():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(80,), "uint8"], buffer3: T.Buffer[(64,), "uint8"]) -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer0 = T.buffer_decl([390336], "int8")
            buffer1 = T.buffer_decl([97156], "int8")
            buffer6 = T.buffer_decl([390336], "int8")
            # body
            p2_data = T.allocate([80], "uint8", "global")
            p2 = T.buffer_decl([80], "uint8", data=p2_data)
            p3_data = T.allocate([64], "uint8", "global")
            p3 = T.buffer_decl([64], "uint8", data=p3_data)
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, buffer0[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 64, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, buffer0[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, buffer6[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p2[0], 80, 0, p3[0], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(144,), "uint8"]) -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer0 = T.buffer_decl([390336], "int8")
            buffer1 = T.buffer_decl([97156], "int8")
            buffer6 = T.buffer_decl([390336], "int8")
            # body
            p3_data = T.allocate([144], "uint8", "global")
            p3 = T.buffer_decl([144], "uint8", data=p3_data)
            T.evaluate(T.call_extern("ethosu_pooling", "int8", 214, 227, 2, 214, 0, 227, buffer1[0], 0, 0, 0, T.float32(1), 0, "NHWC", 454, 2, 1, "int8", 214, 114, 2, 214, 0, 114, buffer0[0], 0, 0, 0, T.float32(1), 0, "NHCWB16", 1824, 16, 1, "MAX", 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 144, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 214, 114, 2, 214, 0, 114, buffer0[0], 0, 0, 0, T.float32(0.00392157), -128, "NHCWB16", 1824, 16, 1, "int8", 214, 114, 5, 214, 0, 114, buffer6[0], 0, 0, 0, T.float32(0.0174839), -128, "NHCWB16", 1824, 16, 1, 3, 1, 1, 1, 1, 2, p3[0], 80, 0, p3[80], 64, 0, 1, 0, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {
        0: np.array([0], dtype=np.uint8),
        1: np.array([1], dtype=np.uint8),
    }
    new_const_dict = {0: np.concatenate((const_dict[0], const_dict[1]))}
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_copy_to_buffer_with_local_scope():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer1: T.Buffer[(64,), "uint8"],
        buffer2: T.Buffer[(48,), "uint8"],
        buffer3: T.Buffer[(256,), "uint8"],
        buffer4: T.Buffer[(256,), "uint8"],
        buffer5: T.Buffer[(16,), "uint8"],
        buffer6: T.Buffer[(48,), "uint8"],
        buffer7: T.Buffer[(256,), "uint8"],
        buffer8: T.Buffer[(64,), "uint8"],
        buffer9: T.Buffer[(256,), "int8"],
        ) -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # body
            p1_data = T.allocate([48], "uint8", "global")
            p1 = T.buffer_decl([48], "uint8", data=p1_data)
            p2_data = T.allocate([48], "uint8", "global")
            p2 = T.buffer_decl([48], "uint8", data=p2_data)
            p3_data = T.allocate([256], "int8", "local")
            p3 = T.buffer_decl([256], "int8", data=p3_data, scope="local")
            p5_data = T.allocate([16], "uint8", "global")
            p5 = T.buffer_decl([16], "uint8", data=p5_data)
            p6_data = T.allocate([48], "uint8", "global")
            p6 = T.buffer_decl([48], "uint8", data=p6_data)
            p7_data = T.allocate([256], "int8", "local")
            p7 = T.buffer_decl([256], "int8", data=p7_data, scope="local")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 48, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 48, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 256, p3[0], dtype="handle")) # Local
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 16, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 48, p6[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer1[0], 0, 0, 0, T.float32(0.00392081), -128, "NHWC", 16, 4, 1, "int8", 4, 4, 4, 4, 0, 4, buffer9[0], 0, 0, 0, T.float32(0.00839574), -128, "NHCWB16", 64, 16, 1, 1, 1, 1, 1, 1, 1, p1[0], 48, 0, p2[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 256, p7[0], dtype="handle")) # Local
            T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer9[0], 0, 0, 0, T.float32(0.0078125), 0, "NHCWB16", 64, 16, 1, "int8", 4, 4, 4, 4, 0, 4, buffer8[0], 0, 0, 0, T.float32(0.00372155), -128, "NHWC", 16, 4, 1, 1, 1, 1, 1, 1, 1, p5[0], 16, 0, p6[0], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer1: T.Buffer[(64,), "uint8"],
            buffer2: T.Buffer[(96,), "uint8"],
            buffer4: T.Buffer[(256,), "uint8"],
            buffer5: T.Buffer[(64,), "uint8"],
            buffer7: T.Buffer[(256,), "uint8"],
            buffer8: T.Buffer[(64,), "uint8"],
            buffer9: T.Buffer[(256,), "int8"],
            ) -> None:
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # body
            p1_data = T.allocate([96], "uint8", "global")
            p1 = T.buffer_decl([96], "uint8", data=p1_data)
            p2_data = T.allocate([64], "uint8", "global")
            p2 = T.buffer_decl([64], "uint8", data=p2_data)
            p3_data = T.allocate([256], "int8", "local")
            p3 = T.buffer_decl([256], "int8", data=p3_data, scope="local")
            p7_data = T.allocate([256], "int8", "local")
            p7 = T.buffer_decl([256], "int8", data=p7_data, scope="local")
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 96, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 256, p3[0], dtype="handle")) # Local
            T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 64, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer1[0], 0, 0, 0, T.float32(0.00392081), -128, "NHWC", 16, 4, 1, "int8", 4, 4, 4, 4, 0, 4, buffer9[0], 0, 0, 0, T.float32(0.00839574), -128, "NHCWB16", 64, 16, 1, 1, 1, 1, 1, 1, 1, p1[0], 48, 0, p1[48], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 256, p7[0], dtype="handle")) # Local
            T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 4, 4, 4, 4, 0, 4, buffer9[0], 0, 0, 0, T.float32(0.0078125), 0, "NHCWB16", 64, 16, 1, "int8", 4, 4, 4, 4, 0, 4, buffer8[0], 0, 0, 0, T.float32(0.00372155), -128, "NHWC", 16, 4, 1, 1, 1, 1, 1, 1, 1, p2[0], 16, 0, p2[16], 48, 0, 0, 0, 0, "TANH", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
        3: np.array([3], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
        6: np.array([6], dtype=np.uint8),
    }
    new_const_dict = {
        1: np.concatenate((const_dict[1], const_dict[2])),
        2: const_dict[3],
        3: np.concatenate((const_dict[4], const_dict[5])),
        4: const_dict[6],
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_no_copies():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main() -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([20], "int8")
            ethosu_write = T.buffer_decl([16], "int8")
            # body
            ethosu_write_4_data = T.allocate([16], "int8", "global")
            ethosu_write_4 = T.buffer_decl([16], "int8", data=ethosu_write_4_data)
            T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 1, 4, 4, 1, 0, 4, placeholder[0], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 4, 1, "int8", 1, 4, 1, 1, 0, 4, placeholder[16], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 1, 1, "int8", 1, 4, 4, 1, 0, 4, ethosu_write_4[0], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 4, 1, "MAX", 0, "CLIP", -128, 127, "TFL", 1, 4, 4, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 4, 4, 1, 0, 4, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 4, 4, 1, 0, 4, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))

    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main() -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            placeholder = T.buffer_decl([20], "int8")
            ethosu_write = T.buffer_decl([16], "int8")
            # body
            ethosu_write_4_data = T.allocate([16], "int8", "global")
            ethosu_write_4 = T.buffer_decl([16], "int8", data=ethosu_write_4_data)
            T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 1, 4, 4, 1, 0, 4, placeholder[0], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 4, 1, "int8", 1, 4, 1, 1, 0, 4, placeholder[16], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 1, 1, "int8", 1, 4, 4, 1, 0, 4, ethosu_write_4[0], 0, 0, 0, T.float32(0.00783747), -128, "NHWC", 1, 4, 1, "MAX", 0, "CLIP", -128, 127, "TFL", 1, 4, 4, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_identity", "int8", 1, 4, 4, 1, 0, 4, ethosu_write_4[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "int8", 1, 4, 4, 1, 0, 4, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 4, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {}
    new_const_dict = {}
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_copies_to_the_same_buffer():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(128,), "uint8"], buffer3: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1_data = T.allocate([128], "uint8", "global")
            p1 = T.buffer_decl([128], "uint8", data=p1_data)
            p4_data = T.allocate([32], "uint8", "global")
            p4 = T.buffer_decl([32], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(160,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p5_data = T.allocate([160], "uint8", "global")
            p5 = T.buffer_decl([160], "uint8", data=p5_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 160, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p5[0], 128, 12, p5[128], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 160, p5[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p5[0], 128, 12, p5[128], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {
        0: np.array([0], dtype=np.uint8),
        1: np.array([1], dtype=np.uint8),
    }
    new_const_dict = {0: np.concatenate((const_dict[0], const_dict[1]))}
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_read_from_the_same_buffer():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1, 16, 16, 32), "int8"], buffer1: T.Buffer[(368,), "uint8"], buffer2: T.Buffer[(96,), "uint8"], input_ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([368], "uint8", "global")
            p1 = T.buffer_decl([368], "uint8", data=p1_data)
            p2_data = T.allocate([96], "uint8", "global")
            p2 = T.buffer_decl([96], "uint8", data=p2_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 368, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 96, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p2[0], 48, p2[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        __tvm_meta__ = None


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(464,), "uint8"], input_ethosu_write: T.Buffer[(1,16,16,8), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([464], "uint8", "global")
            p1 = T.buffer_decl([464], "uint8", data=p1_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 464, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p1[368], 48, p1[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
    # fmt: on

    const_dict = {
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
    }
    new_const_dict = {1: np.concatenate((const_dict[1], const_dict[2]))}
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_arbitrary_argument_order():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(368,), "uint8"], buffer2: T.Buffer[(96,), "uint8"], input_ethosu_write: T.Buffer[(1,16,16,8), "int8"], buffer3: T.Buffer[(368,), "uint8"], buffer4: T.Buffer[(96,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([368], "uint8", "global")
            p1 = T.buffer_decl([368], "uint8", data=p1_data)
            p2_data = T.allocate([96], "uint8", "global")
            p2 = T.buffer_decl([96], "uint8", data=p2_data)
            p3_data = T.allocate([368], "uint8", "global")
            p3 = T.buffer_decl([368], "uint8", data=p3_data)
            p4_data = T.allocate([96], "uint8", "global")
            p4 = T.buffer_decl([96], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 368, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 96, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p2[0], 48, p2[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 368, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 192, p3[192], 176, 12, p4[0], 48, p4[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        __tvm_meta__ = None


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(464,), "uint8"], input_ethosu_write: T.Buffer[(1,16,16,8), "int8"], buffer2: T.Buffer[(464,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([464], "uint8", "global")
            p1 = T.buffer_decl([464], "uint8", data=p1_data)
            p2_data = T.allocate([464], "uint8", "global")
            p2 = T.buffer_decl([464], "uint8", data=p2_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 464, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p1[368], 48, p1[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 464, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 192, p2[192], 176, 12, p2[368], 48, p2[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
    # fmt: on

    const_dict = {
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
    }
    new_const_dict = {
        1: np.concatenate((const_dict[1], const_dict[2])),
        3: np.concatenate((const_dict[4], const_dict[5])),
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, False)
    check_const_dictionaries(const_dict, new_const_dict)


def test_arbitrary_argument_order_const_split():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(368,), "uint8"], input_ethosu_write: T.Buffer[(1,16,16,8), "int8"], buffer2: T.Buffer[(96,), "uint8"], buffer3: T.Buffer[(368,), "uint8"], buffer4: T.Buffer[(96,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([368], "uint8", "global")
            p1 = T.buffer_decl([368], "uint8", data=p1_data)
            p2_data = T.allocate([96], "uint8", "global")
            p2 = T.buffer_decl([96], "uint8", data=p2_data)
            p3_data = T.allocate([368], "uint8", "global")
            p3 = T.buffer_decl([368], "uint8", data=p3_data)
            p4_data = T.allocate([96], "uint8", "global")
            p4 = T.buffer_decl([96], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 368, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 96, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p2[0], 48, p2[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 368, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 192, p3[192], 176, 12, p4[0], 48, p4[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        __tvm_meta__ = None


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(464,), "uint8"], input_ethosu_write: T.Buffer[(1,16,16,8), "int8"], buffer2: T.Buffer[(464,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype="int8", data=input_placeholder.data)
            ethosu_write = T.buffer_decl(2048, dtype="int8", data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([464], "uint8", "global")
            p1 = T.buffer_decl([464], "uint8", data=p1_data)
            p2_data = T.allocate([464], "uint8", "global")
            p2 = T.buffer_decl([464], "uint8", data=p2_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 464, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p1[368], 48, p1[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 464, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 192, p2[192], 176, 12, p2[368], 48, p2[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
    # fmt: on

    const_dict = {
        1: np.array([1], dtype=np.uint8),
        3: np.array([3], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
    }
    new_const_dict = {
        1: np.concatenate((const_dict[1], const_dict[3])),
        3: np.concatenate((const_dict[4], const_dict[5])),
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_arbitrary_argument_order_const_split_mixed():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(368,), "uint8"], buffer2: T.Buffer[(368,), "uint8"], input_ethosu_write: T.Buffer[(2,16,16,8), "int8"], buffer3: T.Buffer[(96,), "uint8"], buffer4: T.Buffer[(96,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype='int8', data=input_placeholder.data)
            ethosu_write = T.buffer_decl(4096, dtype='int8', data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([368], "uint8", "global")
            p1 = T.buffer_decl([368], "uint8", data=p1_data)
            p2_data = T.allocate([368], "uint8", "global")
            p2 = T.buffer_decl([368], "uint8", data=p2_data)
            p3_data = T.allocate([96], "uint8", "global")
            p3 = T.buffer_decl([96], "uint8", data=p3_data)
            p4_data = T.allocate([96], "uint8", "global")
            p4 = T.buffer_decl([96], "uint8", data=p4_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 368, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 96, p3[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p3[0], 48, p3[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 368, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 96, p4[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 192, p2[192], 176, 12, p4[0], 48, p4[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        __tvm_meta__ = None


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(input_placeholder: T.Buffer[(1,16,16,32), "int8"], buffer1: T.Buffer[(464,), "uint8"], buffer2: T.Buffer[(464,), "uint8"], input_ethosu_write: T.Buffer[(2,16,16,8), "int8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            # buffer definition
            placeholder = T.buffer_decl(8192, dtype='int8', data=input_placeholder.data)
            ethosu_write = T.buffer_decl(4096, dtype='int8', data=input_ethosu_write.data)
            # body
            p1_data = T.allocate([464], "uint8", "global")
            p1 = T.buffer_decl([464], "uint8", data=p1_data)
            p2_data = T.allocate([464], "uint8", "global")
            p2 = T.buffer_decl([464], "uint8", data=p2_data)
            T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 464, p1[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 192, p1[192], 176, 12, p1[368], 48, p1[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 464, p2[0], dtype="handle"))
            T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[2048], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 192, p2[192], 176, 12, p2[368], 48, p2[416], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
    # fmt: on

    const_dict = {
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
    }
    new_const_dict = {
        1: np.concatenate((const_dict[1], const_dict[4])),
        2: np.concatenate((const_dict[2], const_dict[5])),
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


def test_cycle_count():
    # fmt: off
    @tvm.script.ir_module
    class InputModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(128,), "uint8"], buffer3: T.Buffer[(32,), "uint8"], buffer4: T.Buffer[(112,), "uint8"], buffer5: T.Buffer[(32,), "uint8"], buffer6: T.Buffer[(112,), "uint8"], buffer7: T.Buffer[(32,), "uint8"], buffer8: T.Buffer[(112,), "uint8"], buffer9: T.Buffer[(32,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            v1a = T.var("int32")
            v1b = T.var("int32")
            v1c = T.var("int32")
            v2a = T.var("int32")
            v2b = T.var("int32")
            v2c = T.var("int32")
            v3a = T.var("int32")
            v3b = T.var("int32")
            v3c = T.var("int32")
            v4a = T.var("int32")
            v4b = T.var("int32")
            v4c = T.var("int32")
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p1_data = T.allocate([128], "uint8", "global")
            p1 = T.buffer_decl([128], "uint8", data=p1_data)
            p2_data = T.allocate([112], "uint8", "global")
            p2 = T.buffer_decl([112], "uint8", data=p2_data)
            p3_data = T.allocate([112], "uint8", "global")
            p3 = T.buffer_decl([112], "uint8", data=p3_data)
            p4_data = T.allocate([32], "uint8", "global")
            p4 = T.buffer_decl([32], "uint8", data=p4_data)
            p5_data = T.allocate([32], "uint8", "global")
            p5 = T.buffer_decl([32], "uint8", data=p5_data)
            p6_data = T.allocate([32], "uint8", "global")
            p6 = T.buffer_decl([32], "uint8", data=p6_data)
            p7_data = T.allocate([112], "uint8", "global")
            p7 = T.buffer_decl([112], "uint8", data=p7_data)
            p8_data = T.allocate([3], "uint8", "global")
            p8 = T.buffer_decl([3], "uint8", data=p8_data)
            with T.attr(T.iter_var(v1a, None, "DataPar", ""), "pragma_compute_cycles_hint", 100):
                T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 128, p1[0], dtype="handle"))
            with T.attr(T.iter_var(v1b, None, "DataPar", ""), "pragma_compute_cycles_hint", 101):
                T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 32, p4[0], dtype="handle"))
            with T.attr(T.iter_var(v2a, None, "DataPar", ""), "pragma_compute_cycles_hint", 102):
                T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 112, p2[0], dtype="handle"))
            with T.attr(T.iter_var(v2b, None, "DataPar", ""), "pragma_compute_cycles_hint", 103):
                T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 32, p5[0], dtype="handle"))
            with T.attr(T.iter_var(v1c, None, "DataPar", ""), "pragma_compute_cycles_hint", 300):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, 12, p4[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v3a, None, "DataPar", ""), "pragma_compute_cycles_hint", 104):
                T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 112, p3[0], dtype="handle"))
            with T.attr(T.iter_var(v3b, None, "DataPar", ""), "pragma_compute_cycles_hint", 105):
                T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 32, p6[0], dtype="handle"))
            with T.attr(T.iter_var(v2c, None, "DataPar", ""), "pragma_compute_cycles_hint", 301):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p2[0], 112, 12, p5[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v4a, None, "DataPar", ""), "pragma_compute_cycles_hint", 106):
                T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 112, p7[0], dtype="handle"))
            with T.attr(T.iter_var(v4b, None, "DataPar", ""), "pragma_compute_cycles_hint", 107):
                T.evaluate(T.call_extern("ethosu_copy", buffer9[0], 32, p8[0], dtype="handle"))
            with T.attr(T.iter_var(v3c, None, "DataPar", ""), "pragma_compute_cycles_hint", 302):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, 12, p6[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v4c, None, "DataPar", ""), "pragma_compute_cycles_hint", 303):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p8[0], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))


    @tvm.script.ir_module
    class ReferenceModule:
        @T.prim_func
        def main(buffer2: T.Buffer[(160,), "uint8"], buffer4: T.Buffer[(144,), "uint8"], buffer6: T.Buffer[(144,), "uint8"], buffer8: T.Buffer[(144,), "uint8"]) -> None:
            # function attr dict
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            v1a = T.var("int32")
            v1c = T.var("int32")
            v2a = T.var("int32")
            v2c = T.var("int32")
            v3a = T.var("int32")
            v3c = T.var("int32")
            v4a = T.var("int32")
            v4c = T.var("int32")
            buffer1 = T.buffer_decl([8192], "int8")
            buffer10 = T.buffer_decl([2048], "int8")
            # body
            p4_data = T.allocate([160], "uint8", "global")
            p4 = T.buffer_decl([160], "uint8", data=p4_data)
            p7_data = T.allocate([144], "uint8", "global")
            p7 = T.buffer_decl([144], "uint8", data=p7_data)
            p10_data = T.allocate([144], "uint8", "global")
            p10 = T.buffer_decl([144], "uint8", data=p10_data)
            p11_data = T.allocate([144], "uint8", "global")
            p11 = T.buffer_decl([144], "uint8", data=p11_data)
            with T.attr(T.iter_var(v1a, None, "DataPar", ""), "pragma_compute_cycles_hint", 201):
                T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 160, p4[0], dtype="handle"))
            with T.attr(T.iter_var(v2a, None, "DataPar", ""), "pragma_compute_cycles_hint", 205):
                T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 144, p7[0], dtype="handle"))
            with T.attr(T.iter_var(v1c, None, "DataPar", ""), "pragma_compute_cycles_hint", 300):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p4[0], 128, 12, p4[128], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v3a, None, "DataPar", ""), "pragma_compute_cycles_hint", 209):
                T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 144, p10[0], dtype="handle"))
            with T.attr(T.iter_var(v2c, None, "DataPar", ""), "pragma_compute_cycles_hint", 301):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p7[0], 112, 12, p7[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v4a, None, "DataPar", ""), "pragma_compute_cycles_hint", 213):
                T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 144, p11[0], dtype="handle"))
            with T.attr(T.iter_var(v3c, None, "DataPar", ""), "pragma_compute_cycles_hint", 302):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p10[0], 112, 12, p10[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
            with T.attr(T.iter_var(v4c, None, "DataPar", ""), "pragma_compute_cycles_hint", 303):
                T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, buffer1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, buffer10[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p11[0], 112, 12, p11[112], 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    # fmt: on

    const_dict = {
        0: np.array([0], dtype=np.uint8),
        1: np.array([1], dtype=np.uint8),
        2: np.array([2], dtype=np.uint8),
        3: np.array([3], dtype=np.uint8),
        4: np.array([4], dtype=np.uint8),
        5: np.array([5], dtype=np.uint8),
        6: np.array([6], dtype=np.uint8),
        7: np.array([7], dtype=np.uint8),
    }
    new_const_dict = {
        0: np.concatenate((const_dict[0], const_dict[1])),
        1: np.concatenate((const_dict[2], const_dict[3])),
        2: np.concatenate((const_dict[4], const_dict[5])),
        3: np.concatenate((const_dict[6], const_dict[7])),
    }
    test_mod, const_dict = MergeConstants(const_dict)(InputModule)
    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod, reference_mod, True)
    check_const_dictionaries(const_dict, new_const_dict)


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
        r"Please run the MergeConstants pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        MergeConstants({})(InputModule)


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
        r"Please run the MergeConstants pass in conjunction with the LowerToTIR\(\) pass."
    )
    with pytest.raises(tvm.TVMError, match=err_rgx):
        MergeConstants({})(InputModule)
