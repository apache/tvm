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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")
import numpy as np

import tvm
from tvm.tir import stmt_functor
from tvm.script import tir as T
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator
from tvm.relay.backend.contrib.ethosu import util
import ethosu.vela.api as vapi


# fmt: off
"""A sample tir test case for translator"""
@tvm.script.ir_module
class SingleEthosUConv2D:
    @T.prim_func
    def main(placeholder_3: T.Buffer((8192,), "int8"), ethosu_conv2d_1: T.Buffer((1024,), "int8")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_4 = T.Buffer([1], "uint8")
        placeholder_5 = T.Buffer([1], "uint8")
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 8, 8, 3, 8, 0, 8, placeholder_3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 8, 8, 16, 8, 0, 8, ethosu_conv2d_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 16, 1, 1, 1, 1, 1, 1, 1, placeholder_4[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_5[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "CLIP", 0, 255, "TFL", "NONE", 0, 0, 0, dtype="uint8"))
# fmt: on


# fmt: off
"""A sample tir test case with multiple convolutions for translator"""
@tvm.script.ir_module
class MultiEthosUConv2D:
    @T.prim_func
    def main(placeholder_6: T.Buffer((192,), "int8"), ethosu_conv2d_1: T.Buffer((512,), "int8")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_9 = T.Buffer([1], "uint8")
        placeholder_7 = T.Buffer([1], "uint8")
        placeholder_8 = T.Buffer([1], "uint8")
        placeholder_5 = T.Buffer([1], "uint8")
        # body
        ethosu_conv2d_2 = T.decl_buffer([1024], "uint8")
        ethosu_conv2d_3 = T.decl_buffer([2048], "uint8")
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 3, 4, 0, 8, placeholder_6[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 4, 8, 32, 4, 0, 8, ethosu_conv2d_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 1, 1, 1, 1, 1, 1, placeholder_7[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_8[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 32, 4, 0, 8, ethosu_conv2d_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "uint8", 4, 8, 8, 4, 0, 8, ethosu_conv2d_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_9[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_5[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "CLIP", 0, 255, "TFL", "NONE", 0, 0, 0, dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 3, 4, 0, 8, placeholder_6[96], 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 4, 8, 32, 4, 0, 8, ethosu_conv2d_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 1, 1, 1, 1, 1, 1, placeholder_7[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_8[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "CLIP", 0, 255, "TFL", "NONE", 0, 0, 0, dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 32, 4, 0, 8, ethosu_conv2d_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "uint8", 4, 8, 8, 4, 0, 8, ethosu_conv2d_1[256], 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_9[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_5[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "CLIP", 0, 255, "TFL", "NONE", 0, 0, 0, dtype="uint8"))
# fmt: on


# fmt: off
"""A sample tir test case with copy operations for translator"""
@tvm.script.ir_module
class MultiEthosUCopy:
    @T.prim_func
    def main(placeholder_3: T.Buffer((8192,), "int8"), ethosu_conv2d_1: T.Buffer((2048,), "int8")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_5 = T.Buffer([1], "int32")
        placeholder_4 = T.Buffer([1], "uint8")
        # body
        placeholder_global = T.decl_buffer([256], "uint8")
        placeholder_d_global = T.decl_buffer([8], "int32")
        T.evaluate(T.call_extern("ethosu_copy", placeholder_4[0], 256,  placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", placeholder_5[0], 8, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, placeholder_3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 8, 16, 0, 16, ethosu_conv2d_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 0, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 0, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "CLIP", 0, 255, "TFL", "NONE", 0, 0, 0, dtype="handle"))
# fmt: on


# fmt: off
"""A tir test case with copy operation having a buffer size less than the minimum for a DMA operation"""
@tvm.script.ir_module
class CopyLessMinimal:
    @T.prim_func
    def main(ethos_u_0_i0: T.Buffer((1, 4), "int8"), ethosu_write: T.Buffer((1, 4), "int8")):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "global_symbol": "main", "tir.noalias": T.bool(True)})
        p1_global = T.allocate([4], "int8", "global", annotations={"disable_lower_builtin": T.bool(True)})
        ethosu_write_1 = T.allocate([4], "int8", "global", annotations={"disable_lower_builtin": T.bool(True)})
        p1 = T.Buffer((4,), "int8")
        p1_global_1 = T.Buffer((4,), "int8", data=p1_global)
        T.call_extern("handle", "ethosu_copy", p1[0], 4, p1_global_1[0])
        ethos_u_0_i0_1 = T.Buffer((4,), "int8", data=ethos_u_0_i0.data)
        ethosu_write_2 = T.Buffer((4,), "int8", data=ethosu_write_1, align=4)
        T.call_extern("handle", "ethosu_binary_elementwise", "int8", 1, 1, 4, 1, 0, 1, ethos_u_0_i0_1[0], 0, 0, 0, T.float32(0.0039170472882688046), -128, "NHWC", 1, 1, 1, "int8", 1, 1, 4, 1, 0, 1, p1_global_1[0], 0, 0, 0, T.float32(0.0028046639636158943), -128, "NHWC", 1, 1, 1, "int8", 1, 1, 4, 1, 0, 1, ethosu_write_2[0], 0, 0, 0, T.float32(0.0067217112518846989), -128, "NHWC", 1, 1, 1, "ADD", 0, "NONE", 0, 0, "TFL", 0, 0, 0, 0, 0, 0)
        ethosu_write_3 = T.Buffer((4,), "int8", data=ethosu_write.data)
        T.call_extern("handle", "ethosu_identity", "int8", 1, 4, 1, 1, 0, 4, ethosu_write_2[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "int8", 1, 4, 1, 1, 0, 4, ethosu_write_3[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1, 1, 1, "AVG", 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0)
# fmt: on


# fmt: off
"""A TIR test module of weight streaming"""
@tvm.script.ir_module
class WeightStreamOnly:
    @T.prim_func
    def main(placeholder: T.Buffer((8192,), "int8"), ethosu_write: T.Buffer((2048,), "int8")) -> None:
        buffer = T.Buffer([1], "uint8")
        buffer_1 = T.Buffer([1], "uint8")
        buffer_2 = T.Buffer([1], "uint8")
        buffer_3 = T.Buffer([1], "uint8")
        buffer_4 = T.Buffer([1], "uint8")
        buffer_5 = T.Buffer([1], "uint8")
        buffer_6 = T.Buffer([1], "uint8")
        buffer_7 = T.Buffer([1], "uint8")
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True,
                     "global_symbol": "main", "tir.noalias": True,
                     "constants": {buffer.name: buffer,
                                       buffer_1.name: buffer_1,
                                       buffer_2.name: buffer_2,
                                       buffer_3.name: buffer_3,
                                       buffer_4.name: buffer_4,
                                       buffer_5.name: buffer_5,
                                       buffer_6.name: buffer_6,
                                       buffer_7.name: buffer_7}})
        # body
        placeholder_global_data = T.allocate([128], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global = T.decl_buffer([128], "uint8", data=placeholder_global_data)
        placeholder_d_global_data = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.decl_buffer([32], "uint8", data=placeholder_d_global_data)
        T.evaluate(T.call_extern("ethosu_copy", buffer[0], 128, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_1[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 128, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_2[0], 112, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_3[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 112, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_4[0], 112, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_5[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 112, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_6[0], 112, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_7[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 112, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A TIR test module of weight streaming and direct reading"""
@tvm.script.ir_module
class MixedRead:
    @T.prim_func
    def main(placeholder: T.Buffer((8192,), "int8"), ethosu_write: T.Buffer((2048,), "int8")) -> None:
        buffer = T.Buffer([1], "uint8")
        buffer_1 = T.Buffer([1], "uint8")
        buffer_2 = T.Buffer([1], "uint8")
        buffer_3 = T.Buffer([1], "uint8")
        buffer_4 = T.Buffer([1], "uint8")
        buffer_5 = T.Buffer([1], "uint8")
        buffer_6 = T.Buffer([1], "uint8")
        buffer_7 = T.Buffer([1], "uint8")
        buffer_8 = T.Buffer([1], "uint8")
        buffer_9 = T.Buffer([1], "uint8")
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True,
                     "global_symbol": "main", "tir.noalias": True,
                     "constants": {buffer.name: buffer,
                                   buffer_1.name: buffer_1,
                                   buffer_2.name: buffer_2,
                                   buffer_3.name: buffer_3,
                                   buffer_4.name: buffer_4,
                                   buffer_5.name: buffer_5,
                                   buffer_6.name: buffer_6,
                                   buffer_7.name: buffer_7,
                                   buffer_8.name: buffer_8,
                                   buffer_9.name: buffer_9}})
        # body
        ethosu_write_1_data = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        ethosu_write_1 = T.Buffer([4096], "int8", data=ethosu_write_1_data)
        placeholder_global_data = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global = T.Buffer([80], "uint8", data=placeholder_global_data)
        placeholder_d_global_data = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.Buffer([32], "uint8", data=placeholder_d_global_data)
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, buffer[0], 592, T.int8(-1), T.int8(-1), 12, buffer_1[0], 160, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_2[0], 80, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_3[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 80, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_4[0], 80, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_5[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 80, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_6[0], 80, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_7[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 80, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_8[0], 80, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_9[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 80, T.int8(-1), T.int8(-1), 12, placeholder_d_global[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_buffer_info_extraction():
    test_cases = [
        {
            # Stimulus
            "tir_module": SingleEthosUConv2D,
            "param_dict": {
                tvm.tir.Var("placeholder_4", "uint8"): np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 3, 16], "uint8"
                ),
                tvm.tir.Var("placeholder_5", "uint8"): np.random.randint(
                    np.iinfo("int32").min, np.iinfo("int32").max, [16], "int32"
                ),
            },
            # Reference Outputs
            "data_buffers": {
                "placeholder_3": (
                    [1, 8, 8, 3],
                    "uint8",
                    tir_to_cs_translator.BufferType.input_or_output,
                ),
                "ethosu_conv2d_1": (
                    [1, 8, 8, 16],
                    "uint8",
                    tir_to_cs_translator.BufferType.input_or_output,
                ),
            },
        },
        {
            "tir_module": MultiEthosUConv2D,
            "param_dict": {
                tvm.tir.Var("placeholder_7", "uint8"): np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 3, 32], "uint8"
                ),
                tvm.tir.Var("placeholder_8", "uint8"): np.random.randint(
                    np.iinfo("int32").min, np.iinfo("int32").max, [32], "int32"
                ),
                tvm.tir.Var("placeholder_8", "uint8"): np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 32, 8], "uint8"
                ),
                tvm.tir.Var("placeholder_5", "uint8"): np.random.randint(
                    np.iinfo("int32").min, np.iinfo("int32").max, [8], "int32"
                ),
            },
            # Reference Outputs
            "data_buffers": {
                "placeholder_6": (
                    [1, 8, 8, 3],
                    "uint8",
                    tir_to_cs_translator.BufferType.input_or_output,
                ),
                "ethosu_conv2d_1": (
                    [1, 8, 8, 8],
                    "uint8",
                    tir_to_cs_translator.BufferType.input_or_output,
                ),
                "ethosu_conv2d_2": (
                    [1024],
                    "uint8",
                    tir_to_cs_translator.BufferType.scratch,
                ),
                "ethosu_conv2d_3": (
                    [2048],
                    "uint8",
                    tir_to_cs_translator.BufferType.scratch,
                ),
            },
        },
    ]
    for test_case in test_cases:
        # With Target Hooks the TIR module needs a target attached
        # and lowered via make unpacked API.
        tir_mod = test_case["tir_module"]
        tir_mod["main"] = tir_mod["main"].with_attr(
            "target", tvm.target.Target("ethos-u", host="ethos-u")
        )
        tir_mod = tvm.tir.transform.MakeUnpackedAPI()(tir_mod)
        buffer_info = tir_to_cs_translator.extract_buffer_info(tir_mod, test_case["param_dict"])
        for buffer_var, info in buffer_info.items():
            if buffer_var in test_case["param_dict"].keys():
                assert (
                    info.values.flatten() == test_case["param_dict"][buffer_var].flatten()
                ).all()
                assert info.dtype == test_case["param_dict"][buffer_var].dtype
                info.btype == tir_to_cs_translator.BufferType.constant
            else:
                buffer_name = buffer_var.name
                assert info.btype == test_case["data_buffers"][buffer_name][2]


def test_translate_ethosu_conv2d():
    test_cases = [
        {
            # Stimulus
            "tir_module": SingleEthosUConv2D,
            "param_dict": {
                1: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 3, 16], "uint8"
                ),
                2: np.random.randint(np.iinfo("int32").min, np.iinfo("int32").max, [16], "int32"),
            },
            # Reference outputs
            "ref": [
                {
                    "ifm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(8, 8, 3),
                        "tiles": vapi.NpuTileBox(8, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.5, 10),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(24, 3, 1),
                    },
                    "ofm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(8, 8, 16),
                        "tiles": vapi.NpuTileBox(8, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.25, 14),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(128, 16, 1),
                    },
                    "kernel": vapi.NpuKernel(
                        w=1, h=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
                    ),
                    "padding": vapi.NpuPadding(top=0, left=0, bottom=0, right=0),
                    "activation": {
                        "op": vapi.NpuActivationOp.NONE_OR_RELU,
                        "min": -3.5,
                        "max": 60.25,
                    },
                    "rounding_mode": vapi.NpuRoundingMode.TFL,
                    "ifm_upscale": vapi.NpuResamplingMode.NONE,
                    "w_zero_point": 12,
                }
            ],
        },
        {
            "tir_module": MultiEthosUConv2D,
            "param_dict": {
                1: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 3, 32], "uint8"
                ),
                2: np.random.randint(np.iinfo("int32").min, np.iinfo("int32").max, [32], "int32"),
                3: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [1, 1, 32, 8], "uint8"
                ),
                4: np.random.randint(np.iinfo("int32").min, np.iinfo("int32").max, [8], "int32"),
            },
            # Reference Outputs
            "ref": [
                {
                    "ifm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 3),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.5, 10),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(24, 3, 1),
                    },
                    "ofm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 32),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.25, 14),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(256, 32, 1),
                    },
                    "kernel": vapi.NpuKernel(
                        w=1, h=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
                    ),
                    "padding": vapi.NpuPadding(top=0, left=0, bottom=0, right=0),
                    "activation": {"op": None},
                    "rounding_mode": vapi.NpuRoundingMode.TFL,
                    "ifm_upscale": vapi.NpuResamplingMode.NONE,
                    "w_zero_point": 12,
                },
                {
                    "ifm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 32),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.5, 10),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(256, 32, 1),
                    },
                    "ofm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 8),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.25, 14),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(64, 8, 1),
                    },
                    "kernel": vapi.NpuKernel(
                        w=1, h=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
                    ),
                    "padding": vapi.NpuPadding(top=0, left=0, bottom=0, right=0),
                    "activation": {
                        "op": vapi.NpuActivationOp.NONE_OR_RELU,
                        "min": -3.5,
                        "max": 60.25,
                    },
                    "rounding_mode": vapi.NpuRoundingMode.TFL,
                    "ifm_upscale": vapi.NpuResamplingMode.NONE,
                    "w_zero_point": 12,
                },
                {
                    "ifm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 3),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.5, 10),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(24, 3, 1),
                    },
                    "ofm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 32),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.25, 14),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(256, 32, 1),
                    },
                    "kernel": vapi.NpuKernel(
                        w=1, h=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
                    ),
                    "padding": vapi.NpuPadding(top=0, left=0, bottom=0, right=0),
                    "activation": {
                        "op": vapi.NpuActivationOp.NONE_OR_RELU,
                        "min": -3.5,
                        "max": 60.25,
                    },
                    "rounding_mode": vapi.NpuRoundingMode.TFL,
                    "ifm_upscale": vapi.NpuResamplingMode.NONE,
                    "w_zero_point": 12,
                },
                {
                    "ifm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 32),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.5, 10),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(256, 32, 1),
                    },
                    "ofm": {
                        "data_type": vapi.NpuDataType.UINT8,
                        "shape": vapi.NpuShape3D(4, 8, 8),
                        "tiles": vapi.NpuTileBox(4, 0, 8, [0, 0, 0, 0]),
                        "quantization": vapi.NpuQuantization(0.25, 14),
                        "layout": vapi.NpuLayout.NHWC,
                        "strides": vapi.NpuShape3D(64, 8, 1),
                    },
                    "kernel": vapi.NpuKernel(
                        w=1, h=1, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1
                    ),
                    "padding": vapi.NpuPadding(top=0, left=0, bottom=0, right=0),
                    "activation": {
                        "op": vapi.NpuActivationOp.NONE_OR_RELU,
                        "min": -3.5,
                        "max": 60.25,
                    },
                    "rounding_mode": vapi.NpuRoundingMode.TFL,
                    "ifm_upscale": vapi.NpuResamplingMode.NONE,
                    "w_zero_point": 12,
                },
            ],
        },
    ]

    def extract_ethosu_conv2d_extern_calls(mod):
        """This function will obtain all ethosu_conv2d
        calls from a NPU TIR module
        Parameters
        ----------
        mod : tvm.IRModule
            This is a NPU TIR Module

        Returns
        -------
        list
            of tvm.tir.Call objects
            that are tir extern calls
            for ethosu_conv2d
        """
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_conv2d_calls = list()

        def populate_ethosu_conv2d_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_conv2d"
            ):
                ethosu_conv2d_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_conv2d_calls)
        return ethosu_conv2d_calls

    for test_case in test_cases:
        ethosu_conv2d_calls = extract_ethosu_conv2d_extern_calls(test_case["tir_module"])
        for idx, ethosu_conv2d_call in enumerate(ethosu_conv2d_calls):
            ref = test_case["ref"][idx]
            npu_op, w_zero_point = tir_to_cs_translator.translate_ethosu_conv2d(ethosu_conv2d_call)
            # Compare IFM
            assert npu_op.ifm.data_type == ref["ifm"]["data_type"]
            assert npu_op.ifm.shape == ref["ifm"]["shape"]
            assert npu_op.ifm.tiles.height_0 == ref["ifm"]["tiles"].height_0
            assert npu_op.ifm.tiles.height_1 == ref["ifm"]["tiles"].height_1
            assert npu_op.ifm.tiles.width_0 == ref["ifm"]["tiles"].width_0
            assert npu_op.ifm.quantization == ref["ifm"]["quantization"]
            assert npu_op.ifm.layout == ref["ifm"]["layout"]
            assert npu_op.ifm.strides == ref["ifm"]["strides"]
            # Compare OFM
            assert npu_op.ofm.data_type == ref["ofm"]["data_type"]
            assert npu_op.ofm.shape == ref["ofm"]["shape"]
            assert npu_op.ofm.tiles.height_0 == ref["ofm"]["tiles"].height_0
            assert npu_op.ofm.tiles.height_1 == ref["ofm"]["tiles"].height_1
            assert npu_op.ofm.tiles.width_0 == ref["ofm"]["tiles"].width_0
            assert npu_op.ofm.quantization == ref["ofm"]["quantization"]
            assert npu_op.ofm.layout == ref["ofm"]["layout"]
            assert npu_op.ofm.strides == ref["ofm"]["strides"]
            # Compare kernel and padding
            assert npu_op.kernel.__dict__ == ref["kernel"].__dict__
            assert npu_op.padding == ref["padding"]
            # Compare activation
            if ref["activation"]["op"] is None:
                assert npu_op.activation is None
            else:
                assert npu_op.activation.op_type == ref["activation"]["op"]
                assert npu_op.activation.min == ref["activation"]["min"]
                assert npu_op.activation.max == ref["activation"]["max"]
            # Compare rounding mode
            assert npu_op.rounding_mode == ref["rounding_mode"]
            # Compare ifm upscaling
            assert npu_op.ifm_upscale == ref["ifm_upscale"]
            # Compare weight quantization parameters
            assert w_zero_point == ref["w_zero_point"]


# fmt: off
"""A ethosu_depthwise_conv2d tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuDepthwiseConv2D:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, ethosu_depthwise_conv2d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_1, [18], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_2, [30], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_3 = T.match_buffer(placeholder, [192], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_depthwise_conv2d_1 = T.match_buffer(ethosu_depthwise_conv2d, [126], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 8, 8, 3, 8, 0, 8, placeholder_3[0], 0, 0, 0, T.float32(0.6), 11, "NHWC", 24, 3, 1, "int8", 6, 7, 3, 6, 0, 7, ethosu_depthwise_conv2d_1[0], 0, 0, 0, T.float32(0.26), 15, "NHWC", 21, 3, 1, 2, 3, 1, 1, 1, 1, placeholder_4[0], 18, 13, placeholder_5[0], 30, 0, 0, 0, 0, "CLIP", 15, 105, "TFL", "NONE", 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


def test_translate_ethosu_depthwise_conv2d():
    def extract_ethosu_depthwise_conv2d_extern_call(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_depthwise_conv2d_calls = list()

        def populate_ethosu_depthwise_conv2d_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_depthwise_conv2d"
            ):
                ethosu_depthwise_conv2d_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_depthwise_conv2d_calls)
        return ethosu_depthwise_conv2d_calls[0]

    depthwise_conv2d_call = extract_ethosu_depthwise_conv2d_extern_call(SingleEthosuDepthwiseConv2D)
    npu_op, w_zero_point = tir_to_cs_translator.translate_ethosu_depthwise_conv2d(
        depthwise_conv2d_call
    )

    assert npu_op.ifm.data_type == vapi.NpuDataType.INT8
    assert npu_op.ifm.shape == vapi.NpuShape3D(8, 8, 3)
    assert npu_op.ifm.tiles.height_0 == vapi.NpuTileBox(8, 0, 8, [0, 0, 0, 0]).height_0
    assert npu_op.ifm.tiles.height_1 == vapi.NpuTileBox(8, 0, 8, [0, 0, 0, 0]).height_1
    assert npu_op.ifm.tiles.width_0 == vapi.NpuTileBox(8, 0, 8, [0, 0, 0, 0]).width_0
    assert npu_op.ifm.quantization == pytest.approx(vapi.NpuQuantization(0.6, 11))
    assert npu_op.ifm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm.strides == vapi.NpuShape3D(24, 3, 1)
    # Compare OFM
    assert npu_op.ofm.data_type == vapi.NpuDataType.INT8
    assert npu_op.ofm.shape == vapi.NpuShape3D(6, 7, 3)
    assert npu_op.ofm.tiles.height_0 == vapi.NpuTileBox(6, 0, 8, [0, 0, 0, 0]).height_0
    assert npu_op.ofm.tiles.height_1 == vapi.NpuTileBox(6, 0, 7, [0, 0, 0, 0]).height_1
    assert npu_op.ofm.tiles.width_0 == vapi.NpuTileBox(6, 0, 7, [0, 0, 0, 0]).width_0
    assert npu_op.ofm.quantization == pytest.approx(vapi.NpuQuantization(0.26, 15))
    assert npu_op.ofm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ofm.strides == vapi.NpuShape3D(21, 3, 1)
    # Compare kernel and padding
    assert (
        npu_op.kernel.__dict__
        == vapi.NpuKernel(w=2, h=3, stride_x=1, stride_y=1, dilation_x=1, dilation_y=1).__dict__
    )
    assert npu_op.padding == vapi.NpuPadding(top=0, left=0, bottom=0, right=0)
    # Compare activation
    assert npu_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    assert npu_op.activation.min == 0
    assert npu_op.activation.max == pytest.approx(23.4)
    # Compare rounding mode
    assert npu_op.rounding_mode == vapi.NpuRoundingMode.TFL
    # Compare ifm upscaling
    assert npu_op.ifm_upscale == vapi.NpuResamplingMode.NONE
    # Compare weight quantization parameters
    assert w_zero_point == 13


def test_translate_ethosu_copy():
    def extract_ethosu_copy_extern_calls(mod):
        """This function will obtain all ethosu_conv2d
        calls from a NPU TIR module
        Parameters
        ----------
        mod : tvm.IRModule
            This is a NPU TIR Module

        Returns
        -------
        list
            of tvm.tir.Call objects
            that are tir extern calls
            for ethosu_conv2d
        """
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_copy_calls = list()

        def populate_ethosu_copy_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_copy"
            ):
                ethosu_copy_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_copy_calls)
        return ethosu_copy_calls

    test_cases = [
        {
            # Stimulus
            "tir_module": MultiEthosUCopy,
            "param_dict": {
                1: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [8, 1, 1, 32], "uint8"
                ),
                2: np.random.randint(np.iinfo("int32").min, np.iinfo("int32").max, [8], "int32"),
            },
            # Reference outputs
            "ref": [
                {
                    "src": "placeholder_4",
                    "dest": "placeholder_global",
                    "length": 256,
                },
                {
                    "src": "placeholder_5",
                    "dest": "placeholder_d_global",
                    "length": 32,
                },
            ],
        },
        {
            # Mod contains a copy operation with a buffer size of 4 bytes and it should be replaced by 16
            "tir_module": CopyLessMinimal,
            "param_dict": {
                1: np.random.randint(np.iinfo("int8").min, np.iinfo("int8").max, [1, 4], "int8"),
            },
            # Reference outputs
            "ref": [
                {
                    "src": "p1",
                    "dest": "p1_global_1",
                    "length": 16,
                },
            ],
        },
    ]

    for test_case in test_cases:
        ethosu_copy_calls = extract_ethosu_copy_extern_calls(test_case["tir_module"])
        for idx, ethosu_copy_call in enumerate(ethosu_copy_calls):
            npu_dma_op = tir_to_cs_translator.translate_ethosu_tir_call_extern(ethosu_copy_call)
            assert npu_dma_op.src.address.buffer.name == test_case["ref"][idx]["src"]
            assert npu_dma_op.dest.address.buffer.name == test_case["ref"][idx]["dest"]
            assert npu_dma_op.src.length == test_case["ref"][idx]["length"]
            assert npu_dma_op.dest.length == test_case["ref"][idx]["length"]


# fmt: off
@tvm.script.ir_module
class MixedConstantDatatypes:
    @T.prim_func
    def main(placeholder_4: T.Buffer((2048,), "int8"), ethosu_write_1: T.Buffer((16,), "int8")) -> None:
        buffer = T.Buffer([1], "uint8")
        buffer_1 = T.Buffer([1], "uint8")
        buffer_2 = T.Buffer([1], "int16")
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True,
                     "global_symbol": "main", "tir.noalias": True,
                     "constants": {buffer.name: buffer,
                                   buffer_1.name: buffer_1,
                                   buffer_2.name: buffer_2}})
        # body
        placeholder_global = T.decl_buffer([272], "uint8")
        placeholder_d_global = T.decl_buffer([160], "uint8")
        ethosu_write_2 = T.decl_buffer([16], "int16")
        placeholder_d_global_1 = T.decl_buffer([1], "int16")
        T.evaluate(T.call_extern("ethosu_copy", buffer_1[0], 272, placeholder_global[0], dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_copy", buffer[0], 160, placeholder_d_global[0], dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_depthwise_conv2d", "int8", 8, 16, 16, 8, 0, 16, placeholder_4[0], 0, 0, 0, T.float32(0.0039215548895299435), -128, "NHWC", 256, 16, 1, "int16", 1, 1, 16, 1, 0, 1, ethosu_write_2[0], 0, 0, 0, T.float32(0.0023205536417663097), -128, "NHWC", 1, 1, 1, 16, 8, 1, 1, 1, 1, placeholder_global[0], 272, 0, placeholder_d_global[0], 160, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="int16"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_2[0], 1, placeholder_d_global_1[0], dtype="int16"))
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int16", 1, 1, 16, 1, 0, 1, ethosu_write_2[0], 0, 0, 0, T.float32(0.0023205536417663097), -128, "NHWC", 1, 1, 1, "int16", 1, 1, 1, 1, 0, 1, placeholder_d_global_1[0], 0, 0, 0, T.float32(0.0078125018482064768), 0, "NHWC", 1, 1, 1, "int8", 1, 1, 16, 1, 0, 1, ethosu_write_1[0], 0, 0, 0, T.float32(0.0023205536417663097), -128, "NHWC", 1, 1, 1, "MUL", 0, "NONE", 0, 0, "NATURAL", 0, 0, 0, 0, 0, 0, dtype="int8"))
# fmt: on


def test_assign_addresses():
    test_cases = [
        {
            # Stimulus
            "tir_module": WeightStreamOnly,
            "param_dict": {
                WeightStreamOnly["main"].attrs["constants"]["buffer"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [128], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_1"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_2"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [112], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_3"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_4"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [112], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_5"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_6"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [112], "uint8"
                ),
                WeightStreamOnly["main"].attrs["constants"]["buffer_7"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
            },
        },
        {
            # Stimulus
            "tir_module": MixedRead,
            "param_dict": {
                MixedRead["main"].attrs["constants"]["buffer"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [592], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_1"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [160], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_2"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_3"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_4"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_5"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_6"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_7"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_8"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"
                ),
                MixedRead["main"].attrs["constants"]["buffer_9"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [32], "uint8"
                ),
            },
        },
        {
            # Stimulus
            "tir_module": MixedConstantDatatypes,
            "param_dict": {
                MixedConstantDatatypes["main"].attrs["constants"]["buffer"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [160], "uint8"
                ),
                MixedConstantDatatypes["main"].attrs["constants"]["buffer_2"]: np.random.randint(
                    np.iinfo("int16").min, np.iinfo("int16").max, [1], "int16"
                ),
                MixedConstantDatatypes["main"].attrs["constants"]["buffer_1"]: np.random.randint(
                    np.iinfo("uint8").min, np.iinfo("uint8").max, [272], "uint8"
                ),
            },
        },
    ]

    def extract_call_extern_list(mod):
        """This function will obtain all ethosu_conv2d
        calls from a NPU TIR module
        Parameters
        ----------
        mod : tvm.IRModule
            This is a NPU TIR Module

        Returns
        -------
        list
            of tvm.tir.Call objects
            that are tir extern calls
            for ethosu_conv2d
        """
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        extern_calls = list()

        def populate_extern_calls(stmt):
            if isinstance(stmt, tvm.tir.Call) and stmt.op.name == "tir.call_extern":
                extern_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_extern_calls)
        return extern_calls

    def collect_tir_buffer_info(npu_ops):
        """This is run prior to address assigning to collect tir buffer information
        for verification later on"""
        _npu_op_tir_buffers = dict()
        for npu_op in npu_ops:
            if isinstance(npu_op, vapi.NpuDmaOperation):
                _npu_op_tir_buffers[npu_op] = (npu_op.src.address, npu_op.dest.address)
            elif issubclass(type(npu_op), vapi.NpuBlockOperation):
                _npu_op_tir_buffers[npu_op] = (
                    npu_op.ifm.tiles.addresses[0],
                    npu_op.ofm.tiles.addresses[0],
                    npu_op.weights,
                    npu_op.biases,
                )
        return _npu_op_tir_buffers

    def _check_buffer(address, region, length, buffer_var):
        """Checks whether the buffer information is valid with
        original tir buffers.
        - If its constant, this will check
          the slice in the constant tensor has the values.
        - If its scratch, this will check
          the slice is within scratch and does not have conflicts
          with other scratch tensors.
        - If its input/output, this will check the
          address is zero
        """
        inverse_region_map = {
            0: tir_to_cs_translator.BufferType.constant,
            1: tir_to_cs_translator.BufferType.scratch,
            3: tir_to_cs_translator.BufferType.input,
            4: tir_to_cs_translator.BufferType.output,
        }
        buffer_type = inverse_region_map[region]
        buffer_dtype = buffer_var.type_annotation.element_type.dtype
        dtype_bytes = np.iinfo(np.dtype(buffer_dtype)).bits // 8
        if buffer_type == tir_to_cs_translator.BufferType.constant:
            ref = buffer_info[buffer_var].values
            hex_from = address * dtype_bytes * 2
            hex_to = hex_from + length * dtype_bytes * 2
            constant_hex = constant_hex_string[hex_from:hex_to]
            constant_tensor = np.frombuffer(bytearray.fromhex(constant_hex), dtype=buffer_dtype)
            np.array_equal(constant_tensor, ref)
            # Every buffer is adjusted to align to 16 bytes
            length = util.round_up(length, 16)
            # Mark these constants are read at least once
            constant_tensor_read_mask[address : address + length] = np.ones(
                length, dtype=buffer_dtype
            )
        elif buffer_type == tir_to_cs_translator.BufferType.scratch:
            assert address < tvmbaw_workspace_size

            size_in_bytes = allocate_node_sizes[buffer_var]
            # Every buffer is adjusted to align to 16 bytes
            size_in_bytes = util.round_up(size_in_bytes, 16)
            assert address + size_in_bytes <= tvmbaw_workspace_size
            # The scratch area should not be used by any other buffer
            assert not tvmbaw_workspace_mask[address : address + size_in_bytes].any()
            # The scratch area is marked as used
            tvmbaw_workspace_mask[address : address + size_in_bytes] = np.ones(
                size_in_bytes, dtype="uint8"
            )
        elif buffer_type == tir_to_cs_translator.BufferType.input:
            assert address == 0
        else:
            assert buffer_type == tir_to_cs_translator.BufferType.output
            assert address == 0

    def _get_allocate_node_sizes(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]
        _allocate_node_sizes = dict()

        def analyze_remaining_allocates(stmt):
            if isinstance(stmt, tvm.tir.stmt.Allocate):
                allocate = stmt
                pointer_type = allocate.buffer_var.type_annotation
                storage_scope = pointer_type.storage_scope
                if storage_scope == "global":
                    dtype_bytes = np.iinfo(np.dtype(allocate.dtype)).bits // 8
                    size_in_bytes = int(dtype_bytes * np.prod(list(allocate.extents)))
                    # Every memory address the NPU access have to be 16 byte aligned
                    size_in_bytes = util.round_up(size_in_bytes, 16)
                    _allocate_node_sizes[allocate.buffer_var] = size_in_bytes

        tvm.tir.stmt_functor.post_order_visit(primfunc.body, analyze_remaining_allocates)
        return _allocate_node_sizes

    def verify(npu_ops):
        """This wrapper verifies the allocated addresses matches with original tir buffers"""
        checked_buffers = set()

        def check_buffer(address, region, length, buffer_var):
            if buffer_var not in checked_buffers:
                _check_buffer(address, region, length, buffer_var)
                checked_buffers.add(buffer_var)

        for npu_op in npu_ops:
            if isinstance(npu_op, vapi.NpuDmaOperation):
                src_tir_buffer_var = npu_op_tir_buffers[npu_op][0].buffer.data
                check_buffer(
                    npu_op.src.address, npu_op.src.region, npu_op.src.length, src_tir_buffer_var
                )
                dest_tir_load = npu_op_tir_buffers[npu_op][1].buffer.data
                check_buffer(
                    npu_op.dest.address,
                    npu_op.dest.region,
                    npu_op.dest.length,
                    dest_tir_load,
                )
            elif issubclass(type(npu_op), vapi.NpuBlockOperation):
                ifm_tir_buffer_var = npu_op_tir_buffers[npu_op][0].buffer.data
                ifm_length = (
                    npu_op.ifm.shape.height * npu_op.ifm.shape.width * npu_op.ifm.shape.depth
                )
                check_buffer(
                    npu_op.ifm.tiles.addresses[0],
                    npu_op.ifm.region,
                    ifm_length,
                    ifm_tir_buffer_var,
                )
                ofm_tir_buffer_var = npu_op_tir_buffers[npu_op][1].buffer.data
                ofm_length = (
                    npu_op.ofm.shape.height * npu_op.ofm.shape.width * npu_op.ofm.shape.depth
                )
                check_buffer(
                    npu_op.ofm.tiles.addresses[0],
                    npu_op.ofm.region,
                    ofm_length,
                    ofm_tir_buffer_var,
                )
                for idx, weight in enumerate(npu_op_tir_buffers[npu_op][2]):
                    assert isinstance(weight, vapi.NpuAddressRange)
                    check_buffer(
                        npu_op.weights[idx].address,
                        npu_op.weights[idx].region,
                        npu_op.weights[idx].length,
                        weight.address.buffer.data,
                    )
                for idx, bias in enumerate(npu_op_tir_buffers[npu_op][3]):
                    assert isinstance(bias, vapi.NpuAddressRange)
                    check_buffer(
                        npu_op.biases[idx].address,
                        npu_op.biases[idx].region,
                        npu_op.biases[idx].length,
                        bias.address.buffer.data,
                    )

    for test_case in test_cases:
        tir_mod = test_case["tir_module"]
        tir_mod["main"] = tir_mod["main"].with_attr(
            "target", tvm.target.Target("ethos-u", host="ethos-u")
        )
        tir_mod = tvm.tir.transform.MakeUnpackedAPI()(tir_mod)
        candidate_regions_for_scratch = [5, 2, 1]
        (
            scratch_region_map,
            tvmbaw_workspace_size,
            _,
        ) = tir_to_cs_translator.analyze_scratch_memory_acesses(
            tir_mod, candidate_regions_for_scratch
        )
        allocate_node_sizes = _get_allocate_node_sizes(tir_mod)
        buffer_info = tir_to_cs_translator.extract_buffer_info(tir_mod, test_case["param_dict"])
        extern_calls = extract_call_extern_list(tir_mod)
        _npu_ops = list()
        for extern_call in extern_calls:
            _npu_ops.append(tir_to_cs_translator.translate_ethosu_tir_call_extern(extern_call))
        npu_op_tir_buffers = collect_tir_buffer_info(_npu_ops)
        (_npu_ops, constant_hex_string) = tir_to_cs_translator.assign_addresses(
            buffer_info, _npu_ops, scratch_region_map
        )
        tvmbaw_workspace_mask = np.zeros(tvmbaw_workspace_size, dtype="uint8")
        constant_tensor_read_mask = np.zeros(len(constant_hex_string) // 2, dtype="uint8")
        verify(_npu_ops)
        # This will be only 1 if all allocated scratch is used.
        assert np.prod(tvmbaw_workspace_mask) == 1
        # This will be only 1 if all constant tensors is read at least once.
        assert np.prod(constant_tensor_read_mask) == 1


# fmt: off
"""A ethosu_pooling tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuPooling:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_3: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [75], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_pooling", "int8", 5, 9, 3, 5, 0, 9, placeholder_4[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 5, 3, 5, 0, 5, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 15, 3, 1, "AVG", 2, 3, 2, 1, 1, 1, 1, 1, 1, 0, "CLIP", 10, 100, "TFL", "NONE", 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


def test_translate_ethosu_pooling():
    def extract_ethosu_pooling_extern_call(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_pooling_calls = list()

        def populate_ethosu_pooling_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_pooling"
            ):
                ethosu_pooling_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_pooling_calls)
        return ethosu_pooling_calls[0]

    pooling_call = extract_ethosu_pooling_extern_call(SingleEthosuPooling)
    npu_op = tir_to_cs_translator.translate_ethosu_pooling(pooling_call)

    assert npu_op.ifm.data_type == vapi.NpuDataType.INT8
    assert npu_op.ifm.shape == vapi.NpuShape3D(5, 9, 3)
    assert npu_op.ifm.tiles.height_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_0
    assert npu_op.ifm.tiles.height_1 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_1
    assert npu_op.ifm.tiles.width_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).width_0
    assert npu_op.ifm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ifm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm.strides == vapi.NpuShape3D(27, 3, 1)
    # Compare OFM
    assert npu_op.ofm.data_type == vapi.NpuDataType.INT8
    assert npu_op.ofm.shape == vapi.NpuShape3D(5, 5, 3)
    assert npu_op.ofm.tiles.height_0 == vapi.NpuTileBox(5, 0, 5, [0, 0, 0, 0]).height_0
    assert npu_op.ofm.tiles.height_1 == vapi.NpuTileBox(5, 0, 5, [0, 0, 0, 0]).height_1
    assert npu_op.ofm.tiles.width_0 == vapi.NpuTileBox(5, 0, 5, [0, 0, 0, 0]).width_0
    assert npu_op.ofm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ofm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ofm.strides == vapi.NpuShape3D(15, 3, 1)
    # Compare pooling_type
    assert npu_op.sub_op_type == vapi.NpuPoolingOp.AVERAGE
    # Compare kernel and padding
    assert (
        npu_op.kernel.__dict__
        == vapi.NpuKernel(w=2, h=3, stride_x=2, stride_y=1, dilation_x=1, dilation_y=1).__dict__
    )
    assert npu_op.padding == vapi.NpuPadding(top=1, left=1, bottom=1, right=0)
    # Compare activation
    assert npu_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
    assert npu_op.activation.min == 10
    assert npu_op.activation.max == 100
    # Compare rounding mode
    assert npu_op.rounding_mode == vapi.NpuRoundingMode.TFL
    # Compare ifm upscaling
    assert npu_op.ifm_upscale == vapi.NpuResamplingMode.NONE


# fmt: off
"""A ethosu_binary_elementwise ADD tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseAdd:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(
            placeholder, [270], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        ethosu_write_2 = T.match_buffer(
            ethosu_write, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        # body
        T.evaluate(T.call_extern( "ethosu_binary_elementwise", "int8", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "ADD", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))

    __tvm_meta__ = None
# fmt: on

# fmt: off
"""A ethosu_binary_elementwise SUB tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseSub:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "SUB", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on

# fmt: off
"""A ethosu_binary_elementwise MUL tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMul:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "MUL", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise MIN tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMin:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "MIN", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise Max tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMax:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int8", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "MAX", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise SHR tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseShr:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int32", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int32", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int32", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "SHR", 0, "NONE", 0, 0, "TFL", 0, 0, 0, 0, 0, 0, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise SHL tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseShl:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [270], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [135], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int32", 5, 9, 3, 5, 0, 9, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int32", 5, 9, 3, 5, 0, 9, placeholder_2[135], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "int32", 5, 9, 3, 5, 0, 9, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 27, 3, 1, "SHL", 0, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize("operator_type", ["ADD", "SUB", "MUL", "MIN", "MAX", "SHR", "SHL"])
def test_translate_ethosu_binary_elementwise(operator_type):
    if operator_type == "SHR" or operator_type == "SHL":
        data_type = vapi.NpuDataType.INT32
        data_type_bytes = 4
    else:
        data_type = vapi.NpuDataType.INT8
        data_type_bytes = 1

    def extract_ethosu_binary_elementwise_call_extern(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_binary_elementwise_calls = list()

        def populate_ethosu_binary_elementwise_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_binary_elementwise"
            ):
                ethosu_binary_elementwise_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_binary_elementwise_calls)
        return ethosu_binary_elementwise_calls[0]

    if operator_type == "ADD":
        binary_elementwise = SingleEthosuBinaryElementwiseAdd
    elif operator_type == "SUB":
        binary_elementwise = SingleEthosuBinaryElementwiseSub
    elif operator_type == "MUL":
        binary_elementwise = SingleEthosuBinaryElementwiseMul
    elif operator_type == "MIN":
        binary_elementwise = SingleEthosuBinaryElementwiseMin
    elif operator_type == "MAX":
        binary_elementwise = SingleEthosuBinaryElementwiseMax
    elif operator_type == "SHR":
        binary_elementwise = SingleEthosuBinaryElementwiseShr
    elif operator_type == "SHL":
        binary_elementwise = SingleEthosuBinaryElementwiseShl
    binary_elementwise_call = extract_ethosu_binary_elementwise_call_extern(binary_elementwise)
    npu_op = tir_to_cs_translator.translate_ethosu_binary_elementwise(binary_elementwise_call)

    # Compare IFM
    assert npu_op.ifm.data_type == data_type
    assert npu_op.ifm.shape == vapi.NpuShape3D(5, 9, 3)
    assert npu_op.ifm.tiles.height_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_0
    assert npu_op.ifm.tiles.height_1 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_1
    assert npu_op.ifm.tiles.width_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).width_0
    assert npu_op.ifm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ifm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm.strides == vapi.NpuShape3D(
        27 * data_type_bytes, 3 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare IFM2
    assert npu_op.ifm2.data_type == data_type
    assert npu_op.ifm2.shape == vapi.NpuShape3D(5, 9, 3)
    assert npu_op.ifm2.tiles.height_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_0
    assert npu_op.ifm2.tiles.height_1 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_1
    assert npu_op.ifm2.tiles.width_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).width_0
    assert npu_op.ifm2.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ifm2.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm2.strides == vapi.NpuShape3D(
        27 * data_type_bytes, 3 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare OFM
    assert npu_op.ofm.data_type == data_type
    assert npu_op.ofm.shape == vapi.NpuShape3D(5, 9, 3)
    assert npu_op.ofm.tiles.height_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_0
    assert npu_op.ofm.tiles.height_1 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).height_1
    assert npu_op.ofm.tiles.width_0 == vapi.NpuTileBox(5, 0, 9, [0, 0, 0, 0]).width_0
    assert npu_op.ofm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ofm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ofm.strides == vapi.NpuShape3D(
        27 * data_type_bytes, 3 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare op type
    if operator_type == "ADD":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.ADD
    elif operator_type == "SUB":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SUB
    elif operator_type == "MUL":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MUL
    elif operator_type == "MIN":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MIN
    elif operator_type == "MAX":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MAX
    elif operator_type == "SHR":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SHR
    elif operator_type == "SHL":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SHL
    # Compare reversed_operands
    assert npu_op.reversed_operands == False
    # Compare activation
    if operator_type == "SHR":
        assert npu_op.activation is None
    else:
        assert npu_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
        assert npu_op.activation.min == 10
        assert npu_op.activation.max == 100
    # Compare rounding mode
    assert npu_op.rounding_mode == vapi.NpuRoundingMode.TFL


# fmt: off
"""A ethosu_binary_elementwise ADD with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseAddBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int8", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int8", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "ADD", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on

# fmt: off
"""A ethosu_binary_elementwise SUB with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseSubBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int8", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int8", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "SUB", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on

# fmt: off
"""A ethosu_binary_elementwise MUL with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMulBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int8", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int8", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "MUL", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise MIN with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMinBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int8", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int8", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "MIN", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise MAX with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseMaxBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int8", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int8", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "MAX", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int8"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise SHR with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseShrBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int32", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int32", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int32", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "SHR", 1, "NONE", 0, 0, "TFL", 0, 0, 0, 0, 0, 0, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A ethosu_binary_elementwise SHL with broadcasting tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuBinaryElementwiseShlBroadcasting:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_2 = T.match_buffer(placeholder, [27], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        ethosu_write_2 = T.match_buffer(ethosu_write, [24], dtype="int32", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int32", 2, 3, 4, 2, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "int32", 1, 3, 1, 1, 0, 3, placeholder_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 1, 1, 1, "int32", 2, 3, 4, 2, 0, 3, ethosu_write_2[0], 0, 0, 0, T.float32(1.0), 0, "NHWC", 12, 4, 1, "SHL", 1, "CLIP", 10, 100, "TFL", 0, 0, 0, 0, 0, 0, dtype="int32"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize("operator_type", ["ADD", "SUB", "MUL", "MIN", "MAX", "SHR", "SHL"])
def test_translate_ethosu_binary_elementwise_broadcasting(operator_type):
    if operator_type == "SHR" or operator_type == "SHL":
        data_type = vapi.NpuDataType.INT32
        data_type_bytes = 4
    else:
        data_type = vapi.NpuDataType.INT8
        data_type_bytes = 1

    def extract_ethosu_binary_elementwise_broadcasting_call_extern(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_binary_elementwise_calls = list()

        def populate_ethosu_binary_elementwise_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_binary_elementwise"
            ):
                ethosu_binary_elementwise_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_binary_elementwise_calls)
        return ethosu_binary_elementwise_calls[0]

    if operator_type == "ADD":
        binary_elementwise = SingleEthosuBinaryElementwiseAddBroadcasting
    elif operator_type == "SUB":
        binary_elementwise = SingleEthosuBinaryElementwiseSubBroadcasting
    elif operator_type == "MUL":
        binary_elementwise = SingleEthosuBinaryElementwiseMulBroadcasting
    elif operator_type == "MIN":
        binary_elementwise = SingleEthosuBinaryElementwiseMinBroadcasting
    elif operator_type == "MAX":
        binary_elementwise = SingleEthosuBinaryElementwiseMaxBroadcasting
    elif operator_type == "SHR":
        binary_elementwise = SingleEthosuBinaryElementwiseShrBroadcasting
    elif operator_type == "SHL":
        binary_elementwise = SingleEthosuBinaryElementwiseShlBroadcasting
    binary_elementwise_call = extract_ethosu_binary_elementwise_broadcasting_call_extern(
        binary_elementwise
    )
    npu_op = tir_to_cs_translator.translate_ethosu_binary_elementwise(binary_elementwise_call)

    # Compare IFM
    assert npu_op.ifm.data_type == data_type
    assert npu_op.ifm.shape == vapi.NpuShape3D(2, 3, 4)
    assert npu_op.ifm.tiles.height_0 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).height_0
    assert npu_op.ifm.tiles.height_1 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).height_1
    assert npu_op.ifm.tiles.width_0 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).width_0
    assert npu_op.ifm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ifm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm.strides == vapi.NpuShape3D(
        12 * data_type_bytes, 4 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare IFM2
    assert npu_op.ifm2.data_type == data_type
    assert npu_op.ifm2.shape == vapi.NpuShape3D(1, 3, 1)
    assert npu_op.ifm2.tiles.height_0 == vapi.NpuTileBox(1, 0, 3, [0, 0, 0, 0]).height_0
    assert npu_op.ifm2.tiles.height_1 == vapi.NpuTileBox(1, 0, 3, [0, 0, 0, 0]).height_1
    assert npu_op.ifm2.tiles.width_0 == vapi.NpuTileBox(1, 0, 3, [0, 0, 0, 0]).width_0
    assert npu_op.ifm2.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ifm2.layout == vapi.NpuLayout.NHWC
    assert npu_op.ifm2.strides == vapi.NpuShape3D(
        1 * data_type_bytes, 1 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare OFM
    assert npu_op.ofm.data_type == data_type
    assert npu_op.ofm.shape == vapi.NpuShape3D(2, 3, 4)
    assert npu_op.ofm.tiles.height_0 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).height_0
    assert npu_op.ofm.tiles.height_1 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).height_1
    assert npu_op.ofm.tiles.width_0 == vapi.NpuTileBox(2, 0, 3, [0, 0, 0, 0]).width_0
    assert npu_op.ofm.quantization == vapi.NpuQuantization(1.0, 0)
    assert npu_op.ofm.layout == vapi.NpuLayout.NHWC
    assert npu_op.ofm.strides == vapi.NpuShape3D(
        12 * data_type_bytes, 4 * data_type_bytes, 1 * data_type_bytes
    )
    # Compare op type
    if operator_type == "ADD":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.ADD
    elif operator_type == "SUB":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SUB
    elif operator_type == "MUL":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MUL
    elif operator_type == "MIN":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MIN
    elif operator_type == "MAX":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.MAX
    elif operator_type == "SHR":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SHR
    elif operator_type == "SHL":
        assert npu_op.sub_op_type == vapi.NpuElementWiseOp.SHL
    # Compare reversed_operands
    assert npu_op.reversed_operands == True
    # Compare activation
    if operator_type == "SHR":
        assert npu_op.activation is None
    else:
        assert npu_op.activation.op_type == vapi.NpuActivationOp.NONE_OR_RELU
        assert npu_op.activation.min == 10
        assert npu_op.activation.max == 100
    # Compare rounding mode
    assert npu_op.rounding_mode == vapi.NpuRoundingMode.TFL


if __name__ == "__main__":
    tvm.testing.main()
