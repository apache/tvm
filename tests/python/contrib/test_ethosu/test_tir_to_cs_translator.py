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
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, ethosu_conv2d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_1, [1, 1, 3, 16], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_2, [16], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        placeholder_3 = T.match_buffer(placeholder, [1, 8, 8, 3], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_conv2d_1 = T.match_buffer(ethosu_conv2d, [1, 8, 8, 16], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 8, 8, 3, 8, 0, 8, T.load("uint8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 8, 8, 16, 8, 0, 8, T.load("uint8", ethosu_conv2d_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_4.data, 0), 0, 12, T.load("uint8", placeholder_5.data, 0), 0, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="uint8"))
# fmt: on


# fmt: off
"""A sample tir test case with multiple convolutions for translator"""
@tvm.script.ir_module
class MultiEthosUConv2D:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, ethosu_conv2d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_9 = T.match_buffer(placeholder_3, [1, 1, 32, 8], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_conv2d_1 = T.match_buffer(ethosu_conv2d, [1, 8, 8, 8], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_7 = T.match_buffer(placeholder_1, [1, 1, 3, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_6 = T.match_buffer(placeholder, [1, 8, 8, 3], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_8 = T.match_buffer(placeholder_2, [32], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_4, [8], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_conv2d_2 = T.allocate([1024], "uint8", "global")
        ethosu_conv2d_3 = T.allocate([2048], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 3, 4, 0, 8, T.load("uint8", placeholder_6.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 4, 8, 32, 4, 0, 8, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_7.data, 0), 0, 12, T.load("uint8", placeholder_8.data, 0), 0, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 32, 4, 0, 8, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "uint8", 4, 8, 8, 4, 0, 8, T.load("uint8", ethosu_conv2d_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_9.data, 0), 0, 12, T.load("uint8", placeholder_5.data, 0), 0, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 3, 4, 0, 8, T.load("uint8", placeholder_6.data, 96), 0, 0, 0, T.float32(0.5), 10, "NHWC", 24, 3, 1, "uint8", 4, 8, 32, 4, 0, 8, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 32, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_7.data, 0), 0, 12, T.load("uint8", placeholder_8.data, 0), 0, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="uint8"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 4, 8, 32, 4, 0, 8, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 32, 1, "uint8", 4, 8, 8, 4, 0, 8, T.load("uint8", ethosu_conv2d_1.data, 256), 0, 0, 0, T.float32(0.25), 14, "NHWC", 64, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_9.data, 0), 0, 12, T.load("uint8", placeholder_5.data, 0), 0, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="uint8"))
# fmt: on


# fmt: off
"""A sample tir test case with copy operations for translator"""
@tvm.script.ir_module
class MultiEthosUCopy:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, ethosu_conv2d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_3 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_conv2d_1 = T.match_buffer(ethosu_conv2d, [1, 16, 16, 8], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_2, [8], dtype="int32", elem_offset=0, align=128, offset_factor=1)
        placeholder_4 = T.match_buffer(placeholder_1, [8, 1, 1, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        placeholder_global = T.allocate([256], "uint8", "global")
        placeholder_d_global = T.allocate([8], "int32", "global")
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", placeholder_4.data, 0), 256,  T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("int32", placeholder_5.data, 0), 8, T.load("int32", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 8, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 0, 12, T.load("uint8", placeholder_d_global, 0), 0, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
# fmt: on


# fmt: off
"""A TIR test module of weight streaming"""
@tvm.script.ir_module
class WeightStreamOnly:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_conv2d: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, placeholder_7: T.handle, placeholder_8: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        buffer_4 = T.match_buffer(placeholder_5, [144], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = T.match_buffer(placeholder_4, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = T.match_buffer(placeholder_7, [144], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = T.match_buffer(placeholder_1, [144], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = T.match_buffer(placeholder_6, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_conv2d_1 = T.match_buffer(ethosu_conv2d, [1, 16, 16, 8], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = T.match_buffer(placeholder_3, [144], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = T.match_buffer(placeholder_2, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_9 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer = T.match_buffer(placeholder_8, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        placeholder_global = T.allocate([144], "uint8", "global")
        placeholder_d_global = T.allocate([20], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_5.data, 0), 144, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 144, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1.data, 0), 144, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 144, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4.data, 0), 144, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_6.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 144, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_7.data, 0), 144, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 144, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


# fmt: off
"""A TIR test module of weight streaming and direct reading"""
@tvm.script.ir_module
class MixedRead:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, ethosu_conv2d: T.handle, placeholder_2: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, placeholder_7: T.handle, placeholder_8: T.handle, placeholder_9: T.handle, placeholder_10: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        buffer_5 = T.match_buffer(placeholder_1, [592], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = T.match_buffer(placeholder_2, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = T.match_buffer(placeholder_7, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = T.match_buffer(placeholder_4, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_9 = T.match_buffer(placeholder_5, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_conv2d_1 = T.match_buffer(ethosu_conv2d, [1, 16, 16, 8], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer = T.match_buffer(placeholder_8, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = T.match_buffer(placeholder_10, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_11 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = T.match_buffer(placeholder_6, [20], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = T.match_buffer(placeholder_3, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_8 = T.match_buffer(placeholder_9, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_conv2d_2 = T.allocate([4096], "uint8", "global")
        placeholder_global = T.allocate([80], "uint8", "global")
        placeholder_d_global = T.allocate([20], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 32, 16, 0, 16, T.load("uint8", placeholder_11.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "uint8", 16, 16, 16, 16, 0, 16, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_5.data, 0), 592, 12, T.load("uint8", buffer_7.data, 0), 160, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_6.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 16, 16, 0, 16, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_9.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 16, 16, 0, 16, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 16, 16, 0, 16, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_8.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1.data, 0), 20, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "uint8", 16, 16, 16, 16, 0, 16, T.load("uint8", ethosu_conv2d_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "uint8", 16, 16, 2, 16, 0, 16, T.load("uint8", ethosu_conv2d_1.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 20, 0, 0, 0, 0, "CLIP", 0, 255, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_buffer_info_extraction():
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
            # Reference Outputs
            "constants": {
                "placeholder_4": 1,
                "placeholder_5": 2,
            },
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
            "constants": {
                "placeholder_5": 4,
                "placeholder_7": 1,
                "placeholder_8": 2,
                "placeholder_9": 3,
            },
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
                "ethosu_conv2d_2": ([1024], "uint8", tir_to_cs_translator.BufferType.scratch),
                "ethosu_conv2d_3": ([2048], "uint8", tir_to_cs_translator.BufferType.scratch),
            },
        },
    ]
    for test_case in test_cases:
        buffer_info = tir_to_cs_translator.extract_buffer_info(
            test_case["tir_module"], test_case["param_dict"]
        )
        for buffer_var, info in buffer_info.items():
            buffer_name = buffer_var.name
            if buffer_name in test_case["constants"].keys():
                assert (
                    info.values == test_case["param_dict"][test_case["constants"][buffer_name]]
                ).all()
                assert (
                    info.dtype == test_case["param_dict"][test_case["constants"][buffer_name]].dtype
                )
                info.btype == tir_to_cs_translator.BufferType.constant
            else:
                assert list(info.shape) == test_case["data_buffers"][buffer_name][0]
                assert info.dtype == test_case["data_buffers"][buffer_name][1]
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
            # Compare ifm upscaling
            assert npu_op.ifm_upscale == ref["ifm_upscale"]
            # Compare weight quantization parameters
            assert w_zero_point == ref["w_zero_point"]


# fmt: off
"""A ethosu_depthwise2d tir testcase for the translator"""
@tvm.script.ir_module
class SingleEthosuDepthwise2D:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, ethosu_depthwise2d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_1, [3, 3, 2, 1], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_2, [3, 10], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_3 = T.match_buffer(placeholder, [1, 8, 8, 3], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        ethosu_depthwise2d_1 = T.match_buffer(ethosu_depthwise2d, [1, 6, 7, 3], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        # body
        T.evaluate(T.call_extern("ethosu_depthwise2d", "int8", 8, 8, 3, 8, 0, 8, T.load("int8", placeholder_3.data, 0), 0, 0, 0, T.float32(0.6), 11, "NHWC", 24, 3, 1, "int8", 6, 7, 3, 6, 0, 7, T.load("int8", ethosu_depthwise2d_1.data, 0), 0, 0, 0, T.float32(0.26), 15, "NHWC", 21, 3, 1, 2, 3, 1, 1, 1, 1, T.load("int8", placeholder_4.data, 0), 18, 13, T.load("uint8", placeholder_5.data, 0), 30, 0, 0, 0, 0, "CLIP", 15, 105, "NONE", dtype="int8"))
    __tvm_meta__ = None
# fmt: on


def test_translate_ethosu_depthwise2d():
    def extract_ethosu_depthwise2d_extern_call(mod):
        # There should only be a single function
        assert len(mod.functions.items()) == 1
        primfunc = mod.functions.items()[0][1]

        ethosu_depthwise2d_calls = list()

        def populate_ethosu_depthwise2d_calls(stmt):
            if (
                isinstance(stmt, tvm.tir.Call)
                and stmt.op.name == "tir.call_extern"
                and stmt.args[0] == "ethosu_depthwise2d"
            ):
                ethosu_depthwise2d_calls.append(stmt)

        stmt_functor.post_order_visit(primfunc.body, populate_ethosu_depthwise2d_calls)
        return ethosu_depthwise2d_calls[0]

    depthwise2d_call = extract_ethosu_depthwise2d_extern_call(SingleEthosuDepthwise2D)
    npu_op, w_zero_point = tir_to_cs_translator.translate_ethosu_depthwise2d(depthwise2d_call)

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
                    "length": 8,
                },
            ],
        },
    ]

    for test_case in test_cases:
        ethosu_copy_calls = extract_ethosu_copy_extern_calls(test_case["tir_module"])
        for idx, ethosu_copy_call in enumerate(ethosu_copy_calls):
            npu_dma_op = tir_to_cs_translator.translate_ethosu_tir_extern_call(ethosu_copy_call)
            assert npu_dma_op.src.address.buffer_var.name == test_case["ref"][idx]["src"]
            assert npu_dma_op.dest.address.buffer_var.name == test_case["ref"][idx]["dest"]
            assert npu_dma_op.src.length == test_case["ref"][idx]["length"]
            assert npu_dma_op.dest.length == test_case["ref"][idx]["length"]


def test_assign_addresses():
    test_cases = [
        {
            # Stimulus
            "tir_module": WeightStreamOnly,
            "param_dict": {
                2: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [144], "uint8"),
                3: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                4: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [144], "uint8"),
                5: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                6: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [144], "uint8"),
                7: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                8: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [144], "uint8"),
                9: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
            },
        },
        {
            # Stimulus
            "tir_module": MixedRead,
            "param_dict": {
                1: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [592], "uint8"),
                3: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [160], "uint8"),
                4: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"),
                5: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                6: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"),
                7: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                8: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"),
                9: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
                10: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [80], "uint8"),
                11: np.random.randint(np.iinfo("uint8").min, np.iinfo("uint8").max, [20], "uint8"),
            },
        },
    ]

    def extract_extern_calls(mod):
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
        if buffer_type == tir_to_cs_translator.BufferType.constant:
            ref = buffer_info[buffer_var].values
            assert (constant_tensor[address : address + length] == ref).all()
            # Every buffer is adjusted to align to 16 bytes
            length = util.round_up(length, 16)
            # Mark these constants are read at least once
            constant_tensor_read_mask[address : address + length] = np.ones(length, dtype="uint8")
        elif buffer_type == tir_to_cs_translator.BufferType.scratch:
            shape = list(buffer_info[buffer_var].shape)
            assert length == np.prod(shape)
            assert address < scratch_size
            # Every buffer is adjusted to align to 16 bytes
            length = util.round_up(length, 16)
            assert address + length <= scratch_size
            # The scratch area should not be used by anyother buffer
            assert not scratch_allocation_mask[address : address + length].any()
            # The scratch area is marked as used
            scratch_allocation_mask[address : address + length] = np.ones(length, dtype="uint8")
        elif buffer_type == tir_to_cs_translator.BufferType.input:
            assert address == 0
        else:
            assert buffer_type == tir_to_cs_translator.BufferType.output
            assert address == 0

    def verify(npu_ops):
        """This wrapper verifies the allocated addresses matches with original tir buffers"""
        checked_buffers = set()

        def check_buffer(address, region, length, buffer_var):
            if buffer_var not in checked_buffers:
                _check_buffer(address, region, length, buffer_var)
                checked_buffers.add(buffer_var)

        for npu_op in npu_ops:
            if isinstance(npu_op, vapi.NpuDmaOperation):
                src_tir_buffer_var = npu_op_tir_buffers[npu_op][0].buffer_var
                check_buffer(
                    npu_op.src.address, npu_op.src.region, npu_op.src.length, src_tir_buffer_var
                )
                dest_tir_load = npu_op_tir_buffers[npu_op][1].buffer_var
                check_buffer(
                    npu_op.dest.address,
                    npu_op.dest.region,
                    npu_op.dest.length,
                    dest_tir_load,
                )
            elif issubclass(type(npu_op), vapi.NpuBlockOperation):
                ifm_tir_buffer_var = npu_op_tir_buffers[npu_op][0].buffer_var
                ifm_length = (
                    npu_op.ifm.shape.height * npu_op.ifm.shape.width * npu_op.ifm.shape.depth
                )
                check_buffer(
                    npu_op.ifm.tiles.addresses[0],
                    npu_op.ifm.region,
                    ifm_length,
                    ifm_tir_buffer_var,
                )
                ofm_tir_buffer_var = npu_op_tir_buffers[npu_op][1].buffer_var
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
                        weight.address.buffer_var,
                    )
                for idx, bias in enumerate(npu_op_tir_buffers[npu_op][3]):
                    assert isinstance(bias, vapi.NpuAddressRange)
                    check_buffer(
                        npu_op.biases[idx].address,
                        npu_op.biases[idx].region,
                        npu_op.biases[idx].length,
                        bias.address.buffer_var,
                    )

    for test_case in test_cases:
        buffer_info = tir_to_cs_translator.extract_buffer_info(
            test_case["tir_module"], test_case["param_dict"]
        )
        extern_calls = extract_extern_calls(test_case["tir_module"])
        _npu_ops = list()
        for extern_call in extern_calls:
            _npu_ops.append(tir_to_cs_translator.translate_ethosu_tir_extern_call(extern_call))
        npu_op_tir_buffers = collect_tir_buffer_info(_npu_ops)
        _npu_ops, constant_tensor, scratch_size = tir_to_cs_translator.assign_addresses(
            buffer_info, _npu_ops
        )
        scratch_allocation_mask = np.zeros(scratch_size, dtype="uint8")
        constant_tensor_read_mask = np.zeros(constant_tensor.size, dtype="uint8")
        verify(_npu_ops)
        # This will be only 1 if all allocated scratch is used.
        assert np.prod(scratch_allocation_mask) == 1
        # This will be only 1 if all constant tensors is read at least once.
        assert np.prod(constant_tensor_read_mask) == 1


if __name__ == "__main__":
    pytest.main([__file__])
