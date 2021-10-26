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
from tvm import relay
from tvm.script import tir as T
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import Convolution2DCompute

from .infra import make_ethosu_conv2d


# fmt: off
@tvm.script.ir_module
class WeightStreamOnly:
    @T.prim_func
    def main(placeholder: T.handle, ethosu_write: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, placeholder_7: T.handle, placeholder_8: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.match_buffer(placeholder_7, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = T.match_buffer(placeholder_4, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = T.match_buffer(placeholder_2, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = T.match_buffer(placeholder_8, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = T.match_buffer(placeholder_5, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_9 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = T.match_buffer(placeholder_3, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = T.match_buffer(placeholder_1, [128], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = T.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = T.match_buffer(placeholder_6, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        placeholder_global = T.allocate([128], "uint8", "global")
        placeholder_d_global = T.allocate([32], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_6.data, 0), 128, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 128, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_5.data, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4.data, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_7.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer.data, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_9.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_weight_stream_only():
    def _planner(cached_func, const_dict, sch):
        weights = cached_func.inputs[1]
        bias = cached_func.inputs[2]
        out = cached_func.outputs[0]
        conv_compute = Convolution2DCompute.from_output(out)
        co = conv_compute.split(sch, 3, 2)
        cache_weights = sch.cache_read(weights, "global", [conv_compute.conv2d])
        cache_bias = sch.cache_read(bias, "global", [conv_compute.conv2d])
        sch[cache_weights].compute_at(sch[out], co)
        sch[cache_bias].compute_at(sch[out], co)

    def _get_func():
        ifm = relay.var("ifm", shape=(1, 16, 16, 32), dtype="int8")
        conv = make_ethosu_conv2d(
            ifm,
            32,
            8,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        func = relay.Function(relay.analysis.free_vars(conv), conv)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func()
    mod, consts = lower_to_tir(func, cascader=_planner)
    script = mod.script(show_meta=True)
    test_mod = tvm.script.from_source(script)
    reference_mod = WeightStreamOnly
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = {2: 128, 3: 32, 4: 112, 5: 32, 6: 112, 7: 32, 8: 112, 9: 32}
    test_const_sizes = {}
    for key, value in consts.items():
        test_const_sizes[key] = len(value)

    assert reference_const_sizes == test_const_sizes


# fmt: off
@tvm.script.ir_module
class DirectReadOnly:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, ethosu_write: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.match_buffer(placeholder_3, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = T.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = T.match_buffer(placeholder_1, [592], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = T.match_buffer(placeholder_2, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = T.match_buffer(placeholder_4, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_write_2 = T.allocate([4096], "int8", "global")
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_5.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_1.data, 0), 592, 12, T.load("uint8", buffer_2.data, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 8, 16, 0, 16, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer.data, 0), 160, 12, T.load("uint8", buffer_3.data, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_direct_read_only():
    def _get_func():
        ifm = relay.var("ifm", shape=(1, 16, 16, 32), dtype="int8")
        conv1 = make_ethosu_conv2d(
            ifm,
            32,
            16,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        conv2 = make_ethosu_conv2d(
            conv1,
            16,
            8,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        func = relay.Function(relay.analysis.free_vars(conv2), conv2)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func()
    mod, consts = lower_to_tir(func)

    script = mod.script(show_meta=True)
    test_mod = tvm.script.from_source(script)
    reference_mod = DirectReadOnly
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = {1: 592, 2: 160, 3: 160, 4: 80}
    test_const_sizes = {}
    for key, value in consts.items():
        test_const_sizes[key] = len(value)

    assert reference_const_sizes == test_const_sizes


# fmt: off
@tvm.script.ir_module
class MixedRead:
    @T.prim_func
    def main(placeholder: T.handle, placeholder_1: T.handle, placeholder_2: T.handle, ethosu_write: T.handle, placeholder_3: T.handle, placeholder_4: T.handle, placeholder_5: T.handle, placeholder_6: T.handle, placeholder_7: T.handle, placeholder_8: T.handle, placeholder_9: T.handle, placeholder_10: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.match_buffer(placeholder_7, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = T.match_buffer(placeholder_5, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = T.match_buffer(placeholder_3, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = T.match_buffer(placeholder_4, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = T.match_buffer(placeholder_9, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = T.match_buffer(placeholder_6, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_11 = T.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = T.match_buffer(placeholder_1, [592], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = T.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = T.match_buffer(placeholder_2, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_8 = T.match_buffer(placeholder_8, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_9 = T.match_buffer(placeholder_10, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_write_2 = T.allocate([4096], "int8", "global")
        placeholder_global = T.allocate([80], "uint8", "global")
        placeholder_d_global = T.allocate([32], "uint8", "global")
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder_11.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_6.data, 0), 592, 12, T.load("uint8", buffer_7.data, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_5.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_8.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4.data, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_9.data, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_2, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write_1.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_mixed_read():
    def _planner(cached_func, const_dict, sch):
        weight = cached_func.inputs[4]
        scale_bias = cached_func.inputs[5]
        out = cached_func.outputs[0]
        conv_compute = Convolution2DCompute.from_output(out)
        co = conv_compute.split(sch, 3, 2)
        cache_weight = sch.cache_read(weight, "global", [conv_compute.conv2d])
        cache_scale_bias = sch.cache_read(scale_bias, "global", [conv_compute.conv2d])
        sch[cache_weight].compute_at(sch[out], co)
        sch[cache_scale_bias].compute_at(sch[out], co)

    def _get_func():
        ifm = relay.var("ifm", shape=(1, 16, 16, 32), dtype="int8")
        conv1 = make_ethosu_conv2d(
            ifm,
            32,
            16,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        conv2 = make_ethosu_conv2d(
            conv1,
            16,
            8,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        func = relay.Function(relay.analysis.free_vars(conv2), conv2)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func()
    mod, consts = lower_to_tir(func, cascader=_planner)

    script = mod.script(show_meta=True)
    test_mod = tvm.script.from_source(script)
    reference_mod = MixedRead
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = {
        1: 592,
        2: 160,
        4: 80,
        5: 32,
        6: 80,
        7: 32,
        8: 80,
        9: 32,
        10: 80,
        11: 32,
    }
    test_const_sizes = {}
    for key, value in consts.items():
        test_const_sizes[key] = len(value)

    assert reference_const_sizes == test_const_sizes


if __name__ == "__main__":
    pytest.main([__file__])
