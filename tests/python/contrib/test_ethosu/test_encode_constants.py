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
import numpy as np

import tvm
from tvm import tir
from tvm import script
from tvm import relay
from tvm.script import ty
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import Convolution2DCompute
import pytest

from infra import make_ethosu_conv2d


# fmt: off
@tvm.script.tir
class WeightStreamOnly:
    def main(placeholder: ty.handle, ethosu_write: ty.handle, placeholder_1: ty.handle, placeholder_2: ty.handle, placeholder_3: ty.handle, placeholder_4: ty.handle, placeholder_5: ty.handle, placeholder_6: ty.handle, placeholder_7: ty.handle, placeholder_8: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = tir.match_buffer(placeholder_7, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = tir.match_buffer(placeholder_4, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = tir.match_buffer(placeholder_2, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = tir.match_buffer(placeholder_8, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = tir.match_buffer(placeholder_5, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_9 = tir.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = tir.match_buffer(placeholder_3, [112], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = tir.match_buffer(placeholder_1, [128], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = tir.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = tir.match_buffer(placeholder_6, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        placeholder_global = tir.allocate([128], "uint8", "global")
        placeholder_d_global = tir.allocate([32], "uint8", "global")
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_6.data, 0), 128, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_2.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_9.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 128, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_5.data, 0), 112, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_1.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_9.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 2), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 112, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_4.data, 0), 112, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_7.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_9.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 4), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 112, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer.data, 0), 112, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_3.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_9.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 6), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 112, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_weight_stream_only():
    def _planner(te_graph, const_dict, sch):
        weights = te_graph.inputs[1]
        bias = te_graph.inputs[2]
        out = te_graph.outputs[0]
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
    script = tvm.script.asscript(mod, True)
    test_mod = tvm.script.from_source(script)
    reference_mod = WeightStreamOnly()
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = {2: 128, 3: 32, 4: 112, 5: 32, 6: 112, 7: 32, 8: 112, 9: 32}
    test_const_sizes = {}
    for key, value in consts.items():
        test_const_sizes[key] = len(value)

    assert reference_const_sizes == test_const_sizes


# fmt: off
@tvm.script.tir
class DirectReadOnly:
    def main(placeholder: ty.handle, placeholder_1: ty.handle, placeholder_2: ty.handle, placeholder_3: ty.handle, placeholder_4: ty.handle, ethosu_write: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = tir.match_buffer(placeholder_3, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = tir.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        placeholder_5 = tir.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = tir.match_buffer(placeholder_1, [592], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = tir.match_buffer(placeholder_2, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = tir.match_buffer(placeholder_4, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_write_2 = tir.allocate([4096], "int8", "global")
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_5.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", buffer_1.data, 0), 592, 12, tir.load("uint8", buffer_2.data, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 8, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", buffer.data, 0), 160, 12, tir.load("uint8", buffer_3.data, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
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

    script = tvm.script.asscript(mod, True)
    test_mod = tvm.script.from_source(script)
    reference_mod = DirectReadOnly()
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = {1: 592, 2: 160, 3: 160, 4: 80}
    test_const_sizes = {}
    for key, value in consts.items():
        test_const_sizes[key] = len(value)

    assert reference_const_sizes == test_const_sizes


# fmt: off
@tvm.script.tir
class MixedRead:
    def main(placeholder: ty.handle, placeholder_1: ty.handle, placeholder_2: ty.handle, ethosu_write: ty.handle, placeholder_3: ty.handle, placeholder_4: ty.handle, placeholder_5: ty.handle, placeholder_6: ty.handle, placeholder_7: ty.handle, placeholder_8: ty.handle, placeholder_9: ty.handle, placeholder_10: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = tir.match_buffer(placeholder_7, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = tir.match_buffer(placeholder_5, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = tir.match_buffer(placeholder_3, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = tir.match_buffer(placeholder_4, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = tir.match_buffer(placeholder_9, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = tir.match_buffer(placeholder_6, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_11 = tir.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = tir.match_buffer(placeholder_1, [592], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = tir.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = tir.match_buffer(placeholder_2, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_8 = tir.match_buffer(placeholder_8, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_9 = tir.match_buffer(placeholder_10, [32], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        ethosu_write_2 = tir.allocate([4096], "int8", "global")
        placeholder_global = tir.allocate([80], "uint8", "global")
        placeholder_d_global = tir.allocate([32], "uint8", "global")
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_11.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", buffer_6.data, 0), 592, 12, tir.load("uint8", buffer_7.data, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_2.data, 0), 80, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_3.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 80, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_1.data, 0), 80, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_5.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 2), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 80, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer.data, 0), 80, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_8.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 4), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 80, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_4.data, 0), 80, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_9.data, 0), 32, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, tir.load("int8", ethosu_write_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 6), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 80, 12, tir.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_mixed_read():
    def _planner(te_graph, const_dict, sch):
        weight = te_graph.inputs[4]
        scale_bias = te_graph.inputs[5]
        out = te_graph.outputs[0]
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

    script = tvm.script.asscript(mod, True)
    test_mod = tvm.script.from_source(script)
    reference_mod = MixedRead()
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
