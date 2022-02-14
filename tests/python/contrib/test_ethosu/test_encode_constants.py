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
import numpy as np

pytest.importorskip("ethosu.vela")
import tvm
from tvm import relay
from tvm.script import tir as T
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import Convolution2DCompute
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator

from .infra import make_ethosu_conv2d, make_ethosu_binary_elementwise


# fmt: off
@tvm.script.ir_module
class WeightStreamOnly:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 16, 16, 32), "int8"], ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        buffer_4 = T.buffer_var("uint8", "")
        buffer_5 = T.buffer_var("uint8", "")
        buffer_6 = T.buffer_var("uint8", "")
        buffer_7 = T.buffer_var("uint8", "")
        # body
        placeholder_global = T.allocate([128], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer, 0), 128, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 128, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_5, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_6, 0), 112, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_7, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 112, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
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

    reference_const_sizes = [128, 32, 112, 32, 112, 32, 112, 32]
    test_const_size = [value.size for value in list(consts.values())]
    assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class RereadWeights:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 16, 16, 32), "int8"], ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        # body
        placeholder_global = T.allocate([304], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer, 0), 304, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1, 0), 80, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 1, 8, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 304, 12, T.load("uint8", placeholder_d_global, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer, 0), 304, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_1, 0), 80, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, T.load("int8", placeholder.data, 256), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, T.load("int8", ethosu_write.data, 64), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 1, 8, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 304, 12, T.load("uint8", placeholder_d_global, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_re_read_weights():
    def _cascader(cached_func, const_dict, sch):
        weights = cached_func.inputs[1]
        bias = cached_func.inputs[2]
        out = cached_func.outputs[0]
        conv_compute = Convolution2DCompute.from_output(out)
        co = conv_compute.split(sch, 2, 8)
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
    mod, consts = lower_to_tir(func, cascader=_cascader)
    script = mod.script(show_meta=True)
    test_mod = tvm.script.from_source(script)
    reference_mod = RereadWeights
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

    reference_const_sizes = [304, 80]
    test_const_size = [value.size for value in list(consts.values())]
    assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class DirectReadOnly:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 16, 16, 32), "int8"], ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        # body
        ethosu_write_1 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 592, 12, T.load("uint8", buffer_1, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 8, 16, 0, 16, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer_2, 0), 160, 12, T.load("uint8", buffer_3, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
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

    reference_const_sizes = [592, 160, 160, 80]
    test_const_size = [value.size for value in list(consts.values())]
    assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class MixedRead:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 16, 16, 32), "int8"], ethosu_write: T.Buffer[(1, 16, 16, 8), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_var("uint8", "")
        buffer_1 = T.buffer_var("uint8", "")
        buffer_2 = T.buffer_var("uint8", "")
        buffer_3 = T.buffer_var("uint8", "")
        buffer_4 = T.buffer_var("uint8", "")
        buffer_5 = T.buffer_var("uint8", "")
        buffer_6 = T.buffer_var("uint8", "")
        buffer_7 = T.buffer_var("uint8", "")
        buffer_8 = T.buffer_var("uint8", "")
        buffer_9 = T.buffer_var("uint8", "")
        # body
        ethosu_write_1 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, T.load("int8", placeholder.data, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", buffer, 0), 592, 12, T.load("uint8", buffer_1, 0), 160, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_2, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_3, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 0), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_4, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_5, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 2), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_6, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_7, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 4), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_8, 0), 80, T.load("uint8", placeholder_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", T.load("uint8", buffer_9, 0), 32, T.load("uint8", placeholder_d_global, 0), dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, T.load("int8", ethosu_write_1, 0), 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, T.load("int8", ethosu_write.data, 6), 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, T.load("uint8", placeholder_global, 0), 80, 12, T.load("uint8", placeholder_d_global, 0), 32, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
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

    reference_const_sizes = [
        592,
        160,
        80,
        32,
        80,
        32,
        80,
        32,
        80,
        32,
    ]
    test_const_size = [value.size for value in list(consts.values())]
    assert reference_const_sizes == test_const_size


def test_constant_as_input():
    """Test to check that constants specified as inputs aren't
    interpreted as an encoded constant."""

    def get_graph():
        dtype = "uint8"
        ifm = relay.var("ifm", shape=(1, 16, 16, 32), dtype=dtype)
        conv1 = make_ethosu_conv2d(
            ifm,
            32,
            16,
            (1, 1),
            (0, 0),
            (1, 1),
            (1, 1),
        )
        scalar = relay.const(np.ones((1, 1, 1, 1), dtype=dtype), dtype=dtype)
        add1 = make_ethosu_binary_elementwise(
            conv1, scalar, ifm_channels=32, ifm2_channels=1, operator_type="ADD", ofm_dtype=dtype
        )
        func = relay.Function(relay.analysis.free_vars(add1), add1)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    tir_mod, params = lower_to_tir(get_graph(), copy_constants())

    # Check tile address for the scalar constant input hasn't been
    # overwritten.
    extern_calls = tir_mod["main"].body.body.body.body.body
    binary_elementwise = extern_calls[-1].value
    args = binary_elementwise.args

    reason = "Tile address overwritten"
    assert args[26] == 0, reason
    assert args[27] == 0, reason
    assert args[28] == 0, reason

    # More generally, check compiles successfully to make sure
    # nothing else was overrwritten.
    # With Target Hooks the TIR module needs a target attached
    # and lowered via make unpacked API.
    tir_mod["main"] = tir_mod["main"].with_attr("target", tvm.target.Target("ethos-u"))
    tir_mod = tvm.tir.transform.MakeUnpackedAPI()(tir_mod)
    tir_to_cs_translator.translate(tir_mod, params)


if __name__ == "__main__":
    pytest.main([__file__])
