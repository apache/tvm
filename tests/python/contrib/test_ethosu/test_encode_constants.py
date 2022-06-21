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
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import OperatorCompute
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants
from tvm.relay.backend.contrib.ethosu import tir_to_cs_translator

from .infra import make_ethosu_conv2d, make_ethosu_binary_elementwise


# fmt: off
@tvm.script.ir_module
class WeightStreamOnlyU55:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer1 = T.buffer_decl([128], "uint8")
        buffer2 = T.buffer_decl([32], "uint8")
        buffer3 = T.buffer_decl([112], "uint8")
        buffer4 = T.buffer_decl([32], "uint8")
        buffer5 = T.buffer_decl([112], "uint8")
        buffer6 = T.buffer_decl([32], "uint8")
        buffer7 = T.buffer_decl([112], "uint8")
        buffer8 = T.buffer_decl([32], "uint8")
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], "int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], "int8", data=ethosu_write.data)
        # body
        p1 = T.allocate([128], "uint8", "global", annotations={"disable_lower_builtin":True})
        p2 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        p3 = T.allocate([112], "uint8", "global", annotations={"disable_lower_builtin":True})
        p4 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        buffer9 = T.buffer_decl([112], "uint8", data=p1.data)
        T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 128, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 32, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 112, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 32, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 128, T.int8(-1), T.int8(-1), 12, p2[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 112, buffer9[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 32, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, T.int8(-1), T.int8(-1), 12, p4[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 112, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 32, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, buffer9[0], 112, T.int8(-1), T.int8(-1), 12, p2[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 112, T.int8(-1), T.int8(-1), 12, p4[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class WeightStreamOnlyU65:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        # buffer definition
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], dtype="int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", data=ethosu_write.data)
        buffer_encoded_1 = T.buffer_decl([160], dtype="uint8")
        buffer_encoded_1_1 = T.buffer_decl([32], dtype="uint8")
        buffer_encoded_2_1 = T.buffer_decl([160], dtype="uint8")
        buffer_encoded_3_1 = T.buffer_decl([32], dtype="uint8")
        buffer_encoded_4_1 = T.buffer_decl([176], dtype="uint8")
        buffer_encoded_5_1 = T.buffer_decl([32], dtype="uint8")
        buffer_encoded_6_1 = T.buffer_decl([160], dtype="uint8")
        buffer_encoded_7_1 = T.buffer_decl([32], dtype="uint8")
        # body
        placeholder_global = T.allocate([176], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global_2 = T.allocate([160], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global_2 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global_1 = T.buffer_decl([160], dtype="uint8", data=placeholder_global.data)
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1[0], 160, placeholder_global_1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1_1[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_2_1[0], 160, placeholder_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_3_1[0], 32, placeholder_d_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_1[0], 80, placeholder_global_1[80], 80, 12, placeholder_d_global[0], 16, placeholder_d_global[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_4_1[0], 176, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_5_1[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 80, placeholder_global_2[80], 80, 12, placeholder_d_global_2[0], 16, placeholder_d_global_2[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_6_1[0], 160, placeholder_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_7_1[0], 32, placeholder_d_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 96, placeholder_global[96], 80, 12, placeholder_d_global[0], 16, placeholder_d_global[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 80, placeholder_global_2[80], 80, 12, placeholder_d_global_2[0], 16, placeholder_d_global_2[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "accelerator, reference_mod, reference_const_sizes",
    [
        (
            "ethos-u55-128",
            WeightStreamOnlyU55,
            [128, 32, 112, 32, 112, 32, 112, 32],
        ),
        (
            "ethos-u65-512",
            WeightStreamOnlyU65,
            [160, 32, 160, 32, 176, 32, 160, 32],
        ),
    ],
)
def test_weight_stream_only(accelerator, reference_mod, reference_const_sizes):
    def _planner(cached_func, const_dict, sch):
        weights = cached_func.inputs[1]
        bias = cached_func.inputs[2]
        out = cached_func.outputs[0]
        conv_compute = OperatorCompute.from_output(out)
        co = conv_compute.split(sch, 3, 2)
        cache_weights = sch.cache_read(weights, "global", [conv_compute.op])
        cache_bias = sch.cache_read(bias, "global", [conv_compute.op])
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

    config = {
        "accelerator_config": accelerator,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        func = _get_func()
        mod, consts = _lower_to_tir(func, cascader=_planner)
        script = mod.script(show_meta=True)
        test_mod = tvm.script.from_source(script)
        tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

        test_const_size = [value.size for value in list(consts.values())]
        assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class RereadWeightsU55:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer1 = T.buffer_decl([304], "uint8")
        buffer2 = T.buffer_decl([80], "uint8")
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], "int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], "int8", data=ethosu_write.data)
        # body
        p1 = T.allocate([304], "uint8", "global", annotations={"disable_lower_builtin":True})
        p2 = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        p3 = T.allocate([304], "uint8", "global", annotations={"disable_lower_builtin":True})
        p4 = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 304, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 304, p3[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 80, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 304, T.int8(-1), T.int8(-1), 12, p2[0], 80, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[256], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[64], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p3[0], 304, T.int8(-1), T.int8(-1), 12, p4[0], 80, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class RereadWeightsU65:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        # buffer definition
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], dtype="int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", data=ethosu_write.data)
        placeholder_encoded_1 = T.buffer_decl([368], "uint8")
        placeholder_encoded_1_2 = T.buffer_decl([96], "uint8")
        # body
        placeholder_global = T.allocate([368], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([96], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global_1 = T.allocate([368], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global_1 = T.allocate([96], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_1[0], 368, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_1_2[0], 96, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_1[0], 368, placeholder_global_1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", placeholder_encoded_1_2[0], 96, placeholder_d_global_1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 192, placeholder_global[192], 176, 12, placeholder_d_global[0], 48, placeholder_d_global[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 8, 32, 16, 0, 8, placeholder[256], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 8, 8, 16, 0, 8, ethosu_write[64], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_1[0], 192, placeholder_global_1[192], 176, 12, placeholder_d_global_1[0], 48, placeholder_d_global_1[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))

    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "accelerator, reference_mod, reference_const_sizes",
    [
        (
            "ethos-u55-128",
            RereadWeightsU55,
            [304, 80],
        ),
        (
            "ethos-u65-512",
            RereadWeightsU65,
            [368, 96],
        ),
    ],
)
def test_re_read_weights(accelerator, reference_mod, reference_const_sizes):
    def _cascader(cached_func, const_dict, sch):
        weights = cached_func.inputs[1]
        bias = cached_func.inputs[2]
        out = cached_func.outputs[0]
        conv_compute = OperatorCompute.from_output(out)
        co = conv_compute.split(sch, 2, 8)
        cache_weights = sch.cache_read(weights, "global", [conv_compute.op])
        cache_bias = sch.cache_read(bias, "global", [conv_compute.op])
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

    config = {
        "accelerator_config": accelerator,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        func = _get_func()
        mod, consts = _lower_to_tir(func, cascader=_cascader)
        script = mod.script(show_meta=True)
        test_mod = tvm.script.from_source(script)
        tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

        test_const_size = [value.size for value in list(consts.values())]
        assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class DirectReadOnlyU55:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = T.buffer_decl([592], "uint8")
        buffer_1 = T.buffer_decl([160], "uint8")
        buffer_2 = T.buffer_decl([160], "uint8")
        buffer_3 = T.buffer_decl([80], "uint8")
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], "int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], "int8", data=ethosu_write.data)
        # body
        ethosu_write_1 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, buffer[0], 592, T.int8(-1), T.int8(-1), 12, buffer_1[0], 160, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 8, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, buffer_2[0], 160, T.int8(-1), T.int8(-1), 12, buffer_3[0], 80, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class DirectReadOnlyU65:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        # buffer definition
        placeholder_encoded = T.buffer_decl([608], dtype="uint8")
        placeholder_encoded_1 = T.buffer_decl([160], dtype="uint8")
        placeholder_encoded_2 = T.buffer_decl([208], dtype="uint8")
        placeholder_encoded_3 = T.buffer_decl([96], dtype="uint8")
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], dtype="int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", data=ethosu_write.data)
        # body
        ethosu_write_2 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, placeholder_encoded[0], 304, placeholder_encoded[304], 304, 12, placeholder_encoded_1[0], 80, placeholder_encoded_1[80], 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 8, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_encoded_2[0], 112, placeholder_encoded_2[112], 96, 12, placeholder_encoded_3[0], 48, placeholder_encoded_3[48], 48, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "accelerator, reference_mod, reference_const_sizes",
    [
        (
            "ethos-u55-128",
            DirectReadOnlyU55,
            [592, 160, 160, 80],
        ),
        (
            "ethos-u65-512",
            DirectReadOnlyU65,
            [608, 160, 208, 96],
        ),
    ],
)
def test_direct_read_only(accelerator, reference_mod, reference_const_sizes):
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

    config = {
        "accelerator_config": accelerator,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        func = _get_func()
        mod, consts = _lower_to_tir(func)

        script = mod.script(show_meta=True)
        test_mod = tvm.script.from_source(script)
        tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

        test_const_size = [value.size for value in list(consts.values())]
        assert reference_const_sizes == test_const_size


# fmt: off
@tvm.script.ir_module
class MixedReadU55:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer1 = T.buffer_decl([80], "uint8")
        buffer2 = T.buffer_decl([32], "uint8")
        buffer3 = T.buffer_decl([80], "uint8")
        buffer4 = T.buffer_decl([32], "uint8")
        buffer5 = T.buffer_decl([80], "uint8")
        buffer6 = T.buffer_decl([32], "uint8")
        buffer7 = T.buffer_decl([80], "uint8")
        buffer8 = T.buffer_decl([32], "uint8")
        buffer9 = T.buffer_decl([592], "uint8")
        buffer10 = T.buffer_decl([160], "uint8")
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], "int8", data=placeholder.data)
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], "int8", data=ethosu_write.data)
        # body
        p1 = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        p2 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        p3 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        p4 = T.allocate([80], "uint8", "global", annotations={"disable_lower_builtin":True})
        p5 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 80, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer2[0], 32, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, p3[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, buffer9[0], 592, T.int8(-1), T.int8(-1), 12, buffer10[0], 160, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 80, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer4[0], 32, p5[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, p3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 80, T.int8(-1), T.int8(-1), 12, p2[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer5[0], 80, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer6[0], 32, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, p3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p4[0], 80, T.int8(-1), T.int8(-1), 12, p5[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer7[0], 80, p4[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer8[0], 32, p5[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, p3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p1[0], 80, T.int8(-1), T.int8(-1), 12, p2[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, p3[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, p4[0], 80, T.int8(-1), T.int8(-1), 12, p5[0], 32, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None


@tvm.script.ir_module
class MixedReadU65:
    @T.prim_func
    def main(placeholder: T.Buffer[(8192,), "int8"], ethosu_write: T.Buffer[(2048,), "int8"]) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        T.preflattened_buffer(ethosu_write, [1, 16, 16, 8], "int8", data=ethosu_write.data)
        T.preflattened_buffer(placeholder, [1, 16, 16, 32], "int8", data=placeholder.data)
        # buffer definition
        buffer_encoded_1 = T.buffer_decl([96], dtype="uint8")
        buffer_encoded_1_2 = T.buffer_decl([32], dtype="uint8")
        placeholder_encoded_1 = T.buffer_decl([608], dtype="uint8")
        placeholder_encoded_1_2 = T.buffer_decl([160], dtype="uint8")
        buffer_encoded_2_1 = T.buffer_decl([96], dtype="uint8")
        buffer_encoded_3_1 = T.buffer_decl([32], dtype="uint8")
        buffer_encoded_4_1 = T.buffer_decl([96], dtype="uint8")
        buffer_encoded_5_1 = T.buffer_decl([32], dtype="uint8")
        buffer_encoded_6_1 = T.buffer_decl([96], dtype="uint8")
        buffer_encoded_7_1 = T.buffer_decl([32], dtype="uint8")
        placeholder_global = T.allocate([96], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        ethosu_write_2 = T.allocate([4096], "int8", "global", annotations={"disable_lower_builtin":True})
        placeholder_global_2 = T.allocate([96], "uint8", "global", annotations={"disable_lower_builtin":True})
        placeholder_d_global_2 = T.allocate([32], "uint8", "global", annotations={"disable_lower_builtin":True})
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1[0], 96, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_1_2[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 256, 16, 1, 1, 1, 1, 1, 1, 1, placeholder_encoded_1[0], 304, placeholder_encoded_1[304], 304, 12, placeholder_encoded_1_2[0], 80, placeholder_encoded_1_2[80], 80, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_2_1[0], 96, placeholder_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_3_1[0], 32, placeholder_d_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 48, placeholder_global[48], 48, 12, placeholder_d_global[0], 16, placeholder_d_global[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_4_1[0], 96, placeholder_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_5_1[0], 32, placeholder_d_global[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[2], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 48, placeholder_global_2[48], 48, 12, placeholder_d_global_2[0], 16, placeholder_d_global_2[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_6_1[0], 96, placeholder_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer_encoded_7_1[0], 32, placeholder_d_global_2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[4], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global[0], 48, placeholder_global[48], 48, 12, placeholder_d_global[0], 16, placeholder_d_global[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 16, 16, 16, 16, 0, 16, ethosu_write_2[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 256, 16, 1, "int8", 16, 16, 2, 16, 0, 16, ethosu_write[6], 0, 0, 0, T.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, placeholder_global_2[0], 48, placeholder_global_2[48], 48, 12, placeholder_d_global_2[0], 16, placeholder_d_global_2[16], 16, 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


@pytest.mark.parametrize(
    "accelerator, reference_mod, reference_const_sizes",
    [
        (
            "ethos-u55-128",
            MixedReadU55,
            [592, 160, 80, 32, 80, 32, 80, 32, 80, 32],
        ),
        (
            "ethos-u65-512",
            MixedReadU65,
            [608, 160, 96, 32, 96, 32, 96, 32, 96, 32],
        ),
    ],
)
def test_mixed_read(accelerator, reference_mod, reference_const_sizes):
    def _planner(cached_func, const_dict, sch):
        weight = cached_func.inputs[4]
        scale_bias = cached_func.inputs[5]
        out = cached_func.outputs[0]
        conv_compute = OperatorCompute.from_output(out)
        co = conv_compute.split(sch, 3, 2)
        cache_weight = sch.cache_read(weight, "global", [conv_compute.op])
        cache_scale_bias = sch.cache_read(scale_bias, "global", [conv_compute.op])
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

    config = {
        "accelerator_config": accelerator,
    }
    with tvm.transform.PassContext(config={"relay.ext.ethos-u.options": config}):
        func = _get_func()
        mod, consts = _lower_to_tir(func, cascader=_planner)

        script = mod.script(show_meta=True)
        test_mod = tvm.script.from_source(script)
        tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)

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

    tir_mod, params = _lower_to_tir(get_graph(), copy_constants())

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
