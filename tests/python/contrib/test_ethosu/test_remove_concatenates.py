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
import tvm.script
from tvm.script import tir
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from .infra import make_ethosu_conv2d


# fmt: off
@tvm.script.ir_module
class ReferenceModule:
    @tir.prim_func
    def main(placeholder: tir.handle, placeholder_1: tir.handle, placeholder_2: tir.handle, placeholder_3: tir.handle, placeholder_4: tir.handle, placeholder_5: tir.handle, placeholder_6: tir.handle, placeholder_7: tir.handle, placeholder_8: tir.handle, placeholder_9: tir.handle, T_concat: tir.handle) -> None:
        # function attr dict
        tir.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = tir.match_buffer(placeholder_2, [2992], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = tir.match_buffer(placeholder_4, [2992], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_10 = tir.match_buffer(placeholder_1, [1, 8, 10, 16], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_2 = tir.match_buffer(placeholder_9, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_3 = tir.match_buffer(placeholder_8, [2992], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_4 = tir.match_buffer(placeholder_5, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_5 = tir.match_buffer(placeholder_6, [2992], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        T_concat_1 = tir.match_buffer(T_concat, [1, 8, 32, 16], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        placeholder_11 = tir.match_buffer(placeholder, [1, 8, 12, 16], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_6 = tir.match_buffer(placeholder_7, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        buffer_7 = tir.match_buffer(placeholder_3, [160], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        # body
        T_concat_2 = tir.allocate([2816], "int8", "global", annotations={"disable_lower_builtin": True})
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 8, 10, 16, 8, 0, 10, tir.load("int8", placeholder_10.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 160, 16, 1, "int8", 8, 10, 16, 8, 0, 10, tir.load("int8", T_concat_2, 192), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 352, 16, 1, 3, 3, 1, 1, 1, 1, tir.load("uint8", buffer_1.data, 0), 2992, 12, tir.load("uint8", buffer_4.data, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 8, 10, 16, 8, 0, 10, tir.load("int8", T_concat_2, 192), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 352, 16, 1, "int8", 8, 10, 16, 8, 0, 10, tir.load("int8", T_concat_1.data, 352), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 512, 16, 1, 3, 3, 1, 1, 1, 1, tir.load("uint8", buffer_3.data, 0), 2992, 12, tir.load("uint8", buffer_2.data, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 8, 12, 16, 8, 0, 12, tir.load("int8", placeholder_11.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 192, 16, 1, "int8", 8, 12, 16, 8, 0, 12, tir.load("int8", T_concat_2, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 352, 16, 1, 3, 3, 1, 1, 1, 1, tir.load("uint8", buffer.data, 0), 2992, 12, tir.load("uint8", buffer_7.data, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 8, 22, 16, 8, 0, 22, tir.load("int8", T_concat_2, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 352, 16, 1, "int8", 8, 22, 16, 8, 0, 22, tir.load("int8", T_concat_1.data, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 512, 16, 1, 3, 3, 1, 1, 1, 1, tir.load("uint8", buffer_5.data, 0), 2992, 12, tir.load("uint8", buffer_6.data, 0), 160, 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_concat():
    def _get_func():
        ifm1 = relay.var("ifm1", shape=(1, 8, 12, 16), dtype="int8")
        ifm2 = relay.var("ifm2", shape=(1, 8, 10, 16), dtype="int8")
        conv1 = make_ethosu_conv2d(ifm1, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1))
        conv2 = make_ethosu_conv2d(ifm2, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1))
        conc1 = relay.concatenate((conv1, conv2), axis=2)
        conv3 = make_ethosu_conv2d(conc1, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1))
        conv4 = make_ethosu_conv2d(conv2, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1))
        conc2 = relay.concatenate((conv3, conv4), axis=2)
        func = relay.Function(relay.analysis.free_vars(conc2), conc2)
        func = run_opt_pass(func, relay.transform.InferType())
        return func

    func = _get_func()
    mod, _ = lower_to_tir(func)
    script = mod.script(show_meta=True)
    test_mod = tvm.script.from_source(script)

    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)


if __name__ == "__main__":
    pytest.main([__file__])
