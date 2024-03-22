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
from tvm import relay
from tvm.relay.backend.contrib.ethosu.tir.compiler import _lower_to_tir
from tvm.relay.testing import run_opt_pass
from tvm.script import tir as T

from .infra import make_ethosu_conv2d


# fmt: off
@tvm.script.ir_module
class ReferenceModule:
    @T.prim_func
    def main(input_placeholder: T.Buffer((1,8,12,16), "int8"), input_placeholder_1: T.Buffer((1,8,10,16), "int8"), input_T_concat: T.Buffer((1,8,32,16), "int8")) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})

        placeholder = T.Buffer(1536, dtype="int8", data=input_placeholder.data)
        placeholder_1 = T.Buffer(1280, dtype="int8", data=input_placeholder_1.data)
        T_concat = T.Buffer(4096, dtype="int8", data=input_T_concat.data)

        buffer = T.Buffer([2992], "uint8")
        buffer_1 = T.Buffer([160], "uint8")
        buffer_2 = T.Buffer([2992], "uint8")
        buffer_3 = T.Buffer([160], "uint8")
        buffer_4 = T.Buffer([2992], "uint8")
        buffer_5 = T.Buffer([160], "uint8")
        buffer_6 = T.Buffer([2992], "uint8")
        buffer_7 = T.Buffer([160], "uint8")
        # body
        T_concat_1_data = T.allocate([2816], "int8", "global", annotations={"disable_lower_builtin":True})
        T_concat_1 = T.Buffer([2816], "int8", data=T_concat_1_data)
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 10, 16, 8, 0, 10, placeholder_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 160, 16, 1, "int8", 8, 10, 16, 8, 0, 10, T_concat_1[192], 0, 0, 0, T.float32(0.25), 14, "NHWC", 352, 16, 1, 3, 3, 1, 1, 1, 1, buffer[0], 2992, T.int8(-1), T.int8(-1), 12, buffer_1[0], 160, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 10, 16, 8, 0, 10, T_concat_1[192], 0, 0, 0, T.float32(0.5), 10, "NHWC", 352, 16, 1, "int8", 8, 10, 16, 8, 0, 10, T_concat[352], 0, 0, 0, T.float32(0.25), 14, "NHWC", 512, 16, 1, 3, 3, 1, 1, 1, 1, buffer_2[0], 2992, T.int8(-1), T.int8(-1), 12, buffer_3[0], 160, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 12, 16, 8, 0, 12, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 192, 16, 1, "int8", 8, 12, 16, 8, 0, 12, T_concat_1[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 352, 16, 1, 3, 3, 1, 1, 1, 1, buffer_4[0], 2992, T.int8(-1), T.int8(-1), 12, buffer_5[0], 160, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 8, 22, 16, 8, 0, 22, T_concat_1[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 352, 16, 1, "int8", 8, 22, 16, 8, 0, 22, T_concat[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 512, 16, 1, 3, 3, 1, 1, 1, 1, buffer_6[0], 2992, T.int8(-1), T.int8(-1), 12, buffer_7[0], 160, T.int8(-1), T.int8(-1), 1, 1, 1, 1, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
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
    mod, _ = _lower_to_tir(func)
    script = mod.script()
    test_mod = tvm.script.from_source(script)

    reference_mod = ReferenceModule
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)


if __name__ == "__main__":
    tvm.testing.main()
