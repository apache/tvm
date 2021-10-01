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
from tvm.script import tir, ty
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_tir
from tvm.relay.backend.contrib.ethosu.tir.scheduler import copy_constants

from .infra import make_ethosu_conv2d


# fmt: off
@tvm.script.tir
class ReferenceModule:
    def main(placeholder: ty.handle, placeholder_1: ty.handle, placeholder_2: ty.handle, ethosu_write: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        buffer = tir.match_buffer(placeholder_2, [80], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        placeholder_3 = tir.match_buffer(placeholder, [1, 16, 16, 32], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        buffer_1 = tir.match_buffer(placeholder_1, [304], dtype="uint8", elem_offset=0, align=128, offset_factor=1)
        ethosu_write_1 = tir.match_buffer(ethosu_write, [1, 16, 16, 8], dtype="int8", elem_offset=0, align=128, offset_factor=1)
        # body
        placeholder_global = tir.allocate([304], "uint8", "global")
        placeholder_d_global = tir.allocate([80], "uint8", "global")
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer_1.data, 0), 304, tir.load("uint8", placeholder_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_copy", tir.load("uint8", buffer.data, 0), 80, tir.load("uint8", placeholder_d_global, 0), dtype="handle"))
        tir.evaluate(tir.call_extern("ethosu_conv2d", "int8", 16, 16, 32, 16, 0, 16, tir.load("int8", placeholder_3.data, 0), 0, 0, 0, tir.float32(0.5), 10, "NHWC", 512, 32, 1, "int8", 16, 16, 8, 16, 0, 16, tir.load("int8", ethosu_write_1.data, 0), 0, 0, 0, tir.float32(0.25), 14, "NHWC", 128, 8, 1, 1, 1, 1, 1, 1, 1, tir.load("uint8", placeholder_global, 0), 304, 12, tir.load("uint8", placeholder_d_global, 0), 80, 0, 0, 0, 0, "NONE", 0, 0, "NONE", dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_copy():
    def _get_func():
        data = relay.var("data", shape=(1, 16, 16, 32), dtype="int8")
        conv = make_ethosu_conv2d(
            data,
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
    mod, _ = lower_to_tir(func, cascader=copy_constants())

    script = tvm.script.asscript(mod, True)
    test_mod = tvm.script.from_source(script)
    reference_mod = ReferenceModule()
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)


if __name__ == "__main__":
    pytest.main([__file__])
