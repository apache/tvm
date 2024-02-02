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
""" Instrument test cases.
"""

import tvm
from tvm import relax
from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

# pylint: disable=invalid-name,missing-function-docstring,no-value-for-parameter


def test_tir_print_all_passes(capsys):
    @T.prim_func
    def func(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, (128, 128, 128, 128))
        B = T.match_buffer(b, (128, 128, 128, 128))
        for i, j, k, l in T.grid(128, 128, 128, 128):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0

    with tvm.transform.PassContext(opt_level=3, instruments=[PrintBeforeAll(), PrintAfterAll()]):
        tvm.lower(func)
    all_passes_output = capsys.readouterr().out
    assert "Before Running Pass:" in all_passes_output
    assert "After Running Pass:" in all_passes_output
    assert "pass name: tir." in all_passes_output


def test_relax_print_all_passes(capsys):
    @I.ir_module
    class Module:
        @R.function
        def func(x: R.Tensor((16,), "float32"), y: R.Tensor((16,), "float32")):
            z = R.add(x, y)
            y = z
            return y

    pipeline = relax.get_pipeline("default_build")
    with tvm.transform.PassContext(opt_level=3, instruments=[PrintBeforeAll(), PrintAfterAll()]):
        pipeline(Module)
    all_passes_output = capsys.readouterr().out
    assert "Before Running Pass:" in all_passes_output
    assert "After Running Pass:" in all_passes_output
    assert "pass name: _pipeline" in all_passes_output
