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
# pylint: disable=missing-function-docstring,missing-module-docstring
import tvm
from tvm.script import ty
from tvm import te, tir
import numpy as np
import tvm.testing
import pytest


@tvm.script.tir
class Module:
    def tir_packed_call() -> None:
        A = tir.var("handle")
        B = tir.var("handle")
        C = tir.var("handle")
        # body
        tir.evaluate(
            tir.tvm_call_cpacked(
                "tvm_test_cpacked",
                A,
                B,
                C,
                dtype="int32",
            )
        )


@tvm.script.tir
class Expected:
    def tir_packed_call() -> None:
        A = tir.var("handle")
        B = tir.var("handle")
        C = tir.var("handle")

        # body
        tvm_value_2 = tir.var("handle")
        tvm_value_1 = tir.var("handle")
        tvm_value_0 = tir.var("handle")
        with tir.let(tvm_value_2, tir.tvm_stack_alloca("array", 1, dtype="handle")):
            with tir.let(tvm_value_1, tir.tvm_stack_alloca("array", 1, dtype="handle")):
                with tir.let(tvm_value_0, tir.tvm_stack_alloca("array", 1, dtype="handle")):
                    tir.evaluate(tir.tvm_struct_set(tvm_value_0, 0, 1, A, dtype="handle"))
                    tir.evaluate(tir.tvm_struct_set(tvm_value_1, 0, 1, B, dtype="handle"))
                    tir.evaluate(tir.tvm_struct_set(tvm_value_2, 0, 1, C, dtype="handle"))
                    tir.evaluate(
                        tir.tvm_call_cpacked(
                            "tvm_test_cpacked",
                            tvm_value_0,
                            tvm_value_1,
                            tvm_value_2,
                            dtype="int32",
                        )
                    )


def test_aot_packed_call():
    mod = Module()
    expected = Expected()
    out = tir.transform.LegalizePackedCalls()(mod)
    tvm.ir.assert_structural_equal(expected, out, map_free_vars=True)


if __name__ == "__main__":
    pytest.main([__file__])
