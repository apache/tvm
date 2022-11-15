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
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T


@tvm.script.ir_module
class Module:
    @T.prim_func
    def tvm_test_cpacked(
        A: T.Buffer[(1,), "float32"],
        B: T.Buffer[(1,), "float32"],
        C: T.Buffer[(1,), "float32"],
        device_context: T.Buffer[(1,), "float32"],
    ) -> T.handle:
        T.evaluate(C.data)

    @T.prim_func
    def tir_packed_call() -> None:
        A = T.var("handle")
        B = T.var("handle")
        C = T.var("handle")
        device_context = T.var("handle")
        # body
        T.evaluate(
            T.tvm_call_cpacked(
                "tvm_test_cpacked",
                A,
                B,
                C,
                device_context,
                dtype="int32",
            )
        )


@tvm.script.ir_module
class Expected:
    @T.prim_func
    def tvm_test_cpacked(
        A: T.Buffer[(1,), "float32"],
        B: T.Buffer[(1,), "float32"],
        C: T.Buffer[(1,), "float32"],
        device_context: T.Buffer[(1,), "float32"],
    ) -> T.handle:
        T.evaluate(C.data)

    @T.prim_func
    def tir_packed_call() -> None:
        A = T.var("handle")
        B = T.var("handle")
        C = T.var("handle")
        device_context = T.var("handle")

        # body
        T.evaluate(
            T.tvm_call_cpacked(
                "tvm_test_cpacked",
                T.tvm_stack_make_array(
                    A,
                    T.tvm_stack_make_shape(1, dtype="handle"),
                    T.reinterpret(T.uint64(0), dtype="handle"),
                    T.uint32(1),
                    T.Cast("float32", 0),
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    B,
                    T.tvm_stack_make_shape(1, dtype="handle"),
                    T.reinterpret(T.uint64(0), dtype="handle"),
                    T.uint32(1),
                    T.Cast("float32", 0),
                    0,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    C,
                    T.tvm_stack_make_shape(1, dtype="handle"),
                    T.reinterpret(T.uint64(0), dtype="handle"),
                    T.uint32(1),
                    T.Cast("float32", 0),
                    0,
                    dtype="handle",
                ),
                device_context,
                dtype="int32",
            )
        )


def test_aot_packed_call():
    mod = Module
    expected = Expected
    out = tir.transform.LegalizePackedCalls()(mod)
    tvm.ir.assert_structural_equal(expected, out, map_free_vars=True)


if __name__ == "__main__":
    pytest.main([__file__])
