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

import tvm
import tvm.testing
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy

default_lwp_test_config = {
    "tir.instrument_lwp": True,
    "tir.lwp_disable_func_prof": True,
    "tir.reset_start_id": True,
}


@T.prim_func
def input1(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    for i, j in T.grid(8, 8):
        for k, l in T.grid(8, 16):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
        for k, l in T.grid(8, 16):
            with T.block("C"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2


@T.prim_func
def input2(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    D = T.match_buffer(d, (8, 8, 128), dtype="int32")
    for i in T.serial(0, 8):
        for j in T.serial(0, 8):
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
        for j in T.serial(0, 8):
            for k, l in T.grid(8, 16):
                with T.block("C"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] + 2
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    C[vi, vj, vk * 16 + vl] = C[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]


@T.prim_func
def input3(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    D = T.match_buffer(d, (8, 8, 128), dtype="int32")
    for i in T.serial(0, 8):
        for j in T.parallel(0, 8):
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
        for j in T.serial(0, 8):
            for k in T.parallel(0, 8):
                for l in T.serial(0, 16):
                    with T.block("C"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] + 2
            for k in T.parallel(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = C[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]


@T.prim_func
def test1_expected_output(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    for i, j in T.grid(8, 8):
        T.evaluate(T.start_profile_intrinsic(3, dtype="handle"))
        for k, l in T.grid(8, 16):
            with T.block("B"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
        T.evaluate(T.end_profile_intrinsic(3, dtype="handle"))
        T.evaluate(T.start_profile_intrinsic(5, dtype="handle"))
        for k, l in T.grid(8, 16):
            with T.block("C"):
                vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
        T.evaluate(T.end_profile_intrinsic(5, dtype="handle"))


@T.prim_func
def test2_expected_output(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    T.evaluate(T.start_profile_intrinsic(1, dtype="handle"))
    for i in T.serial(0, 8):
        T.evaluate(T.start_profile_intrinsic(2, dtype="handle"))
        for j in T.serial(0, 8):
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("C"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
        T.evaluate(T.end_profile_intrinsic(2, dtype="handle"))
    T.evaluate(T.end_profile_intrinsic(1, dtype="handle"))


@T.prim_func
def test3_expected_output(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    T.evaluate(T.start_profile_intrinsic(1, dtype="handle"))
    for i in T.serial(0, 8):
        T.evaluate(T.start_profile_intrinsic(2, dtype="handle"))
        for j in T.serial(0, 8):
            T.evaluate(T.start_profile_intrinsic(3, dtype="handle"))
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            T.evaluate(T.end_profile_intrinsic(3, dtype="handle"))
            T.evaluate(T.start_profile_intrinsic(5, dtype="handle"))
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("C"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
            T.evaluate(T.end_profile_intrinsic(5, dtype="handle"))
        T.evaluate(T.end_profile_intrinsic(2, dtype="handle"))
    T.evaluate(T.end_profile_intrinsic(1, dtype="handle"))


@T.prim_func
def test4_expected_output(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    D = T.match_buffer(d, (8, 8, 128), dtype="int32")
    for i in T.serial(0, 8):
        T.evaluate(T.start_profile_intrinsic(2, dtype="handle"))
        for j in T.serial(0, 8):
            T.evaluate(T.start_profile_intrinsic(3, dtype="handle"))
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            T.evaluate(T.end_profile_intrinsic(3, dtype="handle"))
            T.evaluate(T.start_profile_intrinsic(5, dtype="handle"))
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
            T.evaluate(T.end_profile_intrinsic(5, dtype="handle"))
        T.evaluate(T.end_profile_intrinsic(2, dtype="handle"))
        T.evaluate(T.start_profile_intrinsic(7, dtype="handle"))
        for j in T.serial(0, 8):
            T.evaluate(T.start_profile_intrinsic(8, dtype="handle"))
            for k, l in T.grid(8, 16):
                with T.block("C"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] + 2
            T.evaluate(T.end_profile_intrinsic(8, dtype="handle"))
            T.evaluate(T.start_profile_intrinsic(10, dtype="handle"))
            for k, l in T.grid(8, 16):
                with T.block("B"):
                    vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                    C[vi, vj, vk * 16 + vl] = C[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
            T.evaluate(T.end_profile_intrinsic(10, dtype="handle"))
        T.evaluate(T.end_profile_intrinsic(7, dtype="handle"))


@T.prim_func
def test5_expected_output(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    T.evaluate(T.start_profile_intrinsic(1, dtype="handle"))
    for i in T.serial(0, 8):
        T.evaluate(T.start_profile_intrinsic(2, dtype="handle"))
        for j in T.serial(0, 8):
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("C"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * 2
        T.evaluate(T.end_profile_intrinsic(2, dtype="handle"))
    T.evaluate(T.end_profile_intrinsic(1, dtype="handle"))


@T.prim_func
def test6_expected_output(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 128), dtype="int32")
    B = T.match_buffer(b, (8, 8, 128), dtype="int32")
    C = T.match_buffer(c, (8, 8, 128), dtype="int32")
    D = T.match_buffer(d, (8, 8, 128), dtype="int32")
    for i in T.serial(0, 8):
        T.evaluate(T.start_profile_intrinsic(2, dtype="handle"))
        for j in T.parallel(0, 8):
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = A[vi, vj, vk * 16 + vl] * 2
            for k in T.serial(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        B[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
        T.evaluate(T.end_profile_intrinsic(2, dtype="handle"))
        T.evaluate(T.start_profile_intrinsic(7, dtype="handle"))
        for j in T.serial(0, 8):
            T.evaluate(T.start_profile_intrinsic(8, dtype="handle"))
            for k in T.parallel(0, 8):
                for l in T.serial(0, 16):
                    with T.block("C"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = B[vi, vj, vk * 16 + vl] + 2
            T.evaluate(T.end_profile_intrinsic(8, dtype="handle"))
            T.evaluate(T.start_profile_intrinsic(10, dtype="handle"))
            for k in T.parallel(0, 8):
                for l in T.serial(0, 16):
                    with T.block("B"):
                        vi, vj, vk, vl = T.axis.remap("SSSS", [i, j, k, l])
                        C[vi, vj, vk * 16 + vl] = C[vi, vj, vk * 16 + vl] * D[vi, vj, vk * 16 + vl]
            T.evaluate(T.end_profile_intrinsic(10, dtype="handle"))
        T.evaluate(T.end_profile_intrinsic(7, dtype="handle"))


# By default, only loops with siblings are instrumented.
def test1():
    with tvm.transform.PassContext(config=default_lwp_test_config):
        mod = tvm.IRModule.from_expr(input1)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test1_expected_output)


# By default, only loops with siblings are instrumented. Here, 'lwp_max_depth'
# doesn't have any effect unless 'instr_siblings' is set to False (ex: test3).
def test2():
    test2_config = default_lwp_test_config.copy()
    test2_config.update({"tir.lwp_max_depth": 3})
    with tvm.transform.PassContext(config=test2_config):
        mod = tvm.IRModule.from_expr(input1)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test1_expected_output)


# test3: Use 'lwp_max_depth' to instrument loops upto a certain depth. This flag
# is effective only when 'instr_siblings' is disabled. Also, note that inner-most
# loops are always excluded from instrumentation unless overwritten using
# 'lwp_min_height' (ex: test5)
def test3():
    test3_config = default_lwp_test_config.copy()
    test3_config.update({"tir.lwp_max_depth": 3, "tir.instr_siblings": False})
    with tvm.transform.PassContext(config=test3_config):
        mod = tvm.IRModule.from_expr(input1)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test3_expected_output)


# test4: Use 'lwp_min_height' to exclude inner loops upto a certain height from
# instrumentation.
def test4():
    with tvm.transform.PassContext(config=default_lwp_test_config):
        mod = tvm.IRModule.from_expr(input2)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test4_expected_output)


# test5: Use both 'lwp_min_height' and 'lwp_max_depth'.
# instrumentation.
def test5():
    test5_config = default_lwp_test_config.copy()
    test5_config.update(
        {"tir.lwp_max_depth": 3, "tir.instr_siblings": False, "tir.lwp_min_height": 2}
    )
    with tvm.transform.PassContext(config=test5_config):
        mod = tvm.IRModule.from_expr(input1)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test5_expected_output)


# test6: Tests instrumentation for the parallel loops
def test6():
    with tvm.transform.PassContext(config=default_lwp_test_config):
        mod = tvm.IRModule.from_expr(input3)
        mod = tvm.tir.transform.InstrumentProfileIntrinsics()(mod)
    tvm.ir.assert_structural_equal(mod["main"], test6_expected_output)


if __name__ == "__main__":
    tvm.testing.main()
