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
import re
from tvm.script import tir as T, ir as I


def test_popcount():
    target = "llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"

    def check_correct_assembly(type, elements, counts):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((elements,), type), B: T.Buffer((elements,), type)):
                T.func_attr({"tir.noalias": True})
                for i in T.vectorized(elements):
                    with T.sblock("B"):
                        v_i = T.axis.spatial(elements, i)
                        T.reads(A[v_i])
                        T.writes(B[v_i])
                        B[v_i] = T.popcount(A[v_i])

        f = tvm.tir.build(Module, target=target)
        # Verify we see the correct number of vpaddl and vcnt instructions in the assembly
        assembly = f.inspect_source("asm")
        matches = re.findall("vpaddl", assembly)
        assert len(matches) == counts
        matches = re.findall("vcnt", assembly)
        assert len(matches) == 1

    check_correct_assembly("uint16", 8, 1)
    check_correct_assembly("uint16", 4, 1)
    check_correct_assembly("uint32", 4, 2)
    check_correct_assembly("uint32", 2, 2)
    check_correct_assembly("uint64", 2, 3)


def test_vmlal_s16():
    target = "llvm -mtriple=armv7l-none-linux-gnueabihf -mcpu=cortex-a53 -mattr=+neon"

    def check_correct_assembly(N):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(var_A: T.handle, var_B: T.handle, C: T.Buffer((N,), "int32")):
                T.func_attr({"tir.noalias": True})
                K = T.int32(is_size_var=True)
                A = T.match_buffer(var_A, (K, N), "int8")
                B = T.match_buffer(var_B, (K, N), "int8")
                for n in T.vectorized(N):
                    for rv in range(K):
                        with T.sblock("C"):
                            v_n, v_rv = T.axis.remap("SR", [n, rv])
                            T.reads(A[v_rv, v_n], B[v_rv, v_n])
                            T.writes(C[v_n])
                            with T.init():
                                C[v_n] = 0
                            C[v_n] = C[v_n] + T.Cast("int32", A[v_rv, v_n]) * T.Cast(
                                "int32", B[v_rv, v_n]
                            )

        f = tvm.tir.build(Module, target=target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.inspect_source("asm")
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4

    check_correct_assembly(8)
    check_correct_assembly(16)
    check_correct_assembly(32)
    check_correct_assembly(64)

    def check_broadcast_correct_assembly(N):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(var_A: T.handle, var_B: T.handle, C: T.Buffer((N,), "int32")):
                T.func_attr({"tir.noalias": True})
                K = T.int32(is_size_var=True)
                A = T.match_buffer(var_A, (K, N), "int8")
                B = T.match_buffer(var_B, (K,), "int8")
                for n in T.vectorized(N):
                    for rv in range(K):
                        with T.sblock("C"):
                            v_n, v_rv = T.axis.remap("SR", [n, rv])
                            T.reads(A[v_rv, v_n], B[v_rv])
                            T.writes(C[v_n])
                            with T.init():
                                C[v_n] = 0
                            C[v_n] = C[v_n] + T.Cast("int32", A[v_rv, v_n]) * T.Cast(
                                "int32", B[v_rv]
                            )

        f = tvm.tir.build(Module, target=target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.inspect_source("asm")
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4

    check_broadcast_correct_assembly(8)
    check_broadcast_correct_assembly(16)
    check_broadcast_correct_assembly(32)
    check_broadcast_correct_assembly(64)


if __name__ == "__main__":
    test_popcount()
    test_vmlal_s16()
