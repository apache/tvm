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
import re

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


def test_popcount():
    target = {
        "kind": "llvm",
        "mtriple": "armv7l-none-linux-gnueabihf",
        "mcpu": "cortex-a53",
        "mattr": ["+neon"],
    }

    def check_correct_assembly(type, elements, counts):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(A: T.Buffer((elements,), type), B: T.Buffer((elements,), type)):
                T.func_attr({"tirx.noalias": True})
                for i in T.vectorized(elements):
                    B[i] = T.popcount(A[i])

        f = tvm.tirx.build(Module, target=target)
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
    target = {
        "kind": "llvm",
        "mtriple": "armv7l-none-linux-gnueabihf",
        "mcpu": "cortex-a53",
        "mattr": ["+neon"],
    }

    def check_correct_assembly(N):
        K = T.dynamic("K", "int32")

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((K, N), "int8"), B: T.Buffer((K, N), "int8"), C: T.Buffer((N,), "int32")
            ):
                T.func_attr({"tirx.noalias": True})

                for n in T.vectorized(N):
                    C[n] = 0
                    for rv in range(K):
                        C[n] = C[n] + T.Cast("int32", A[rv, n]) * T.Cast("int32", B[rv, n])

        f = tvm.tirx.build(Module, target=target)

        # Verify we see the correct number of vmlal.s16 instructions
        assembly = f.inspect_source("asm")
        matches = re.findall("vmlal.s16", assembly)
        assert len(matches) == N // 4

    check_correct_assembly(8)
    check_correct_assembly(16)
    check_correct_assembly(32)
    check_correct_assembly(64)

    def check_broadcast_correct_assembly(N):
        K = T.dynamic("K", "int32")

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((K, N), "int8"), B: T.Buffer((K,), "int8"), C: T.Buffer((N,), "int32")
            ):
                T.func_attr({"tirx.noalias": True})

                for n in T.vectorized(N):
                    C[n] = 0
                    for rv in range(K):
                        C[n] = C[n] + T.Cast("int32", A[rv, n]) * T.Cast("int32", B[rv])

        f = tvm.tirx.build(Module, target=target)

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
