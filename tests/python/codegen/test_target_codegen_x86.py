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
import platform
import re

import pytest

import tvm
from tvm.script import tir as T, ir as I

llvm_version = tvm.target.codegen.llvm_version_major()
machine = platform.machine()

if machine not in ["x86_64", "AMD64", "amd64"]:
    pytest.skip(f"Requires x86_64, but machine is {machine}", allow_module_level=True)


@tvm.testing.requires_llvm
@pytest.mark.skipif(llvm_version < 6, reason=f"Requires LLVM 6+, got {llvm_version}")
def test_fp16_to_fp32():
    def fp16_to_fp32(target, width, match=None, not_match=None):
        elements = 64

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((elements, width), "float16"),
                B: T.Buffer((elements, width), "float32"),
            ):
                T.func_attr({"tir.noalias": True})
                for i0 in range(elements):
                    for i1 in T.vectorized(width):
                        with T.sblock("B"):
                            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                            T.reads(A[v_i0, v_i1])
                            T.writes(B[v_i0, v_i1])
                            B[v_i0, v_i1] = T.Cast("float32", A[v_i0, v_i1])

        f = tvm.tir.build(Module, target=target)

        assembly = f.inspect_source("asm").splitlines()
        if match:
            matches = [l for l in assembly if re.search(match, l)]
            assert matches
        if not_match:
            not_matches = [l for l in assembly if re.search(not_match, l)]
            assert not not_matches

    fp16_to_fp32({"kind": "llvm", "mcpu": "skylake-avx512"}, 15, match="vcvtph2ps.*mm")
    fp16_to_fp32({"kind": "llvm", "mcpu": "skylake-avx512"}, 16, match="vcvtph2ps.*mm")
    fp16_to_fp32({"kind": "llvm", "mcpu": "skylake-avx512"}, 17, match="vcvtph2ps.*mm")
    fp16_to_fp32({"kind": "llvm", "mcpu": "skylake-avx512"}, 49, match="vcvtph2ps.*mm")
    fp16_to_fp32(
        {"kind": "llvm", "mcpu": "skylake-avx512", "mattr": ["-avx512f"]}, 49, match="vcvtph2ps.*mm"
    )
    fp16_to_fp32(
        {"kind": "llvm", "mcpu": "skylake-avx512", "mattr": ["-f16c", "-avx512f"]},
        49,
        not_match="vcvtph2ps",
    )
    fp16_to_fp32({"kind": "llvm", "mcpu": "core-avx2"}, 8, match="vcvtph2ps.*mm")
    fp16_to_fp32({"kind": "llvm", "mcpu": "core-avx2"}, 9, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm", 9, not_match="vcvtph2ps")


is_32bit = platform.architecture()[0] == "32bit"


if __name__ == "__main__":
    test_fp16_to_fp32()
