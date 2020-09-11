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
from tvm import te
import re


def test_fp16_to_fp32():
    if tvm.target.codegen.llvm_version_major() < 6:
        print(
            "Skipping due to LLVM version being {} < 6".format(
                tvm.target.codegen.llvm_version_major()
            )
        )
        return

    def fp16_to_fp32(target, width, match=None, not_match=None):
        elements = 64
        n = tvm.runtime.convert(elements)
        A = te.placeholder((n, width), dtype="float16", name="A")
        B = te.compute(A.shape, lambda *i: A(*i).astype("float32"), name="B")
        s = te.create_schedule(B.op)
        s[B].vectorize(s[B].op.axis[1])
        f = tvm.build(s, [A, B], target)

        assembly = f.get_source("asm").splitlines()
        if match:
            matches = [l for l in assembly if re.search(match, l)]
            assert matches
        if not_match:
            not_matches = [l for l in assembly if re.search(not_match, l)]
            assert not not_matches

    fp16_to_fp32(
        "llvm -mcpu=skylake-avx512", 15, match="vcvtph2ps.*ymm", not_match="vcvtph2ps.*zmm"
    )
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 16, match="vcvtph2ps.*zmm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 17, match="vcvtph2ps.*zmm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 49, match="vcvtph2ps.*zmm")
    fp16_to_fp32(
        "llvm -mcpu=skylake-avx512 -mattr=-avx512f",
        49,
        match="vcvtph2ps.*ymm",
        not_match="vcvtph2ps.*zmm",
    )
    fp16_to_fp32("llvm -mcpu=skylake-avx512 -mattr=-f16c,-avx512f", 49, not_match="vcvtph2ps")
    fp16_to_fp32("llvm -mcpu=core-avx2", 8, match="vcvtph2ps.*ymm")
    fp16_to_fp32("llvm -mcpu=core-avx2", 9, match="vcvtph2ps.*ymm")
    fp16_to_fp32("llvm", 9, not_match="vcvtph2ps")


if __name__ == "__main__":
    test_fp16_to_fp32()
