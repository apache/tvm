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
import numpy as np
import platform
import pytest
import re
import textwrap

import tvm
from tvm import te

llvm_version = tvm.target.codegen.llvm_version_major()
machine = platform.machine()

if machine not in ["i386", "x86_64", "AMD64", "amd64"]:
    pytest.skip(f"Requires x86_64/i386, but machine is {machine}", allow_module_level=True)


@tvm.testing.requires_llvm
@pytest.mark.skipif(llvm_version < 6, reason=f"Requires LLVM 6+, got {llvm_version}")
def test_fp16_to_fp32():
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

    fp16_to_fp32("llvm -mcpu=skylake-avx512", 15, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 16, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 17, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512", 49, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512 -mattr=-avx512f", 49, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=skylake-avx512 -mattr=-f16c,-avx512f", 49, not_match="vcvtph2ps")
    fp16_to_fp32("llvm -mcpu=core-avx2", 8, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm -mcpu=core-avx2", 9, match="vcvtph2ps.*mm")
    fp16_to_fp32("llvm", 9, not_match="vcvtph2ps")


is_32bit = platform.architecture()[0] == "32bit"


@tvm.testing.requires_llvm
@pytest.mark.skipif(is_32bit, reason=f"Fails in CI due to architecture mismatch in JIT")
@pytest.mark.parametrize("feature_string", ["-sse2", "+sse2"])
def test_fp16_fp32_conversions(feature_string):
    relay_model = textwrap.dedent(
        """
        #[version = "0.0.5"]
        def @main(%inp : Tensor[(3), float32], %cst : Tensor[(3), float32]) {
            %1 = cast(%inp, dtype="float16");
            %2 = cast(%cst, dtype="float16");
            %3 = add(%1, %2);
            %4 = cast(%3, dtype="float32");
            %4
        }
        """
    )

    ir_mod = tvm.relay.fromtext(relay_model)

    arch = "i386" if machine == "i386" else "x86_64"
    aot_factory = tvm.relay.build(
        ir_mod,
        params={"cst": np.array([1.0, 2.0, 3.0], dtype="float32")},
        target=f"llvm --mtriple={arch} --mattr={feature_string}",
        executor=tvm.relay.backend.Executor(
            "aot", {"interface-api": "packed", "unpacked-api": False}
        ),
    )

    mod_name = aot_factory["list_module_names"]()[0]
    executor = aot_factory[mod_name]
    mod = executor(tvm.cpu(0))

    inp = tvm.nd.array(np.array([1.1, 2.1, 3.1], dtype="float32"), device=tvm.cpu(0))

    mod.get_function("set_input")(0, inp)
    mod.get_function("run")()
    out = mod.get_function("get_output")(0)

    expected = np.array([2.1, 4.1, 6.1], dtype="float32")
    np.testing.assert_allclose(out.asnumpy(), expected, rtol=1e-3)


if __name__ == "__main__":
    test_fp16_to_fp32()
