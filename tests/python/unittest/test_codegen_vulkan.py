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


def test_vector_comparison():
    if not tvm.module.enabled("vulkan"):
        print("Skipping due to no Vulkan module")
        return

    target = 'vulkan'

    def check_correct_assembly(dtype):
        n = (1024,)
        A = tvm.placeholder(n, dtype=dtype, name='A')
        B = tvm.compute(
            A.shape,
            lambda i: tvm.expr.Select(
                A[i] >= 0, A[i] + tvm.const(1, dtype),
                tvm.const(0, dtype)), name='B')
        s = tvm.create_schedule(B.op)

        (bx, tx) = s[B].split(s[B].op.axis[0], factor=128)
        (tx, vx) = s[B].split(tx, factor=4)
        s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
        s[B].vectorize(vx)
        f = tvm.build(s, [A, B], target)

        # Verify we generate the boolx4 type declaration and the OpSelect
        # v4{float,half,int} instruction
        assembly = f.imported_modules[0].get_source()
        matches = re.findall("%v4bool = OpTypeVector %bool 4", assembly)
        assert len(matches) == 1
        matches = re.findall("OpSelect %v4.*", assembly)
        assert len(matches) == 1
    check_correct_assembly('float32')
    check_correct_assembly('int32')
    check_correct_assembly('float16')


if __name__ == "__main__":
    test_vector_comparison()
