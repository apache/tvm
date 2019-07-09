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
"""codegen related to bool types"""

import tvm
import numpy as np

def test_cmp_load_store():
    n = 32
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) > B(*i), name='C')
    D = tvm.compute(C.shape, lambda *i: tvm.all(C(*i),
                                                A(*i) > 1).astype('float32'), name="D")


    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        s = tvm.create_schedule(D.op)
        xo, xi = s[C].split(C.op.axis[0], factor=4)
        xo1, xo2 = s[C].split(xo, factor=13)
        s[C].parallel(xo2)
        # BUILD and invoke the kernel.
        f = tvm.build(s, [A, B, D], "llvm")
        ctx = tvm.cpu(0)
        a_np = np.random.uniform(size=n).astype(A.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros(n, dtype=D.dtype), ctx)
        f(a, b, d)
        np.testing.assert_equal(
            d.asnumpy(), np.logical_and(a.asnumpy() > b.asnumpy(), a.asnumpy() > 1).astype('float32'))

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            return
        s = tvm.create_schedule(D.op)
        for stage in [C, D]:
            xo, xi = s[stage].split(stage.op.axis[0], factor=4)
            s[stage].bind(xo, tvm.thread_axis("blockIdx.x"))
            s[stage].bind(xi, tvm.thread_axis("threadIdx.x"))
        f = tvm.build(s, [A, B, D], device)
        a_np = np.random.uniform(size=n).astype(A.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        d = tvm.nd.array(np.zeros(n, dtype=D.dtype), ctx)
        f(a, b, d)
        np.testing.assert_equal(
            d.asnumpy(), np.logical_and(a.asnumpy() > b.asnumpy(), a.asnumpy() > 1).astype('float32'))


    check_llvm()
    for device in ["vulkan", "opencl", "cuda", "rocm", "metal"]:
        check_device(device)



if __name__ == "__main__":
    test_cmp_load_store()
