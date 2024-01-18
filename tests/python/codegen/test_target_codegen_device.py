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
from tvm.contrib import utils
import numpy as np
import tvm.testing


@tvm.testing.requires_gpu
def test_large_uint_imm():
    value = (1 << 63) + 123
    other = tvm.tir.const(3, "uint64")
    n = 12
    num_thread = 2

    A = te.compute((n,), lambda *i: tvm.tir.const(value, "uint64") + other, name="A")
    s = te.create_schedule(A.op)
    xo, xi = s[A].split(A.op.axis[0], factor=num_thread)
    s[A].bind(xi, te.thread_axis("threadIdx.x"))
    s[A].bind(xo, te.thread_axis("blockIdx.x"))

    def check_target(device):
        if not tvm.testing.device_enabled(device):
            return
        dev = tvm.device(device, 0)
        f = tvm.build(s, [A], device)
        # launch the kernel.
        a = tvm.nd.empty((n,), dtype=A.dtype, device=dev)
        f(a)
        assert a.numpy()[0] == value + 3

    check_target("cuda")
    check_target("vulkan -from_device=0")


@tvm.testing.requires_gpu
def test_add_pipeline():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(), name="C")
    D = te.compute(A.shape, lambda *i: C(*i) + 1, name="D")
    s = te.create_schedule(D.op)

    # GPU schedule have to split by gridIdx and threadIdx
    num_thread = 256
    xo, xi = s[C].split(C.op.axis[0], factor=num_thread)
    s[C].bind(xi, te.thread_axis("threadIdx.x"))
    s[C].bind(xo, te.thread_axis("blockIdx.x"))

    xo, xi = s[D].split(D.op.axis[0], factor=num_thread)
    s[D].bind(xi, te.thread_axis("threadIdx.x"))
    s[D].bind(xo, te.thread_axis("blockIdx.x"))

    def check_target(device, host="stackvm"):
        if not tvm.testing.device_enabled(device) or not tvm.testing.device_enabled(host):
            return
        dev = tvm.device(device, 0)
        mhost = tvm.driver.build(s, [A, B, D], target=tvm.target.Target(device, host))
        f = mhost.entry_func
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=()).astype(B.dtype), dev)
        d = tvm.nd.array(np.zeros(n, dtype=D.dtype), dev)
        f(a, b, d)
        tvm.testing.assert_allclose(d.numpy(), a.numpy() + b.numpy() + 1)

    check_target("cuda", host="llvm")
    check_target("nvptx", host="llvm")
    check_target("vulkan", host="llvm")
    check_target("rocm", host="llvm")


if __name__ == "__main__":
    test_large_uint_imm()
    test_add_pipeline()
