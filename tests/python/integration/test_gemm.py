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
import numpy as np
import time
import tvm.testing


@tvm.testing.requires_gpu
def test_gemm():
    # graph
    nn = 1024
    n = tvm.runtime.convert(nn)
    m = n
    l = n
    A = te.placeholder((n, l), name="A")
    B = te.placeholder((m, l), name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((n, m), lambda ii, jj: te.sum(A[ii, k] * B[jj, k], axis=k), name="CC")
    # schedule
    s = te.create_schedule(C.op)
    xtile, ytile = 32, 32
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis("threadIdx.y")

    CC = s.cache_write(C, "local")
    AA = s.cache_read(A, "shared", [CC])
    BB = s.cache_read(B, "shared", [CC])
    by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    s[C].reorder(by, bx, yi, xi)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    ty, yi = s[C].split(yi, nparts=num_thread)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].reorder(ty, tx, yi, xi)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    yo, xo = CC.op.axis
    s[CC].reorder(k, yo, xo)

    s[CC].compute_at(s[C], tx)
    s[AA].compute_at(s[CC], k)
    s[BB].compute_at(s[CC], k)
    s[AA].double_buffer()
    s[BB].double_buffer()
    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
    tx, xi = s[AA].split(xi, nparts=num_thread)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
    tx, xi = s[BB].split(xi, nparts=num_thread)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)

    # lowering test
    s = s.normalize()

    # one line to build the function.
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled.." % device)
            return

        with tvm.target.Target(device):
            f = tvm.build(s, [A, B, C])

        # launch the kernel.
        n = nn
        m = n
        l = n
        a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        ftimer = f.time_evaluator(f.entry_name, ctx, number=1)
        tcost = ftimer(a, b, c).mean
        print("%s: exec=%g sec/op" % (ctx, tcost))
        tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)

    check_device("vulkan")
    check_device("nvptx -mcpu=sm_20")
    check_device("rocm")
    check_device("metal")
    check_device("opencl")
    check_device("cuda")


if __name__ == "__main__":
    test_gemm()
