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
"""Example code to do square matrix multiplication on Android Phone."""
import tvm
from tvm import te
import os
from tvm import rpc
from tvm.contrib import utils, ndk
import numpy as np

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -mtriple=%s-linux-android" % arch


def ngflops(N):
    return 2.0 * float(N * N * N) / (10**9)


dtype = "float32"


def evaluate(func, dev, N, times):
    a_np = np.random.uniform(size=(N, N)).astype(dtype)
    b_np = np.random.uniform(size=(N, N)).astype(dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, N), dtype=dtype), dev)

    time_f = func.time_evaluator(func.entry_name, dev, number=times)
    cost = time_f(a, b, c).mean
    gf = ngflops(N) / cost
    print("%g secs/op, %g GFLOPS" % (cost, gf))
    np.testing.assert_almost_equal(c.numpy(), a_np.dot(b_np), decimal=2)


def test_gemm_gpu(N, times, bn, num_block, num_thread):
    assert bn <= N
    assert num_thread * num_thread * 16 <= N
    assert num_block * num_block * 2 <= N
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="Btmp")
    k = te.reduce_axis((0, N), name="k")

    packedB = te.compute((N, N / bn, bn), lambda x, y, z: B[x, y * bn + z], name="B")

    C = te.compute(
        (N, N), lambda ii, jj: te.sum(A[ii, k] * packedB[k, jj / bn, jj % bn], axis=k), name="C"
    )

    s = te.create_schedule(C.op)
    CC = s.cache_write(C, "local")

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

    pby, pbi = s[packedB].split(packedB.op.axis[0], nparts=num_thread)
    pbx, pbj = s[packedB].split(packedB.op.axis[1], nparts=num_thread)
    s[packedB].bind(pby, thread_y)
    s[packedB].bind(pbx, thread_x)
    pbz, pbk = s[packedB].split(packedB.op.axis[2], factor=8)
    s[packedB].vectorize(pbk)

    by, yi = s[C].split(C.op.axis[0], nparts=num_block)
    bx, xi = s[C].split(C.op.axis[1], nparts=num_thread)

    s[C].bind(by, block_y)
    s[C].bind(bx, thread_y)
    s[C].reorder(by, bx, yi, xi)

    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=num_block)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=num_thread)

    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)

    s[C].bind(ty, block_x)
    s[C].bind(tx, thread_x)

    xyi, xxi = s[C].split(xi, factor=8)
    s[C].reorder(tyz, txz, ty, tx, yi, xyi, xxi)
    s[C].vectorize(xxi)

    s[CC].compute_at(s[C], yi)
    yo, xo = CC.op.axis
    s[CC].reorder(k, yo, xo)
    xo, xi = s[CC].split(xo, factor=8)
    s[CC].vectorize(xi)

    ko, ki = s[CC].split(k, factor=2)
    s[CC].unroll(ki)

    print(tvm.lower(s, [A, B, C], simple_mode=True))

    f = tvm.build(s, [A, B, C], tvm.target.Target("opencl", host=target), name="gemm_gpu")
    temp = utils.tempdir()
    path_dso = temp.relpath("gemm_gpu.so")
    f.export_library(path_dso, fcompile=ndk.create_shared)

    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)
    dev = remote.cl(0)
    remote.upload(path_dso)
    f = remote.load_module("gemm_gpu.so")

    evaluate(f, dev, N, times)


if __name__ == "__main__":
    test_gemm_gpu(1024, times=5, bn=8, num_block=2, num_thread=8)
