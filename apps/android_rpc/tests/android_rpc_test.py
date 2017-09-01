"""Testcode for Android RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""

import tvm
import os
from tvm.contrib import rpc, util, ndk, rpc_proxy
import numpy as np

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_ANDROID_RPC_PROXY_HOST"]
proxy_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -target=%s-linux-android" % arch

def test_rpc_module():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    temp = util.tempdir()
    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    # Build the dynamic lib.
    # If we don't want to do metal and only use cpu, just set target to be target
    f = tvm.build(s, [A, B], "opencl", target_host=target, name="myadd")
    path_dso1 = temp.relpath("dev_lib.so")
    f.export_library(path_dso1, ndk.create_shared)

    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso2 = temp.relpath("cpu_lib.so")
    f.export_library(path_dso2, ndk.create_shared)

    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)

    print('Run GPU test ...')
    ctx = remote.cl(0)
    remote.upload(path_dso1)
    f1 = remote.load_module("dev_lib.so")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
    time_f = f1.time_evaluator(f1.entry_name, ctx, number=10)
    cost = time_f(a, b).mean
    print('%g secs/op' % cost)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    print('Run CPU test ...')
    ctx = remote.cpu(0)
    remote.upload(path_dso2)
    f2 = remote.load_module("cpu_lib.so")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
    time_f = f2.time_evaluator(f2.entry_name, ctx, number=10)
    cost = time_f(a, b).mean
    print('%g secs/op' % cost)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

if __name__ == "__main__":
    test_rpc_module()
