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
"""Testcode for Android RPC.

To use it, start an RPC tracker with "python -m tvm.exec.rpc_tracker".
Use the tracker's address and port when configuring the RPC app.
Use "android" as the key if you wish to avoid modifying this script.
"""

import tvm
from tvm import te
import os
from tvm import rpc
from tvm.contrib import utils, ndk
import numpy as np

# Set to be address of tvm proxy.
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -mtriple=%s-linux-android" % arch

# whether enable to execute test on OpenCL target
test_opencl = False
# whether enable to execute test on Vulkan target
test_vulkan = False


def test_rpc_module():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    temp = utils.tempdir()

    # Establish remote connection with target hardware
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request(key, priority=0, session_timeout=60)

    # Compile the Graph for CPU target
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso_cpu = temp.relpath("cpu_lib.so")
    f.export_library(path_dso_cpu, fcompile=ndk.create_shared)

    # Execute the portable graph on cpu target
    print("Run CPU test ...")
    dev = remote.cpu(0)
    remote.upload(path_dso_cpu)
    f2 = remote.load_module("cpu_lib.so")
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
    time_f = f2.time_evaluator(f2.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("%g secs/op\n" % cost)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    # Compile the Graph for OpenCL target
    if test_opencl:
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=64)
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        s[B].bind(xo, te.thread_axis("blockIdx.x"))
        # Build the dynamic lib.
        # If we don't want to do metal and only use cpu, just set target to be target
        f = tvm.build(s, [A, B], tvm.target.Target("opencl", host=target), name="myadd")
        path_dso_cl = temp.relpath("dev_lib_cl.so")
        f.export_library(path_dso_cl, fcompile=ndk.create_shared)

        print("Run GPU(OpenCL Flavor) test ...")
        dev = remote.cl(0)
        remote.upload(path_dso_cl)
        f1 = remote.load_module("dev_lib_cl.so")
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
        time_f = f1.time_evaluator(f1.entry_name, dev, number=10)
        cost = time_f(a, b).mean
        print("%g secs/op\n" % cost)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    # Compile the Graph for Vulkan target
    if test_vulkan:
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=64)
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        s[B].bind(xo, te.thread_axis("blockIdx.x"))
        # Build the dynamic lib.
        # If we don't want to do metal and only use cpu, just set target to be target
        f = tvm.build(s, [A, B], tvm.target.Target("vulkan", host=target), name="myadd")
        path_dso_vulkan = temp.relpath("dev_lib_vulkan.so")
        f.export_library(path_dso_vulkan, fcompile=ndk.create_shared)

        print("Run GPU(Vulkan Flavor) test ...")
        dev = remote.vulkan(0)
        remote.upload(path_dso_vulkan)
        f1 = remote.load_module("dev_lib_vulkan.so")
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
        time_f = f1.time_evaluator(f1.entry_name, dev, number=10)
        cost = time_f(a, b).mean
        print("%g secs/op\n" % cost)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)


if __name__ == "__main__":
    test_rpc_module()
