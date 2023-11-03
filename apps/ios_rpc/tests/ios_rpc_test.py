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
"""Testcode for iOS RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""

import argparse
import os
import re
import sys

import numpy as np
import tvm
from tvm import rpc, te
from tvm.contrib import utils, xcode

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target = "llvm -mtriple=%s-apple-darwin" % arch

MODES = {"proxy": rpc.connect, "tracker": rpc.connect_tracker, "standalone": rpc.connect}


# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src, target):
    return xcode.compile_metal(src, sdk=sdk)


def test_rpc_module(host, port, key, mode):
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    temp = utils.tempdir()
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    # Build the dynamic lib.
    # If we don't want to do metal and only use cpu, just set target to be target
    f = tvm.build(s, [A, B], tvm.target.Target("metal", host=target), name="myadd")
    path_dso1 = temp.relpath("dev_lib.dylib")
    f.export_library(path_dso1, fcompile=xcode.create_dylib, arch=arch, sdk=sdk)

    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso2 = temp.relpath("cpu_lib.dylib")
    f.export_library(path_dso2, fcompile=xcode.create_dylib, arch=arch, sdk=sdk)

    # connect to the proxy
    if mode == "tracker":
        remote = MODES[mode](host, port).request(key)
    else:
        remote = MODES[mode](host, port, key=key)
    remote.upload(path_dso1)
    dev = remote.metal(0)
    f1 = remote.load_module("dev_lib.dylib")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
    time_f = f1.time_evaluator(f1.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("Metal: %g secs/op" % cost)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    # CPU
    dev = remote.cpu(0)
    remote.upload(path_dso2)
    f2 = remote.load_module("cpu_lib.dylib")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), dev)
    time_f = f2.time_evaluator(f2.entry_name, dev, number=10)
    cost = time_f(a, b).mean
    print("CPU: %g secs/op" % cost)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo app demonstrates how ios_rpc works.")
    parser.add_argument("--host", required=True, type=str, help="Adress of rpc server")
    parser.add_argument("--port", type=int, default=9090, help="rpc port (default: 9090)")
    parser.add_argument("--key", type=str, default="iphone", help="device key (default: iphone)")
    parser.add_argument(
        "--mode",
        type=str,
        default="tracker",
        help="type of RPC connection (default: tracker), possible values: {}".format(
            ", ".join(MODES.keys())
        ),
    )

    args = parser.parse_args()
    assert args.mode in MODES.keys()
    test_rpc_module(args.host, args.port, args.key, args.mode)
