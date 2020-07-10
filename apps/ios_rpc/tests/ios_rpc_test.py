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

import tvm
from tvm import te
import os
import re
import sys
from tvm import rpc
from tvm.contrib import util, xcode
import numpy as np

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

if not re.match(r"^platform=.*,id=.*$", destination):
    print("Bad format: {}".format(destination))
    print("Example of expected string: platform=iOS,id=1234567890abcabcabcabc1234567890abcabcab")
    sys.exit(1)

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target = "llvm -mtriple=%s-apple-darwin" % arch

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src):
    return xcode.compile_metal(src, sdk=sdk)

def test_rpc_module():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name='A')
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    temp = util.tempdir()
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(xi, te.thread_axis("threadIdx.x"))
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    # Build the dynamic lib.
    # If we don't want to do metal and only use cpu, just set target to be target
    f = tvm.build(s, [A, B], "metal", target_host=target, name="myadd")
    path_dso1 = temp.relpath("dev_lib.dylib")
    f.export_library(path_dso1, xcode.create_dylib,
                     arch=arch, sdk=sdk)
    xcode.codesign(path_dso1)

    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].parallel(xi)
    s[B].pragma(xo, "parallel_launch_point")
    s[B].pragma(xi, "parallel_barrier_when_finish")
    f = tvm.build(s, [A, B], target, name="myadd_cpu")
    path_dso2 = temp.relpath("cpu_lib.dylib")
    f.export_library(path_dso2, xcode.create_dylib,
                     arch=arch, sdk=sdk)
    xcode.codesign(path_dso2)

    # Start RPC test server that contains the compiled library.
    server = xcode.popen_test_rpc(proxy_host, proxy_port, key,
                                  destination=destination,
                                  libs=[path_dso1, path_dso2])

    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)
    ctx = remote.metal(0)
    f1 = remote.load_module("dev_lib.dylib")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
    time_f = f1.time_evaluator(f1.entry_name, ctx, number=10)
    cost = time_f(a, b).mean
    print('%g secs/op' % cost)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
    # CPU
    ctx = remote.cpu(0)
    f2 = remote.load_module("cpu_lib.dylib")
    a_np = np.random.uniform(size=1024).astype(A.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
    time_f = f2.time_evaluator(f1.entry_name, ctx, number=10)
    cost = time_f(a, b).mean
    print('%g secs/op' % cost)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

test_rpc_module()
