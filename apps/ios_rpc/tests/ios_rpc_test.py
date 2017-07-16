"""Testcode for iOS RPC.

To use it, start a rpc proxy with "python -m tvm.exec.rpc_proxy".
And configure the proxy host field as commented.
"""

import tvm
import os
from tvm.contrib import rpc, util, xcode
import numpy as np

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]
proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
arch = "arm64"
sdk = "iphoneos"
target = "llvm -target=%s-apple-darwin" % arch

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src):
    return xcode.compile_metal(src, sdk=sdk)

def test_rpc_module():
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    temp = util.tempdir()
    # Build the dynamic lib.
    # If we don't want to do metal and only use cpu, just set target to be target
    f = tvm.build(s, [A, B], "metal", target_host=target, name="myadd")
    path_dso = temp.relpath("dev_lib.dylib")
    f.export_library(path_dso, xcode.create_dylib,
                     arch=arch, sdk=sdk)
    xcode.codesign(path_dso)
    # Start RPC test server that contains the compiled library.
    server = xcode.popen_test_rpc(proxy_host, proxy_port, key,
                                  destination=destination,
                                  libs=[path_dso],
                                  options=["-quiet"])
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

test_rpc_module()
