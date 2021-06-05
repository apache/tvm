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
import tvm.testing
from tvm import te
import numpy as np
import os

os.environ["XCL_EMULATION_MODE"] = "1"
os.environ["CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA"] = "1"


@tvm.register_func
def tvm_callback_vhls_postproc(code):
    """Hook to inspect the Vivado HLS code before actually run it"""
    print(code)
    return code


def test_exp():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: te.exp(A(*i)), name="B")
    s = te.create_schedule(B.op)
    # create iter var and assign them tags.
    px, x = s[B].split(B.op.axis[0], nparts=1)
    s[B].bind(px, te.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.testing.device_enabled(device):
            return
        dev = tvm.device(device, 0)
        fexp = tvm.build(s, [A, B], device, host, name="myexp")
        dev = tvm.device(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fexp(a, b)
        tvm.testing.assert_allclose(b.numpy(), np.exp(a.numpy()), rtol=1e-5)

    check_device("sdaccel")
    if "AWS_PLATFORM" in os.environ:
        check_device("sdaccel -device=" + os.environ.get("AWS_PLATFORM"))

    check_device("aocl_sw_emu")


def test_multi_kernel():
    # graph
    n = tvm.runtime.convert(1024)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")
    D = te.compute(A.shape, lambda *i: A(*i) + C(*i), name="D")
    s = te.create_schedule(D.op)
    # create iter var and assign them tags.
    px, x = s[C].split(C.op.axis[0], nparts=1)
    s[C].bind(px, te.thread_axis("pipeline"))
    px, x = s[D].split(D.op.axis[0], nparts=1)
    s[D].bind(px, te.thread_axis("pipeline"))

    # one line to build the function.
    def check_device(device, host="llvm"):
        if not tvm.testing.device_enabled(device):
            return
        dev = tvm.device(device, 0)
        fadd = tvm.build(s, [A, B, C, D], device, host, name="myadd")
        dev = tvm.device(device, 0)
        # launch the kernel.
        n = 1024
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.random.uniform(size=n).astype(C.dtype), dev)
        d = tvm.nd.array(np.random.uniform(size=n).astype(D.dtype), dev)
        fadd(a, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), a.numpy() * 2 + b.numpy(), rtol=1e-5)

    check_device("sdaccel")
    check_device("aocl_sw_emu")


if __name__ == "__main__":
    test_exp()
    test_multi_kernel()
