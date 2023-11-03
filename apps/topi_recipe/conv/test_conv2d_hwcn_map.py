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
"""Example code to do convolution."""
import os

import numpy as np
import tvm
from tvm import te, topi
from tvm.contrib import nvcc
from tvm.topi.utils import get_const_tuple

TASK = "conv2d_hwcn_map"
USE_MANUAL_CODE = False


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code, target):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_conv2d_hwcn_map():
    batch = 64
    in_channel = 128
    in_height = 16
    in_width = 16
    num_filter = 128
    kernel = 3
    stride = 2
    padding = "SAME"

    A = te.placeholder((in_height, in_width, in_channel, batch), name="A")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W")
    B = topi.nn.conv2d_hwcn(A, W, stride, padding)
    C = topi.nn.relu(B)
    s1 = topi.cuda.schedule_conv2d_hwcn([B])
    s2 = topi.cuda.schedule_conv2d_hwcn([C])

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    w_np = np.random.uniform(size=get_const_tuple(W.shape)).astype(W.dtype)
    b_np = tvm.topi.testing.conv2d_hwcn_python(a_np, w_np, stride, padding)
    c_np = np.maximum(b_np, 0)

    def check_device(device):
        if not tvm.runtime.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        dev = tvm.device(device, 0)
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

        with tvm.transform.PassContext(
            config={
                "tir.UrollLoop": {"auto_unroll_max_step": 128, "explicit_unroll": device == "rocm"}
            }
        ):
            func1 = tvm.build(s1, [A, W, B], device)
            func1(a, w, b)
            tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)
            func2 = tvm.build(s2, [A, W, C], device)
            func2(a, w, c)
            tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)

    for device in ["cuda", "opencl", "rocm"]:
        check_device(device)


if __name__ == "__main__":
    test_conv2d_hwcn_map()
