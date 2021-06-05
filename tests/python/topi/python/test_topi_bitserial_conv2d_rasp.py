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
import os
import re
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple


def generate_quantized_np(shape, bits, out_dtype):
    np.random.seed(0)
    min_val = 0
    max_val = 1 << bits
    return np.random.randint(min_val, max_val, size=shape).astype(out_dtype)


# Verify that certain special instructions from the tensorize pass exist
def verify_bitserial_conv2d_nhwc(
    batch,
    in_size,
    in_channel,
    num_filter,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    unipolar,
    use_relu=False,
):
    in_height = in_width = in_size
    input_type = "uint32"
    out_dtype = "int16"

    device = "llvm -device=arm_cpu -model=bcm2837 -mtriple=armv7l-linux-gnueabihf -mattr=+neon"
    with tvm.target.Target(device):
        A = te.placeholder((batch, in_height, in_width, in_channel), dtype=input_type, name="A")
        W = te.placeholder((kernel, kernel, in_channel, num_filter), dtype=input_type, name="W")
        B = topi.arm_cpu.bitserial_conv2d_nhwc(
            A, W, stride, padding, activation_bits, weight_bits, "uint8", out_dtype, unipolar
        )
        if use_relu:
            B = topi.nn.relu(B)
        s = topi.arm_cpu.schedule_bitserial_conv2d_nhwc([B])

    func = tvm.build(s, [A, W, B], device)

    assembly = func.get_source("asm")
    matches = re.findall("vpadal", assembly)
    assert len(matches) > 0
    matches = re.findall("vcnt", assembly)
    assert len(matches) > 0
    matches = re.findall("vpadd", assembly)
    assert len(matches) > 0

    dev = tvm.device(device, 0)
    if "arm" not in os.uname()[4]:
        print("Skipped running code, not an arm device")
        return

    print("Running on target: %s" % device)

    def get_ref_data():
        a_np = generate_quantized_np(get_const_tuple(A.shape), activation_bits, input_type)
        w_np = generate_quantized_np(get_const_tuple(W.shape), weight_bits, input_type)
        if unipolar:
            w_ = np.copy(w_np).astype(out_dtype)
            for x in np.nditer(w_, op_flags=["readwrite"]):
                x[...] = 1 if x == 1 else -1
            b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, w_, stride, padding).astype(out_dtype)
        else:
            b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, w_np, stride, padding).astype(
                out_dtype
            )
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
    func = tvm.build(s, [A, W, B], device)

    func(a, w, b)
    np.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)


def test_bitserial_conv2d():
    in_size = 56
    ic, oc = 64, 64
    k = 3
    stride = 1
    pad = 1

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, False)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, False)

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 1, 1, True)
    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, True)

    verify_bitserial_conv2d_nhwc(1, in_size, ic, oc, k, stride, pad, 2, 1, True, True)


if __name__ == "__main__":
    test_bitserial_conv2d()
