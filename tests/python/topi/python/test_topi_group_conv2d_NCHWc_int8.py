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

"""Test for NCHW[x]c convolution"""

import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple
import pytest


def _transform_data(data, bn):
    # NCHW -> NCHW[x]c
    batch_size, channel, height, width = data.shape
    data = np.reshape(data, (batch_size, channel // bn, bn, height, width))
    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data


def _transform_kernel(kernel, ic_bn, oc_bn):
    # OIHW -> OIHW[x]i[x]o
    out_channel, in_channel, kh, kw = kernel.shape
    kernel = np.reshape(
        kernel, (out_channel // oc_bn, oc_bn, in_channel // ic_bn, ic_bn // 4, kh, kw, 4)
    )
    kernel = np.transpose(kernel, (0, 2, 4, 5, 3, 1, 6))
    return kernel


def verify_group_conv2d_NCHWc_int8(
    batch,
    in_channel,
    groups,
    in_size,
    num_filter,
    kernel,
    stride,
    padding,
    dilation=1,
    add_bias=False,
    add_relu=False,
    dtype="int32",
):
    assert dilation == 1, "conv2d_NCHWc does not support dilation for now."
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, groups, in_size, num_filter, kernel, stride, padding)
    )

    in_height = in_width = in_size

    # for testing functionality,
    # we choose arbitrary block size that can divide the channel,
    # regardless of the performance.
    oc_block = 1
    for bn in range(16, 0, -1):
        if num_filter % bn == 0:
            oc_block = bn
            break

    ic_block = 8
    autotvm.GLOBAL_SCOPE.silent = True
    A = te.placeholder(
        (batch, in_channel // ic_block, in_height, in_width, ic_block), name="A", dtype="uint8"
    )
    W = te.placeholder(
        (
            num_filter // oc_block,
            in_channel // ic_block // groups,
            kernel,
            kernel,
            ic_block // 4,
            oc_block,
            4,
        ),
        name="W",
        dtype="int8",
    )

    @memoize("topi.tests.test_topi_conv2d_NCHWc_int8.verify_conv2d_NCHWc_int8")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype("uint8")
        w_np = np.random.uniform(size=(num_filter, in_channel // groups, kernel, kernel)).astype(
            "int8"
        )
        c_np = tvm.topi.testing.conv2d_nchw_python(a_np, w_np, stride, padding, groups)
        return (
            _transform_data(a_np, ic_block),
            _transform_kernel(w_np, ic_block, oc_block),
            _transform_data(c_np, oc_block),
        )

    a_np, w_np, c_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(dev):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            C = topi.x86.conv2d_NCHWc(
                A,
                W,
                (stride, stride),
                (padding, padding),
                (dilation, dilation),
                "NCHW%dc" % ic_block,
                "NCHW%dc" % oc_block,
                dtype,
            )
            s = topi.x86.schedule_conv2d_NCHWc([C])

        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)
        func = tvm.build(
            s,
            [A, W, C],
            device,
            name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
            % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation),
        )
        # print(tvm.lower(s, [A, W, C], simple_mode=True))
        func(a, w, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    # for device in ["llvm"]:
    for device in ["llvm -mcpu=skylake-avx512"]:
        with autotvm.tophub.context(device):  # load tophub pre-tuned parameters
            check_device(device)
    autotvm.GLOBAL_SCOPE.silent = False


@tvm.testing.uses_gpu
@pytest.mark.skip
def test_conv2d_NCHWc():
    # ResNet50 workloads
    verify_group_conv2d_NCHWc_int8(1, 256, 32, 224, 64, 7, 2, 3)


if __name__ == "__main__":
    # The test requires Skylake and newer Intel machines to generate the correct
    # instruction. This test directly calls the topi operator, requiring correct
    # kernel shape. For older generation of Intel machines, the kernel needs to
    # be 6D. This test tests 7D kernel, that can only work on Skylake+ machines.
    # So, disabling the test.

    # test_conv2d_NCHWc()
    pass
