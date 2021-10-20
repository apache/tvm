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
import pytest
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm import topi
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple


def verify_conv2d_1x1_nhwc_pack_int8(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1
):
    in_height = in_width = in_size

    A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="uint8")
    W = te.placeholder((kernel, kernel, in_channel, num_filter), name="W", dtype="int8")

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    adtype = A.dtype
    wdtype = W.dtype

    @memoize("topi.tests.test_topi_conv2d_1x1_nhwc_pack_int8.verify_nhwc.v2")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(adtype)
        w_np = np.random.uniform(size=w_shape).astype(wdtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (dilation, dilation, 1, 1))
        b_np = tvm.topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.Target(device):
            B = topi.nn.conv2d(A, W, stride, padding, dilation, layout="NHWC", out_dtype="int32")
            s = topi.x86.schedule_conv2d_nhwc_pack_int8([B])
        a = tvm.nd.array(a_np, dev)
        w = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), dev)
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-5)

    # for device in ['llvm -mcpu=skylake-avx512']:
    for device in ["llvm"]:
        check_device(device)


# TODO(@llyfacebook): Please fix https://github.com/apache/tvm/issues/4122 to enable this test.
@pytest.mark.skip
def test_conv2d_nhwc():
    verify_conv2d_1x1_nhwc_pack_int8(1, 256, 32, 256, 1, 1, 0)


if __name__ == "__main__":
    # test_conv2d_nhwc()
    pass
