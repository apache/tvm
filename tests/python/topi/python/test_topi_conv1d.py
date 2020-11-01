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
"""Test code for transposed convolution."""
import numpy as np
import itertools
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.utils import get_const_tuple


_conv1d_ncw_implement = {
    "generic": (topi.nn.conv1d_ncw, topi.generic.schedule_conv1d_ncw),
    "cpu": (topi.nn.conv1d_ncw, topi.x86.schedule_conv1d_ncw),
    "gpu": (topi.cuda.conv1d_ncw, topi.cuda.schedule_conv1d_ncw),
}

_conv1d_nwc_implement = {
    "generic": (topi.nn.conv1d_nwc, topi.generic.schedule_conv1d_nwc),
    "cpu": (topi.nn.conv1d_nwc, topi.x86.schedule_conv1d_nwc),
    "gpu": (topi.cuda.conv1d_nwc, topi.cuda.schedule_conv1d_nwc),
}


def verify_conv1d(
    batch,
    in_channels,
    in_width,
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    padding="VALID",
    layout="NCW",
):
    if layout == "NCW":
        in_shape = [batch, in_channels, in_width]
        kernel_shape = [filters, in_channels, kernel_size]
    else:
        in_shape = [batch, in_width, in_channels]
        kernel_shape = [kernel_size, in_channels, filters]

    dtype = "float32"
    A = te.placeholder(in_shape, name="A", dtype=dtype)
    W = te.placeholder(kernel_shape, name="W", dtype=dtype)

    def get_ref_data(layout):
        a_np = np.random.uniform(size=in_shape).astype(dtype)
        w_np = np.random.uniform(size=kernel_shape).astype(dtype)
        if layout == "NWC":
            np_in = np.transpose(a_np, [0, 2, 1])
            np_w = np.transpose(w_np, [2, 1, 0])
        else:
            np_in = a_np
            np_w = w_np
        b_np = tvm.topi.testing.conv1d_ncw_python(np_in, np_w, stride, padding, dilation)
        if layout == "NWC":
            b_np = np.transpose(b_np, [0, 2, 1])
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data(layout)

    def check_device(device, ctx):
        if layout == "NCW":
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _conv1d_ncw_implement)
        else:
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _conv1d_nwc_implement)
        with tvm.target.Target(device):
            B = fcompute(A, W, stride, padding, dilation, "float32")
            s = fschedule([B])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)

        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_conv1d():
    for layout in ["NCW", "NWC"]:
        # Most basic test case
        verify_conv1d(1, 1, 8, 1, 3, 1, 1, "VALID", layout)
        # With padding
        verify_conv1d(1, 1, 8, 1, 3, 1, 1, "SAME", layout)
        # Realistic dimensions
        verify_conv1d(1, 16, 32, 16, 3, 1, 1, "SAME", layout)
        # With stride
        verify_conv1d(1, 16, 32, 16, 3, 2, 1, "SAME", layout)
        # With dilation
        verify_conv1d(1, 16, 32, 16, 3, 1, 2, "SAME", layout)
        # Large batch size
        verify_conv1d(8, 16, 32, 16, 3, 1, 1, "SAME", layout)
        # Other kernel sizes
        verify_conv1d(1, 16, 32, 16, 3, 1, 1, "SAME", layout)
        verify_conv1d(1, 16, 32, 16, 2, 1, 1, "SAME", layout)
        verify_conv1d(1, 16, 32, 16, 1, 1, 1, "SAME", layout)
        # Non-power-of-two shape
        verify_conv1d(1, 17, 12, 21, 3, 1, 1, "SAME", layout)
        verify_conv1d(1, 5, 27, 18, 3, 1, 1, "VALID", layout)


if __name__ == "__main__":
    test_conv1d()
