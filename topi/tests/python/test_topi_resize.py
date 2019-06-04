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
"""Test code for bilinear scale """
import numpy as np
import tvm
import topi
import topi.testing
import math

from common import get_all_backend

def verify_resize(batch, in_channel, in_height, in_width, out_height, out_width, layout='NCHW', align_corners=False, method="BILINEAR"):
    if layout == 'NCHW':
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A', dtype='float32')
        dtype = A.dtype
        out_shape = (batch, in_channel, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout == 'NHWC':
        A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A', dtype='float32')
        dtype = A.dtype
        out_shape = (batch, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    B = topi.image.resize(A, (out_height, out_width), layout=layout, align_corners=align_corners, method=method)

    if method == "BILINEAR":
        b_np = topi.testing.bilinear_resize_python(a_np, (out_height, out_width), layout, align_corners)
    else:
        scale_h = out_height / in_height
        scale_w = out_width / in_width
        b_np = topi.testing.upsampling_python(a_np, (scale_h, scale_w), layout)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_injective(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)

        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-3, atol=1e-3)

    for device in get_all_backend():
        check_device(device)


def test_resize():
    # Scale NCHW
    verify_resize(4, 16, 32, 32, 50, 50, 'NCHW')
    # Scale NCHW + Align Corners
    verify_resize(6, 32, 64, 64, 20, 20, 'NCHW', True)
    # Scale NHWC
    verify_resize(4, 16, 32, 32, 50, 50, "NHWC")
    # Scale NHWC + Align Corners
    verify_resize(6, 32, 64, 64, 20, 20, "NHWC", True)
    # Nearest + Fractional
    verify_resize(4, 16, 32, 32, 50, 50, 'NCHW', method="NEAREST_NEIGHBOR")
    verify_resize(4, 16, 32, 32, 50, 50, 'NHWC', method="NEAREST_NEIGHBOR")

if __name__ == "__main__":
    test_resize()
