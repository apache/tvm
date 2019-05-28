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
"""Test code for upsampling"""
import numpy as np
import tvm
import topi
import topi.testing
import math

from common import get_all_backend

def verify_upsampling(batch, in_channel, in_height, in_width, scale, layout='NCHW', method="NEAREST_NEIGHBOR"):


    if layout == 'NCHW':
        A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
        dtype = A.dtype
        out_shape = (batch, in_channel, in_height*scale, in_width*scale)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout == 'NHWC':
        A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
        dtype = A.dtype
        out_shape = (batch, in_height*scale, in_width*scale, in_channel)
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError(
            'Layout not supported {} '.format(layout))

    B = topi.nn.upsampling(A, scale, layout=layout, method=method)

    if method == "BILINEAR":
        out_size = (in_height*scale, in_width*scale)
        b_np = topi.testing.bilinear_resize_python(a_np, out_size, layout)
    else:
        b_np = topi.testing.upsampling_python(a_np, (scale, scale), layout)

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

        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5, atol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_upsampling():
    # NEAREST_NEIGHBOR - NCHW
    verify_upsampling(8, 16, 32, 32, 2)
    verify_upsampling(2, 32, 64, 64, 3)

    # NEAREST_NEIGHBOR - NHWC
    verify_upsampling(8, 16, 32, 32, 2, layout="NHWC")
    verify_upsampling(2, 32, 64, 64, 3, layout="NHWC")

    # BILINEAR - NCHW
    verify_upsampling(2, 2, 32, 32, 2, method="BILINEAR")
    verify_upsampling(2, 2, 32, 32, 3, method="BILINEAR")

    # BILINEAR - NHWC
    verify_upsampling(2, 2, 32, 32, 2, layout="NHWC", method="BILINEAR")
    verify_upsampling(2, 2, 32, 32, 3, layout="NHWC", method="BILINEAR")

if __name__ == "__main__":
    test_upsampling()
