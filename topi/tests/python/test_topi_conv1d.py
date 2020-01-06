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
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from common import get_all_backend


def verify_conv1d(batch,
                  in_channels,
                  in_width,
                  filters,
                  kernel_size=3,
                  stride=1,
                  dilation=1,
                  padding='VALID',
                  pad_method='SYMMETRIC',
                  layout='NCW'):
    if layout == 'NCW':
        in_shape = [batch, in_channels, in_width]
        kernel_shape = [filters, in_channels, kernel_size]
    else:
        in_shape = [batch, in_width, in_channels]
        kernel_shape = [kernel_size, in_channels, filters]

    dtype = 'float32'
    A = tvm.placeholder(in_shape, name='A', dtype=dtype)
    W = tvm.placeholder(kernel_shape, name='W', dtype=dtype)

    def get_ref_data(layout):
        a_np = np.random.uniform(size=in_shape).astype(dtype)
        w_np = np.random.uniform(size=kernel_shape).astype(dtype)
        if layout == 'NWC':
            np_in = np.transpose(a_np, [0, 2, 1])
            np_w = np.transpose(w_np, [2, 1, 0])
        else:
            np_in = a_np
            np_w = w_np
        b_np = topi.testing.conv1d_ncw_python(np_in, np_w, stride, padding,
                                              dilation, pad_method)
        if layout == 'NWC':
            b_np = np.transpose(b_np, [0, 2, 1])
        return a_np, w_np, b_np

    a_np, w_np, b_np = get_ref_data(layout)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        with tvm.target.create(device):
            B = topi.nn.conv1d(A, W, stride, padding, dilation, layout,
                               pad_method, 'float32')
            if layout == 'NCW':
                s = topi.generic.schedule_conv1d_ncw([B])
            else:
                s = topi.generic.schedule_conv1d_nwc([B])

        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)

        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm', 'cuda']:#get_all_backend():
        check_device(device)


def test_conv1d_transpose():
    verify_conv1d(1, 1, 8, 1, 3, 1, 1, 'VALID', 'SYMMETRIC', "NCW")
    verify_conv1d(1, 1, 8, 1, 3, 1, 1, 'VALID', 'SYMMETRIC', "NWC")


if __name__ == "__main__":
    test_conv1d_transpose()
