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
"""Test code for FIFO buffer"""

import tvm
import topi
import numpy as np
from common import get_all_backend
from tvm.contrib.pickle_memoize import memoize

def verify_fifo_buffer(buffer_shape, data_shape, axis, dtype='float32'):
    buffer = tvm.placeholder(buffer_shape, name='buffer', dtype=dtype)
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)

    # Use memoize, pickle the test data for next time use
    @memoize('topi.tests.test_fifo_buffer')
    def get_ref_data():
        buffer_np = np.random.uniform(size=buffer_shape).astype(dtype)
        data_np = np.random.uniform(size=data_shape).astype(dtype)

        # Reference implementation of FIFO queue
        begin = data_np.shape[axis]
        end = buffer_np.shape[axis] + data_np.shape[axis]
        ndim = len(buffer_np.shape)
        ss = tuple((slice(begin, end, 1) if x == axis else slice(None)) for x in range(ndim))
        out_np = np.concatenate((buffer_np, data_np), axis=axis)[ss]
        return (buffer_np, data_np, out_np)

    # Get the test data
    buffer_np, data_np, out_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print('  Skip because %s is not enabled' % device)
            return
        print('  Running on target: {}'.format(device))

        with tvm.target.create(device):
            out = topi.nn.fifo_buffer(data, buffer, axis=axis)
            s = topi.generic.schedule_injective([out])

        buffer_tvm = tvm.nd.array(buffer_np, ctx=ctx)
        data_tvm = tvm.nd.array(data_np, ctx=ctx)
        out_tvm = tvm.nd.empty(shape=buffer_shape, ctx=ctx, dtype=dtype)
        f = tvm.build(s, [data, buffer, out], device, name='fifo')
        f(data_tvm, buffer_tvm, out_tvm)
        tvm.testing.assert_allclose(out_tvm.asnumpy(), out_np)

    for device in get_all_backend():
        check_device(device)

def test_fifo_buffer():
    for ndim in [1, 2, 3, 4, 5, 6]:
        for axis in range(ndim):
            buffer_shape = tuple(7 for _ in range(ndim))
            data_shape = tuple((2 if i == axis else 7) for i in range(ndim))
            print('Testing FIFO buffer op: buffer_shape = {}, data_shape = {}, axis = {}'
                  .format(buffer_shape, data_shape, axis))
            verify_fifo_buffer(buffer_shape, data_shape, axis)

if __name__ == '__main__':
    test_fifo_buffer()
