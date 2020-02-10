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
"""Test code for depth to space"""
import numpy as np
import tvm
import topi
import topi.testing

from common import get_all_backend


def verify_depth_to_space(block_size, batch, in_channel, in_height, in_width, layout='NCHW', mode='DCR'):
    out_channel = int(in_channel / (block_size * block_size))
    out_height = int(in_height * block_size)
    out_width = int(in_width * block_size)

    if layout == 'NCHW':
        in_shape = [batch, in_channel, in_height, in_width]
        out_shape = [batch, out_channel, out_height, out_width]
    elif layout == 'NHWC':
        in_shape = [batch, in_height, in_width, in_channel]
        out_shape = [batch, out_height, out_width, out_channel]
    else:
        raise NotImplementedError('Layout not supported {}'.format(layout))

    A = tvm.placeholder(in_shape, name='A', dtype='float32')
    dtype = A.dtype
    a_np = np.random.uniform(size=in_shape).astype(dtype)

    B = topi.nn.depth_to_space(A, block_size=block_size, layout=layout, mode=mode)
    if layout == 'NHWC':
        a_np = np.transpose(a_np, axes=[0, 3, 1, 2])
    b_np = topi.testing.depth_to_space_python(a_np, block_size, mode=mode)
    if layout == 'NHWC':
        a_np = np.transpose(a_np, axes=[0, 2, 3, 1])
        b_np = np.transpose(b_np, axes=[0, 2, 3, 1])

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


def test_depth_to_space():
    for layout in ['NCHW', 'NHWC']:
        for mode in ['DCR', 'CDR']:
            # Simplest possible case
            verify_depth_to_space(2, 1, 4, 1, 1, layout=layout, mode=mode)
            # Average input size
            verify_depth_to_space(2, 1, 32, 32, 32, layout=layout, mode=mode)
            # Large block size
            verify_depth_to_space(8, 1, 256, 32, 32, layout=layout, mode=mode)
            # Large batch size
            verify_depth_to_space(4, 8, 32, 32, 32, layout=layout, mode=mode)
            # Large input size
            verify_depth_to_space(4, 8, 32, 128, 128, layout=layout, mode=mode)


if __name__ == "__main__":
    test_depth_to_space()
