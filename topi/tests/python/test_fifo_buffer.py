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

def verify_conv1d_integration():
    batch_size = 1
    num_channel = 1
    num_filter = 1

    # Note: TVM doesn't have a separate op for 1D convolution, so we use conv2d instead.
    # We set height=1 to indicate that convolution is really 1D.
    stride = (1, 1)
    dilate = (1, 1)
    padding = (0, 0)

    kernel_size = (1, 3)
    input_window_size = (1, 10)
    inc_input_size = (1, 2)
    context_size = (1, 4)
    inc_output_size = (1, 2)
    output_window_size = (1, 8)

    num_iteration = 20
    buffer_axis = 3

    kernel_shape = (num_filter, num_channel, kernel_size[0], kernel_size[1])
    input_window_shape = (batch_size, num_channel, input_window_size[0], input_window_size[1])
    inc_input_shape = (batch_size, num_channel, inc_input_size[0], inc_input_size[1])
    inc_output_shape = (batch_size, num_filter, inc_output_size[0], inc_output_size[1])
    context_shape = (batch_size, num_channel, context_size[0], context_size[1])
    output_window_shape = (batch_size, num_filter, output_window_size[0], output_window_size[1])
    # Rule: Convolution of Tensor[context_shape] and Tensor[kernel_shape]
    #       produces Tensor[inc_input_shape]

    dtype = 'float32'

    inc_input = tvm.placeholder(inc_input_shape, name='inc_input', dtype=dtype)
    input_window = tvm.placeholder(input_window_shape, name='input_window', dtype=dtype)
    context = tvm.placeholder(context_shape, name='context', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    inc_output = tvm.placeholder(inc_input_shape, name='inc_output', dtype=dtype)
    output_window = tvm.placeholder(output_window_shape, name='output_window', dtype=dtype)

    # Use memoize, pickle the test data for next time use
    @memoize('topi.tests.test_fifo_buffer_conv1d_integration')
    def get_data():
        # Generate [num_iteration] slices of input
        inc_input_np = np.random.uniform(size=tuple([num_iteration] + list(inc_input_shape)))\
                       .astype(dtype)
        input_window_np = np.zeros(input_window_shape, dtype=dtype)
        kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
        context_np = np.zeros(context_shape, dtype=dtype)
        output_window_np = np.zeros(output_window_shape, dtype=dtype)

        return (inc_input_np, input_window_np, kernel_np, context_np, output_window_np)

    # Get the test data
    inc_input_np, input_window_np, kernel_np, context_np, output_window_np = get_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print('  Skip because %s is not enabled' % device)
            return
        print('  Running on target: {}'.format(device))

        with tvm.target.create(device):
            out = topi.nn.fifo_buffer(inc_input, context, axis=buffer_axis)
            s = topi.generic.schedule_injective([out])
            update_context = tvm.build(s, [inc_input, context, out], device, name='update_context')

            out = topi.nn.conv2d(context, kernel, strides=stride, padding=padding, dilation=dilate,
                                 layout='NCHW', out_dtype=dtype)
            s = topi.generic.schedule_conv2d_nchw([out])
            conv2d_inc = tvm.build(s, [context, kernel, out], device, name='conv2d_inc')

            out = topi.nn.fifo_buffer(inc_output, output_window, axis=buffer_axis)
            s = topi.generic.schedule_injective([out])
            update_output_window = tvm.build(s, [inc_output, output_window, out], device,
                 name='update_output_window')

            out = topi.nn.fifo_buffer(inc_input, input_window, axis=buffer_axis)
            s = topi.generic.schedule_injective([out])
            update_input_window = tvm.build(s, [inc_input, input_window, out], device,
                                            name='update_input_window')

            out = topi.nn.conv2d(input_window, kernel, strides=stride, padding=padding,
                                 dilation=dilate, layout='NCHW', out_dtype=dtype)
            s = topi.generic.schedule_conv2d_nchw([out])
            conv2d = tvm.build(s, [input_window, kernel, out], device, name='conv2d')

        input_window_tvm = tvm.nd.array(input_window_np, ctx=ctx)
        new_input_window_tvm = tvm.nd.empty(shape=input_window_shape, ctx=ctx, dtype=dtype)
        kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
        context_tvm = tvm.nd.array(context_np, ctx=ctx)
        new_context_tvm = tvm.nd.empty(shape=context_shape, ctx=ctx, dtype=dtype)
        inc_output_tvm = tvm.nd.empty(shape=inc_output_shape, ctx=ctx, dtype=dtype)
        output_window_tvm = tvm.nd.array(output_window_np, ctx=ctx)
        new_output_window_tvm = tvm.nd.empty(shape=output_window_shape, ctx=ctx, dtype=dtype)
        output_window_ref_tvm = tvm.nd.empty(shape=output_window_shape, ctx=ctx, dtype=dtype)

        for i in range(num_iteration):
            # Take i-th slice of inc_input_np
            inc_input_tvm = tvm.nd.array(inc_input_np[i], ctx=ctx)

            # Compute new output window incrementally, using the FIFO buffer op
            update_context(inc_input_tvm, context_tvm, new_context_tvm)
            conv2d_inc(new_context_tvm, kernel_tvm, inc_output_tvm)
            update_output_window(inc_output_tvm, output_window_tvm, new_output_window_tvm)
            context_tvm = new_context_tvm
            output_window_tvm = new_output_window_tvm

            # Compute full input window, so that we have a baseline
            update_input_window(inc_input_tvm, input_window_tvm, new_input_window_tvm)
            input_window_tvm = new_input_window_tvm
            conv2d(input_window_tvm, kernel_tvm, output_window_ref_tvm)
            # Incrementally updating the output window should be equivalent to computing it from
            # scratch using the input window
            tvm.testing.assert_allclose(output_window_tvm.asnumpy(),
                                        output_window_ref_tvm.asnumpy())

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

def test_conv1d_integration():
    print('Testing FIFO buffer with 1D convolution')
    verify_conv1d_integration()

if __name__ == '__main__':
    test_fifo_buffer()
    test_conv1d_integration()
