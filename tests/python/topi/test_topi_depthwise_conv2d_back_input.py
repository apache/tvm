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
import tvm
from tvm import te
from tvm import topi
import numpy as np
from tvm.contrib.pickle_memoize import memoize
from scipy import signal
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple
import tvm.topi.testing
from tvm.topi.cuda.depthwise_conv2d import schedule_depthwise_conv2d_backward_input_nhwc
import tvm.testing


def verify_depthwise_conv2d_back_input(
    batch, in_channel, in_h, channel_multiplier, filter_h, stride_h, padding_h
):
    in_w = in_h
    filter_channel = in_channel
    filter_w = filter_h
    stride_w = stride_h
    padding_w = padding_h

    out_h = np.int32((in_h + 2 * padding_h - filter_h) / stride_h + 1)
    out_w = np.int32((in_w + 2 * padding_w - filter_w) / stride_w + 1)
    out_channel = in_channel * channel_multiplier

    ishape = [batch, in_h, in_w, in_channel]
    oshape = [batch, out_h, out_w, out_channel]

    # placeholder
    Out_grad = te.placeholder(oshape, name="Out_grad")
    Filter = te.placeholder((filter_h, filter_w, filter_channel, channel_multiplier))
    # declare
    In_grad = topi.nn.depthwise_conv2d_backward_input_nhwc(
        Filter,
        Out_grad,
        oshape,
        ishape,
        stride=[stride_h, stride_w],
        padding=[padding_h, padding_w],
    )
    # schedule
    schedule = schedule_depthwise_conv2d_backward_input_nhwc(In_grad)

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        # build the kernel
        f = tvm.build(schedule, [Filter, Out_grad, In_grad], device)
        # prepare pod type for test data closure
        dtype = Out_grad.dtype
        out_grad_shape = get_const_tuple(Out_grad.shape)
        filter_shape = get_const_tuple(Filter.shape)

        # use memoize to pickle the test data for next time use
        @memoize("topi.tests.test_topi_depthwise_conv2d_backward_input.nhwc")
        def get_ref_data():
            out_grad_np = np.random.uniform(size=out_grad_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            dilated_out_grad_np = tvm.topi.testing.dilate_python(
                out_grad_np, [1, stride_h, stride_w, 1]
            )
            # padding params in forward propagation
            fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
                [padding_h, padding_w], (filter_h, filter_w)
            )
            # padding params in backward propagation
            bpad_top = filter_h - 1 - fpad_top
            bpad_bottom = (filter_h - 1 - fpad_bottom) + (stride_h - 1)
            bpad_left = filter_w - 1 - fpad_left
            bpad_right = (filter_w - 1 - fpad_right) + (stride_w - 1)

            padded_out_grad = np.zeros(
                (
                    batch,
                    dilated_out_grad_np.shape[1] + bpad_top + bpad_bottom,
                    dilated_out_grad_np.shape[2] + bpad_left + bpad_right,
                    out_channel,
                )
            )
            padded_out_grad[
                :,
                bpad_top : dilated_out_grad_np.shape[1] + bpad_top,
                bpad_left : dilated_out_grad_np.shape[2] + bpad_left,
                :,
            ] = dilated_out_grad_np

            in_grad_np = np.zeros((batch, in_h, in_w, in_channel))
            for b in range(batch):
                for c in range(in_channel):
                    for m in range(channel_multiplier):
                        in_grad_np[b, :, :, c] += signal.convolve2d(
                            padded_out_grad[b, :, :, c * channel_multiplier + m],
                            filter_np[:, :, c, m],
                            mode="valid",
                        )[0:in_h, 0:in_w]
            return (out_grad_np, filter_np, in_grad_np)

        (out_grad_np, filter_np, in_grad_np) = get_ref_data()

        out_grad_tvm = tvm.nd.array(out_grad_np, dev)
        filter_tvm = tvm.nd.array(filter_np, dev)
        in_grad_tvm = tvm.nd.array(np.zeros(shape=ishape, dtype=dtype), dev)
        # launch the kernel
        timer = f.time_evaluator(f.entry_name, dev, number=1)
        tcost = timer(filter_tvm, out_grad_tvm, in_grad_tvm).mean
        tvm.testing.assert_allclose(in_grad_np, in_grad_tvm.numpy(), rtol=1e-5)

    check_device("opencl")
    check_device("cuda")
    check_device("metal")
    check_device("rocm")
    check_device("vulkan")
    check_device("nvptx")


@tvm.testing.requires_gpu
def test_topi_depthwise_conv2d_backward_input_nhwc():
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 3, 1, 1)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 3, 1, 1)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 5, 1, 2)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 5, 1, 2)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 3, 2, 1)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 3, 2, 1)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 5, 2, 2)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 5, 2, 2)

    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 3, 1, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 3, 1, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 5, 1, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 5, 1, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 3, 2, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 3, 2, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 1, 5, 2, 0)
    verify_depthwise_conv2d_back_input(16, 256, 56, 2, 5, 2, 0)


if __name__ == "__main__":
    test_topi_depthwise_conv2d_backward_input_nhwc()
