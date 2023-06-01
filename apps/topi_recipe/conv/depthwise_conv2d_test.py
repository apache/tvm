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
import os

import numpy as np
import tvm
from scipy import signal
from tvm import te, topi
from tvm.contrib import nvcc
from tvm.topi.cuda.depthwise_conv2d import (
    schedule_depthwise_conv2d_nchw,
    schedule_depthwise_conv2d_nhwc,
)
from tvm.topi.utils import get_const_tuple

TASK = "depthwise_conv2d"
USE_MANUAL_CODE = False


@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    ptx = nvcc.compile_cuda(code, target_format="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code, target):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_depthwise_conv2d_nchw():
    """You may test different settings."""
    batch = 1
    in_channel = 256
    in_height = 96
    in_width = 96

    filter_channel = in_channel
    channel_multiplier = 1
    filter_height = 3
    filter_width = 3

    stride_h = 1
    stride_w = 1

    padding = "SAME"  # or 'VALID'

    # Placeholder
    Input = te.placeholder((batch, in_channel, in_height, in_width), name="Input")
    Filter = te.placeholder(
        (filter_channel, channel_multiplier, filter_height, filter_width), name="Filter"
    )
    Stride = [stride_h, stride_w]
    Scale = te.placeholder((in_channel * channel_multiplier,), name="Scale")
    Shift = te.placeholder((in_channel * channel_multiplier,), name="Shift")
    # Declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nchw(Input, Filter, Stride, padding)
    ScaleShift = topi.nn.scale_shift_nchw(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)
    # Schedule
    s1 = schedule_depthwise_conv2d_nchw(DepthwiseConv2d)
    s2 = schedule_depthwise_conv2d_nchw(ScaleShift)
    s3 = schedule_depthwise_conv2d_nchw(Relu)
    input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
    filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
    scale_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Scale.dtype)
    shift_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Shift.dtype)

    def check_device(device):
        if not tvm.runtime.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        dev = tvm.device(device, 0)
        # Build the kernel
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)
        # Prepare data
        input_tvm = tvm.nd.array(input_np, dev)
        filter_tvm = tvm.nd.array(filter_np, dev)
        scale_tvm = tvm.nd.array(scale_np, dev)
        shift_tvm = tvm.nd.array(shift_np, dev)

        depthwise_conv2d_tvm = tvm.nd.array(
            np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), dev
        )
        scale_shift_tvm = tvm.nd.array(
            np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), dev
        )
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), dev)
        # Measure time cost of kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, dev, number=1000)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # Measure time cost of kernel 2 (depthwise_conv2d + scale_shift)
        timer_2 = f2.time_evaluator(f2.entry_name, dev, number=1000)
        tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # Measure time cost of kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer_3 = f3.time_evaluator(f3.entry_name, dev, number=1000)
        tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        print("Input shape = " + str(get_const_tuple(Input.shape)))
        print("Filter shape = " + str(get_const_tuple(Filter.shape)))
        print("Stride = (%d, %d)" % (stride_h, stride_w))
        print("padding = %s\n" % padding)
        print("Output shape = " + str(get_const_tuple(DepthwiseConv2d.shape)))
        print("average time cost of 1000 runs (depthwise_conv2d) = %g us" % (tcost_1 * 1e6))
        print(
            "average time cost of 1000 runs (depthwise_conv2d + scale_shift) = %g us"
            % (tcost_2 * 1e6)
        )
        print(
            "average time cost of 1000 runs (depthwise_conv2d + scale_shift + relu) = %g us"
            % (tcost_3 * 1e6)
        )
        # correctness
        depthwise_conv2d_scipy = tvm.topi.testing.depthwise_conv2d_python_nchw(
            input_np, filter_np, stride=[stride_h, stride_w], padding=padding
        )
        scale_shift_scipy = np.zeros(shape=get_const_tuple(ScaleShift.shape))
        for c in range(in_channel * channel_multiplier):
            scale_shift_scipy[:, c, :, :] = (
                depthwise_conv2d_scipy[:, c, :, :] * scale_np[c] + shift_np[c]
            )
        relu_scipy = np.maximum(scale_shift_scipy, 0)
        tvm.testing.assert_allclose(depthwise_conv2d_tvm.numpy(), depthwise_conv2d_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(scale_shift_tvm.numpy(), scale_shift_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(relu_tvm.numpy(), relu_scipy, rtol=1e-5)
        print("success")

    for device in ["cuda", "opencl", "rocm"]:
        with tvm.transform.PassContext(
            config={"tir.UnrollLoop": {"auto_max_step": 128, "explicit_unroll": device != "rocm"}}
        ):
            check_device(device)


def test_depthwise_conv2d_nhwc():
    """You may test different settings."""
    batch = 1
    in_channel = 256
    in_height = 96
    in_width = 96

    filter_channel = in_channel
    channel_multiplier = 1
    filter_height = 3
    filter_width = 3

    stride_h = 1
    stride_w = 1

    padding = "SAME"  # or 'VALID'

    # Placeholder
    Input = te.placeholder((batch, in_height, in_width, in_channel), name="Input")
    Filter = te.placeholder(
        (filter_height, filter_width, filter_channel, channel_multiplier), name="Filter"
    )
    Stride = [stride_h, stride_w]
    Scale = te.placeholder((in_channel * channel_multiplier,), name="Scale")
    Shift = te.placeholder((in_channel * channel_multiplier,), name="Shift")
    # Declare
    DepthwiseConv2d = topi.nn.depthwise_conv2d_nhwc(Input, Filter, Stride, padding)
    ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
    Relu = topi.nn.relu(ScaleShift)
    # Schedule
    s1 = schedule_depthwise_conv2d_nhwc(DepthwiseConv2d)
    s2 = schedule_depthwise_conv2d_nhwc(ScaleShift)
    s3 = schedule_depthwise_conv2d_nhwc(Relu)

    input_np = np.random.uniform(size=get_const_tuple(Input.shape)).astype(Input.dtype)
    filter_np = np.random.uniform(size=get_const_tuple(Filter.shape)).astype(Filter.dtype)
    scale_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Scale.dtype)
    shift_np = np.random.uniform(size=(in_channel * channel_multiplier)).astype(Shift.dtype)

    def check_device(device):
        if not tvm.runtime.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        dev = tvm.device(device, 0)
        # Build the kernel
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], device)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], device)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], device)
        # Prepare data
        input_tvm = tvm.nd.array(input_np, dev)
        filter_tvm = tvm.nd.array(filter_np, dev)
        scale_tvm = tvm.nd.array(scale_np, dev)
        shift_tvm = tvm.nd.array(shift_np, dev)
        depthwise_conv2d_tvm = tvm.nd.array(
            np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), dev
        )
        scale_shift_tvm = tvm.nd.array(
            np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), dev
        )
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), dev)
        # Measure time cost of kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, dev, number=1000)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # Measure time cost of kernel 2 (depthwise_conv2d + scale_shift)
        timer_2 = f2.time_evaluator(f2.entry_name, dev, number=1000)
        tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # Measure time cost of kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer_3 = f3.time_evaluator(f3.entry_name, dev, number=1000)
        tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        print("Input shape = " + str(get_const_tuple(Input.shape)))
        print("Filter shape = " + str(get_const_tuple(Filter.shape)))
        print("Stride = (%d, %d)" % (stride_h, stride_w))
        print("padding = %s\n" % padding)
        print("Output shape = " + str(get_const_tuple(DepthwiseConv2d.shape)))
        print("average time cost of 1000 runs (depthwise_conv2d) = %g us" % (tcost_1 * 1e6))
        print(
            "average time cost of 1000 runs (depthwise_conv2d + scale_shift) = %g us"
            % (tcost_2 * 1e6)
        )
        print(
            "average time cost of 1000 runs (depthwise_conv2d + scale_shift + relu) = %g us"
            % (tcost_3 * 1e6)
        )
        # correctness
        depthwise_conv2d_scipy = tvm.topi.testing.depthwise_conv2d_python_nhwc(
            input_np, filter_np, stride=[stride_h, stride_w], padding=padding
        )
        scale_shift_scipy = np.zeros(shape=get_const_tuple(ScaleShift.shape))
        for c in range(in_channel * channel_multiplier):
            scale_shift_scipy[:, :, :, c] = (
                depthwise_conv2d_scipy[:, :, :, c] * scale_np[c] + shift_np[c]
            )
        relu_scipy = np.maximum(scale_shift_scipy, 0)
        tvm.testing.assert_allclose(depthwise_conv2d_tvm.numpy(), depthwise_conv2d_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(scale_shift_tvm.numpy(), scale_shift_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(relu_tvm.numpy(), relu_scipy, rtol=1e-5)
        print("success")

    for device in ["cuda", "opencl", "rocm"]:
        with tvm.transform.PassContext(
            config={"tir.UnrollLoop": {"auto_max_step": 128, "explicit_unroll": device != "cuda"}}
        ):
            check_device(device)


if __name__ == "__main__":
    test_depthwise_conv2d_nchw()
    test_depthwise_conv2d_nhwc()
