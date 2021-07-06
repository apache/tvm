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
from tvm import autotvm
from tvm import topi
import tvm.topi.testing
import numpy as np
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.depthwise_conv2d import _get_workload
from tvm.topi.x86.depthwise_conv2d import _fallback_schedule

import tvm.testing

_depthwise_conv2d_nchw_implement = {
    "generic": [(topi.nn.depthwise_conv2d_nchw, topi.generic.schedule_depthwise_conv2d_nchw)],
    "arm_cpu": [
        (topi.arm_cpu.depthwise_conv2d_nchw, topi.arm_cpu.schedule_depthwise_conv2d_nchw),
        (
            topi.arm_cpu.depthwise_conv2d_nchw_spatial_pack,
            topi.arm_cpu.schedule_depthwise_conv2d_nchw_spatial_pack,
        ),
    ],
    "gpu": [(topi.cuda.depthwise_conv2d_nchw, topi.cuda.schedule_depthwise_conv2d_nchw)],
    "mali": [(topi.mali.depthwise_conv2d_nchw, topi.mali.schedule_depthwise_conv2d_nchw)],
    "bifrost": [(topi.nn.depthwise_conv2d_nchw, topi.bifrost.schedule_depthwise_conv2d_nchw)],
    "intel_graphics": [
        (
            topi.intel_graphics.depthwise_conv2d_nchw,
            topi.intel_graphics.schedule_depthwise_conv2d_nchw,
        )
    ],
}

_depthwise_conv2d_nhwc_implement = {
    "generic": (topi.nn.depthwise_conv2d_nhwc, topi.generic.schedule_depthwise_conv2d_nhwc),
    "arm_cpu": (
        topi.arm_cpu.compute_depthwise_conv2d_nhwc,
        topi.arm_cpu.schedule_depthwise_conv2d_nhwc,
    ),
    "gpu": (topi.nn.depthwise_conv2d_nhwc, topi.cuda.schedule_depthwise_conv2d_nhwc),
}


def compile_depthwise_NHWC_int8_arm(
    batch,
    in_channel,
    in_size,
    kernel,
    depth_multiplier,
    stride,
    padding,
    add_bias=False,
    dilation=1,
):
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right

    in_height = in_width = in_size
    A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="int16")
    W = te.placeholder((kernel, kernel, in_channel, depth_multiplier), name="W", dtype="int16")
    bias = te.placeholder((in_channel * depth_multiplier,), name="bias", dtype="int32")
    dtype = "int32"

    target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
    compute = topi.arm_cpu.compute_depthwise_conv2d_nhwc
    schedule = topi.arm_cpu.schedule_depthwise_conv2d_nhwc

    if not tvm.testing.device_enabled(target):
        print("Skip because %s is not enabled" % target)
        return

    print("Compiling on arm AArch64 target: %s" % target)
    with tvm.target.Target(target):
        assert topi.arm_cpu.arm_utils.is_aarch64_arm(), "AArch64 target not recognized"

        C = compute(A, W, (stride, stride), padding, (dilation, dilation), dtype)
        if add_bias:
            C += bias
            ins_outs = [A, W, bias, C]
        else:
            ins_outs = [A, W, C]

        s = schedule([C])

        func = tvm.build(
            s,
            ins_outs,
            target,
            name="depthwise_conv2d",
        )


def depthwise_conv2d_with_workload_nchw(
    target,
    dev,
    batch,
    in_channel,
    in_height,
    channel_multiplier,
    filter_height,
    stride,
    padding,
    dilation=1,
):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_h = stride_w = stride

    if dilation == 1:
        # here we transform the padding argument from 'str' to  'tuple' ,
        # because we need this to match the "workload" tuple to the records in TopHub
        padt, padl, padb, padr = get_pad_tuple(padding, (filter_height, filter_width))
        padding_args = (padt, padl, padb, padr)
    else:
        padding_args = padding

    # placeholder
    Input = te.placeholder((batch, in_channel, in_height, in_width), name="Input")
    Filter = te.placeholder(
        (filter_channel, channel_multiplier, filter_height, filter_width), name="Filter"
    )
    Scale = te.placeholder((in_channel * channel_multiplier,), name="Scale")
    Shift = te.placeholder((in_channel * channel_multiplier,), name="Shift")

    dtype = "float32"

    with autotvm.tophub.context(target):  # load tophub pre-tuned parameters
        impl_list = tvm.topi.testing.dispatch(target, _depthwise_conv2d_nchw_implement)[:]
        if target == "llvm" and channel_multiplier == 1 and dilation == 1:
            impl_list.append(
                (topi.x86.depthwise_conv2d_nchw, topi.x86.schedule_depthwise_conv2d_nchw)
            )

        for fcompute, fschedule in impl_list:
            with tvm.target.Target(target):
                # declare
                DepthwiseConv2d = fcompute(
                    Input, Filter, (stride_h, stride_w), padding_args, dilation, dtype
                )
                ScaleShift = topi.nn.scale_shift_nchw(DepthwiseConv2d, Scale, Shift)
                Relu = topi.nn.relu(ScaleShift)
                # schedule
                s1 = fschedule(DepthwiseConv2d)
                s2 = fschedule(ScaleShift)
                s3 = fschedule(Relu)
            # build the kernels
            f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], target)
            f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], target)
            f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], target)

            # Prepare pod type for test data closure
            input_shape = get_const_tuple(Input.shape)
            filter_shape = get_const_tuple(Filter.shape)
            scale_shape = get_const_tuple(Scale.shape)
            shift_shape = get_const_tuple(Shift.shape)
            scale_shift_shape = get_const_tuple(ScaleShift.shape)

            # Use memoize, pickle the test data for next time use.
            @memoize("topi.tests.test_topi_depthwise_conv2d.nchw")
            def get_ref_data():
                input_np = np.random.uniform(size=input_shape).astype(dtype)
                filter_np = np.random.uniform(size=filter_shape).astype(dtype)
                dilated_filter_np = tvm.topi.testing.dilate_python(
                    filter_np, (1, 1, dilation, dilation)
                )
                scale_np = np.random.uniform(size=scale_shape).astype(dtype)
                shift_np = np.random.uniform(size=shift_shape).astype(dtype)
                # correctness with scipy
                depthwise_conv2d_scipy = tvm.topi.testing.depthwise_conv2d_python_nchw(
                    input_np, dilated_filter_np, stride, padding
                )
                scale_shift_scipy = np.zeros(shape=scale_shift_shape)
                for c in range(in_channel * channel_multiplier):
                    scale_shift_scipy[:, c, :, :] = (
                        depthwise_conv2d_scipy[:, c, :, :] * scale_np[c] + shift_np[c]
                    )
                    relu_scipy = np.maximum(scale_shift_scipy, 0)
                return (
                    input_np,
                    filter_np,
                    scale_np,
                    shift_np,
                    depthwise_conv2d_scipy,
                    scale_shift_scipy,
                    relu_scipy,
                )

            # Get the test data
            (
                input_np,
                filter_np,
                scale_np,
                shift_np,
                depthwise_conv2d_scipy,
                scale_shift_scipy,
                relu_scipy,
            ) = get_ref_data()

            def verify_workload_padding():
                _, _, out_height, out_width = get_const_tuple(depthwise_conv2d_scipy.shape)
                wkl = _get_workload(
                    Input, Filter, (stride_h, stride_w), padding_args, dilation, dtype
                )

                # check if tile_ow candidates are the factors of the right output weight.
                with tvm.target.Target(target):
                    cfg = autotvm.get_config()
                    _fallback_schedule(cfg, wkl)
                    ow_tile = np.prod(cfg["tile_ow"].size)

                    tvm.testing.assert_allclose(ow_tile, out_width)

            if "llvm" in target:
                verify_workload_padding()

            input_tvm = tvm.nd.array(input_np, dev)
            filter_tvm = tvm.nd.array(filter_np, dev)
            scale_tvm = tvm.nd.array(scale_np, dev)
            shift_tvm = tvm.nd.array(shift_np, dev)
            depthwise_conv2d_tvm = tvm.nd.array(
                np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype),
                dev,
            )
            scale_shift_tvm = tvm.nd.array(
                np.zeros(shape=get_const_tuple(ScaleShift.shape), dtype=ScaleShift.dtype), dev
            )
            relu_tvm = tvm.nd.array(
                np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), dev
            )
            # launch kernel 1 (depthwise_conv2d)
            timer_1 = f1.time_evaluator(f1.entry_name, dev, number=1)
            tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
            # launch kernel 2 (depthwise_conv2d + scale_shift)
            timer_2 = f2.time_evaluator(f2.entry_name, dev, number=1)
            tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
            # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
            timer_3 = f3.time_evaluator(f3.entry_name, dev, number=1)
            tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
            tvm.testing.assert_allclose(
                depthwise_conv2d_tvm.numpy(), depthwise_conv2d_scipy, rtol=1e-5
            )
            tvm.testing.assert_allclose(scale_shift_tvm.numpy(), scale_shift_scipy, rtol=1e-5)
            tvm.testing.assert_allclose(relu_tvm.numpy(), relu_scipy, rtol=1e-5)


def depthwise_conv2d_with_workload_nhwc(
    target,
    dev,
    batch,
    in_channel,
    in_height,
    channel_multiplier,
    filter_height,
    stride_h,
    padding,
    dilation=1,
):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_w = stride_h

    if dilation == 1:
        # here we transform the padding argument from 'str' to  'tuple' ,
        # because we need this to match the "workload" tuple to the records in TopHub
        pad_h, pad_w, _, _ = get_pad_tuple(padding, (filter_height, filter_width))
        padding_args = (pad_h, pad_w)
    else:
        padding_args = padding

    # placeholder
    Input = te.placeholder((batch, in_height, in_width, in_channel), name="Input")
    Filter = te.placeholder(
        (filter_height, filter_width, filter_channel, channel_multiplier), name="Filter"
    )
    Scale = te.placeholder((in_channel * channel_multiplier,), name="Scale")
    Shift = te.placeholder((in_channel * channel_multiplier,), name="Shift")

    dtype = "float32"

    with autotvm.tophub.context(target):  # load tophub pre-tuned parameters
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _depthwise_conv2d_nhwc_implement)
        with tvm.target.Target(target):
            # declare
            DepthwiseConv2d = fcompute(
                Input, Filter, (stride_h, stride_w), padding_args, dilation, dtype
            )
            ScaleShift = topi.nn.scale_shift_nhwc(DepthwiseConv2d, Scale, Shift)
            Relu = topi.nn.relu(ScaleShift)
            # schedule
            s1 = fschedule(DepthwiseConv2d)
            s2 = fschedule(ScaleShift)
            s3 = fschedule(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], target)
        f2 = tvm.build(s2, [Input, Filter, Scale, Shift, ScaleShift], target)
        f3 = tvm.build(s3, [Input, Filter, Scale, Shift, Relu], target)

        # Prepare pod type for test data closure
        input_shape = get_const_tuple(Input.shape)
        filter_shape = get_const_tuple(Filter.shape)
        scale_shape = get_const_tuple(Scale.shape)
        shift_shape = get_const_tuple(Shift.shape)
        scale_shift_shape = get_const_tuple(ScaleShift.shape)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d.nhwc.v2")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            dilated_filter_np = tvm.topi.testing.dilate_python(
                filter_np, (dilation, dilation, 1, 1)
            )
            scale_np = np.random.uniform(size=scale_shape).astype(dtype)
            shift_np = np.random.uniform(size=shift_shape).astype(dtype)
            # correctness with scipy
            depthwise_conv2d_scipy = tvm.topi.testing.depthwise_conv2d_python_nhwc(
                input_np, dilated_filter_np, stride=[stride_h, stride_w], padding=padding
            )
            scale_shift_scipy = np.zeros(shape=scale_shift_shape)
            for c in range(in_channel * channel_multiplier):
                scale_shift_scipy[:, :, :, c] = (
                    depthwise_conv2d_scipy[:, :, :, c] * scale_np[c] + shift_np[c]
                )
                relu_scipy = np.maximum(scale_shift_scipy, 0)
            return (
                input_np,
                filter_np,
                scale_np,
                shift_np,
                depthwise_conv2d_scipy,
                scale_shift_scipy,
                relu_scipy,
            )

        # Get the test data
        (
            input_np,
            filter_np,
            scale_np,
            shift_np,
            depthwise_conv2d_scipy,
            scale_shift_scipy,
            relu_scipy,
        ) = get_ref_data()

        # prepare data
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
        # launch kernel 1 (depthwise_conv2d)
        timer_1 = f1.time_evaluator(f1.entry_name, dev, number=1)
        tcost_1 = timer_1(input_tvm, filter_tvm, depthwise_conv2d_tvm).mean
        # launch kernel 2 (depthwise_conv2d + scale_shift)
        timer_2 = f2.time_evaluator(f2.entry_name, dev, number=1)
        tcost_2 = timer_2(input_tvm, filter_tvm, scale_tvm, shift_tvm, scale_shift_tvm).mean
        # launch kernel 3 (depthwise_conv2d + scale_shift + relu)
        timer_3 = f3.time_evaluator(f3.entry_name, dev, number=1)
        tcost_3 = timer_3(input_tvm, filter_tvm, scale_tvm, shift_tvm, relu_tvm).mean
        relu_scipy = np.maximum(scale_shift_scipy, 0)
        tvm.testing.assert_allclose(depthwise_conv2d_tvm.numpy(), depthwise_conv2d_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(scale_shift_tvm.numpy(), scale_shift_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(relu_tvm.numpy(), relu_scipy, rtol=1e-5)


def _transform_data(data, bn):
    # NCHW -> NCHW[x]c
    batch_size, channel, height, width = data.shape
    data = np.reshape(data, (batch_size, channel // bn, bn, height, width))
    data = np.transpose(data, (0, 1, 3, 4, 2))
    return data


def _transform_kernel(kernel, bn):
    # channel, channel_multiplier, kh, kw -> out_channel_chunk, kh, kw, out_channel_block
    channel, channel_multiplier, kh, kw = kernel.shape
    out_channel = channel * channel_multiplier
    kernel = np.reshape(kernel, (out_channel // bn, bn, kh, kw))
    kernel = np.transpose(kernel, (0, 2, 3, 1))
    out_channel_chunk, kh, kw, out_channel_block = kernel.shape
    return kernel.reshape(out_channel_chunk, 1, kh, kw, 1, out_channel_block)


def depthwise_conv2d_with_workload_NCHWc(
    target,
    dev,
    batch,
    in_channel,
    in_height,
    channel_multiplier,
    filter_height,
    stride,
    padding,
    dilation=1,
):
    in_width = in_height
    filter_channel = in_channel
    filter_width = filter_height
    stride_h = stride_w = stride

    assert (
        channel_multiplier == 1
    ), "depthwise_conv2d_NCHWc currently does not support channel multiplier > 1."
    pad_h, pad_w, _, _ = get_pad_tuple(padding, (filter_height, filter_width))
    padding_args = (pad_h, pad_w)

    out_channel = filter_channel * channel_multiplier
    # for testing functionality,
    # we choose arbitrary block size that can divide the channel,
    # regardless of the performance.
    oc_block = 1
    for bn in range(16, 0, -1):
        if out_channel % bn == 0:
            oc_block = bn
            break

    ic_block = 1
    for bn in range(oc_block, 0, -1):
        if in_channel % bn == 0:
            ic_block = bn
            break

    # placeholder
    Input = te.placeholder(
        (batch, in_channel // ic_block, in_height, in_width, ic_block), name="Input"
    )
    Filter = te.placeholder(
        (out_channel // oc_block, 1, filter_height, filter_width, 1, oc_block), name="Filter"
    )
    in_layout = "NCHW%dc" % ic_block
    out_layout = "NCHW%dc" % oc_block
    dtype = "float32"

    with autotvm.tophub.context(target):  # load tophub pre-tuned parameters
        dev = tvm.device(target, 0)
        with tvm.target.Target(target):
            # declare
            DepthwiseConv2d = topi.x86.depthwise_conv2d_NCHWc(
                Input,
                Filter,
                (stride_h, stride_w),
                padding,
                (dilation, dilation),
                in_layout,
                out_layout,
                dtype,
            )
            # TODO: add scale_shift implement for NCHWc and add test here
            Relu = topi.nn.relu(DepthwiseConv2d)
            # schedule
            s1 = topi.x86.schedule_depthwise_conv2d_NCHWc(DepthwiseConv2d)
            s2 = topi.x86.schedule_depthwise_conv2d_NCHWc(Relu)
        # build the kernels
        f1 = tvm.build(s1, [Input, Filter, DepthwiseConv2d], target)
        f2 = tvm.build(s2, [Input, Filter, Relu], target)

        # Prepare pod type for test data closure
        input_shape = (batch, in_channel, in_height, in_width)
        filter_shape = (filter_channel, channel_multiplier, filter_height, filter_width)

        # Use memoize, pickle the test data for next time use.
        @memoize("topi.tests.test_topi_depthwise_conv2d.NCHWc")
        def get_ref_data():
            input_np = np.random.uniform(size=input_shape).astype(dtype)
            filter_np = np.random.uniform(size=filter_shape).astype(dtype)
            # correctness with scipy
            dw_np = tvm.topi.testing.dilate_python(filter_np, (1, 1, dilation, dilation)).astype(
                dtype
            )
            depthwise_conv2d_scipy = tvm.topi.testing.depthwise_conv2d_python_nchw(
                input_np, dw_np, stride, padding
            )
            relu_scipy = np.maximum(depthwise_conv2d_scipy, 0)
            return (
                _transform_data(input_np, ic_block),
                _transform_kernel(filter_np, oc_block),
                _transform_data(depthwise_conv2d_scipy, oc_block),
                _transform_data(relu_scipy, oc_block),
            )

        # Get the test data
        (input_np, filter_np, depthwise_conv2d_scipy, relu_scipy) = get_ref_data()

        input_tvm = tvm.nd.array(input_np, dev)
        filter_tvm = tvm.nd.array(filter_np, dev)

        depthwise_conv2d_tvm = tvm.nd.array(
            np.zeros(shape=get_const_tuple(DepthwiseConv2d.shape), dtype=DepthwiseConv2d.dtype), dev
        )
        relu_tvm = tvm.nd.array(np.zeros(shape=get_const_tuple(Relu.shape), dtype=Relu.dtype), dev)
        # launch kernel 1 (depthwise_conv2d)
        f1(input_tvm, filter_tvm, depthwise_conv2d_tvm)
        # launch kernel 2 (depthwise_conv2d + relu)
        f2(input_tvm, filter_tvm, relu_tvm)
        tvm.testing.assert_allclose(depthwise_conv2d_tvm.numpy(), depthwise_conv2d_scipy, rtol=1e-5)
        tvm.testing.assert_allclose(relu_tvm.numpy(), relu_scipy, rtol=1e-5)


@tvm.testing.parametrize_targets
def test_depthwise_conv2d_nchw(target, dev):
    # mobilenet workloads
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 32, 112, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 64, 112, 1, 3, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 128, 56, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 128, 56, 1, 3, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 256, 28, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 256, 28, 1, 3, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 512, 14, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 512, 14, 1, 3, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 1024, 7, 1, 3, 1, "SAME")

    depthwise_conv2d_with_workload_nchw(target, dev, 1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 4, 256, 64, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 728, 32, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nchw(target, dev, 4, 256, 64, 2, 5, 2, "VALID")
    # dilation = 2
    depthwise_conv2d_with_workload_nchw(target, dev, 1, 728, 64, 1, 3, 1, "SAME", dilation=2)


@tvm.testing.parametrize_targets
def test_depthwise_conv2d_nhwc(target, dev):
    depthwise_conv2d_with_workload_nhwc(target, dev, 1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_nhwc(target, dev, 4, 256, 64, 2, 5, 2, "SAME")
    depthwise_conv2d_with_workload_nhwc(target, dev, 1, 728, 32, 1, 3, 1, "VALID")
    depthwise_conv2d_with_workload_nhwc(target, dev, 4, 256, 64, 2, 5, 2, "VALID")

    # dilation = 2
    # disabled because it uses too large shared memory on cuda
    # depthwise_conv2d_with_workload_nhwc(target, dev, 1, 728, 64, 1, 3, 1, "SAME", dilation=2)


# test llvm only for now since depthwise_conv2d_NCHWc implement is missing in other backend.
@tvm.testing.parametrize_targets("llvm")
def test_depthwise_conv2d_nchwc(target, dev):
    # NCHW[x]c
    depthwise_conv2d_with_workload_NCHWc(target, dev, 1, 728, 32, 1, 3, 1, "SAME", dilation=2)
    depthwise_conv2d_with_workload_NCHWc(target, dev, 1, 728, 32, 1, 3, 1, "SAME")
    depthwise_conv2d_with_workload_NCHWc(target, dev, 1, 728, 32, 1, 3, 1, "VALID")


def test_depthwise_conv2d_arm():
    # Test compilation on arm targets
    compile_depthwise_NHWC_int8_arm(1, 728, 32, 1, 3, 1, "SAME")
    compile_depthwise_NHWC_int8_arm(1, 728, 32, 1, 1, 1, "SAME", True)


if __name__ == "__main__":
    test_depthwise_conv2d()
