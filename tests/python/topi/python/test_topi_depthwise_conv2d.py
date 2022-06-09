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

import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import autotvm, te, topi
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple
from tvm.contrib.pickle_memoize import memoize
from tvm.topi.nn.depthwise_conv2d import _get_workload
from tvm.topi.x86.depthwise_conv2d import _fallback_schedule
from tvm.topi.generic import conv2d as conv2d_generic


_depthwise_conv2d_implement = {
    "NCHW": {
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
    },
    "NHWC": {
        "generic": [
            (topi.nn.depthwise_conv2d_nhwc, topi.generic.schedule_depthwise_conv2d_nhwc),
            (topi.nn.depthwise_conv2d_nhwc, conv2d_generic.schedule_depthwise_conv2d_nhwc),
        ],
        "arm_cpu": [
            (
                topi.arm_cpu.compute_depthwise_conv2d_nhwc,
                topi.arm_cpu.schedule_depthwise_conv2d_nhwc,
            )
        ],
        "gpu": [(topi.nn.depthwise_conv2d_nhwc, topi.cuda.schedule_depthwise_conv2d_nhwc)],
        "mali": [(topi.mali.depthwise_conv2d_nhwc, topi.mali.schedule_depthwise_conv2d_nhwc)],
        "bifrost": [(topi.mali.depthwise_conv2d_nhwc, topi.mali.schedule_depthwise_conv2d_nhwc)],
    },
    "NCHWc": {
        "generic": [(topi.x86.depthwise_conv2d_NCHWc, topi.x86.schedule_depthwise_conv2d_NCHWc)],
    },
}

random_seed = tvm.testing.parameter(0)

in_dtype, out_dtype = tvm.testing.parameters(
    ("float32", "float32"),
    ("float16", "float16"),
)


@tvm.testing.fixture
def input_shape(layout, batch, in_channel, in_size, filter_shape):
    if layout == "NCHW":
        return (batch, in_channel, in_size, in_size)
    elif layout == "NHWC":
        return (batch, in_size, in_size, in_channel)
    elif layout == "NCHWc":
        oc_block = filter_shape[-1]
        ic_block = next(bn for bn in range(oc_block, 0, -1) if in_channel % bn == 0)
        return (batch, in_channel // ic_block, in_size, in_size, ic_block)


@tvm.testing.fixture
def filter_shape(layout, in_channel, channel_multiplier, kernel):
    filter_channel = in_channel
    if layout == "NCHW":
        return (filter_channel, channel_multiplier, kernel, kernel)
    elif layout == "NHWC":
        return (kernel, kernel, filter_channel, channel_multiplier)
    elif layout == "NCHWc":
        out_channel = in_channel * channel_multiplier
        # For testing the functionality, we choose an arbitrary block
        # size that can divide out_channel, regardless of the
        # performance.
        oc_block = next(bn for bn in range(16, 0, -1) if out_channel % bn == 0)
        return (out_channel // oc_block, 1, kernel, kernel, 1, oc_block)


@tvm.testing.fixture
def scale_shape(layout, in_channel, channel_multiplier, filter_shape):
    out_channel = in_channel * channel_multiplier

    if layout in ("NCHW", "NHWC"):
        return (out_channel,)

    if layout == "NCHWc":
        oc_block = filter_shape[-1]
        return (out_channel // oc_block, oc_block)

    raise ValueError("Unknown layout {}".format(layout))


@tvm.testing.fixture
def shift_shape(scale_shape):
    return scale_shape


@tvm.testing.fixture(cache_return_value=True)
def ref_data(
    random_seed,
    in_dtype,
    out_dtype,
    layout,
    input_shape,
    filter_shape,
    dilation,
    stride,
    padding,
    scale_shape,
    shift_shape,
    use_scale_shift,
    apply_relu,
):
    np.random.seed(random_seed)

    # scipy.signal.convolve2d does not support float16 data types, and
    # the python fallback is too slow for general use.  Computing
    # ref_data in float32 will have fewer rounding errors than the TVM
    # float16 compute, but those vary based on schedule anyways.
    conv_dtype = "float32" if in_dtype == "float16" else in_dtype

    input_np = np.random.uniform(size=input_shape).astype(in_dtype)
    filter_np = np.random.uniform(size=filter_shape).astype(in_dtype)
    scale_np = np.random.uniform(size=scale_shape).astype(out_dtype)
    shift_np = np.random.uniform(size=shift_shape).astype(out_dtype)
    if layout == "NCHW":
        np_depthwise_conv2d = tvm.topi.testing.depthwise_conv2d_python_nchw
        dilation = (1, 1, dilation, dilation)
        reshape = (1, -1, 1, 1)
    elif layout == "NHWC":
        np_depthwise_conv2d = tvm.topi.testing.depthwise_conv2d_python_nhwc
        dilation = (dilation, dilation, 1, 1)
        reshape = (1, 1, 1, -1)
    elif layout == "NCHWc":
        np_depthwise_conv2d = tvm.topi.testing.depthwise_conv2d_python_nchwc
        dilation = (1, 1, dilation, dilation, 1, 1)
        reshape = (1, scale_shape[0], 1, 1, scale_shape[1])

    dilated_filter_np = tvm.topi.testing.dilate_python(filter_np, dilation)
    output_np = np_depthwise_conv2d(
        input_np.astype(conv_dtype), dilated_filter_np.astype(conv_dtype), stride, padding
    ).astype(out_dtype)

    if use_scale_shift:
        output_np = output_np * scale_np.reshape(reshape) + shift_np.reshape(reshape)
    if apply_relu:
        output_np = np.maximum(output_np, 0)

    return (
        input_np,
        filter_np,
        scale_np,
        shift_np,
        output_np,
    )


class BaseDepthwiseConv2D:
    """Provides the test_conv2d test function, to be used by other test classes.

    Test parameter sets are split out into different classes for
    readability (e.g. used for mobilenet), and for restrictions
    (e.g. implemented only for llvm).
    """

    layout = tvm.testing.parameter("NCHW", "NHWC")

    (batch, in_channel, in_size, channel_multiplier, kernel, stride) = tvm.testing.parameters(
        (1, 728, 32, 1, 3, 1),
        (4, 256, 64, 2, 5, 2),
    )
    padding = tvm.testing.parameter("SAME", "VALID")
    dilation = tvm.testing.parameter(1, 2)

    use_scale_shift = tvm.testing.parameter(True, False, ids=["with_scale_shift", "no_scale_shift"])
    apply_relu = tvm.testing.parameter(True, False, ids=["with_relu", "no_relu"])

    run_after_compile = True

    def test_conv2d(
        self,
        target,
        dev,
        in_dtype,
        out_dtype,
        layout,
        input_shape,
        filter_shape,
        scale_shape,
        shift_shape,
        use_scale_shift,
        apply_relu,
        batch,
        in_channel,
        channel_multiplier,
        kernel,
        stride,
        padding,
        dilation,
        ref_data,
    ):
        target = tvm.target.Target(target)
        if (
            target.kind.name == "cuda"
            and in_dtype == "float16"
            and not tvm.contrib.nvcc.have_fp16(dev.compute_version)
        ):
            pytest.xfail("CUDA float16 intrinsics not available")

        if (
            target.kind.name == "vulkan"
            and in_dtype == "float16"
            and (
                not target.attrs.get("supports_float16", False)
                or not target.attrs.get("supports_16bit_buffer", False)
            )
        ):
            pytest.xfail("Vulkan float16 driver support not available")

        # Transform the padding argument from 'str' to 'tuple' to
        # match the "workload" tuple in TopHub.  Which padding_args to
        # use for each layout chosen to reproduce previous behavior.
        if dilation == 1:
            padding_args = get_pad_tuple(padding, (kernel, kernel))
            padding_args_i = [0, 1, 2, 3] if layout == "NCHW" else [0, 1]
            padding_args = [padding_args[i] for i in padding_args_i]
        else:
            padding_args = padding

        # placeholder
        Input = te.placeholder(input_shape, name="Input", dtype=in_dtype)
        Filter = te.placeholder(filter_shape, name="Filter", dtype=in_dtype)
        Scale = te.placeholder(scale_shape, name="Scale", dtype=out_dtype)
        Shift = te.placeholder(shift_shape, name="Shift", dtype=out_dtype)

        if layout == "NCHW":
            topi_scale_shift = topi.nn.scale_shift_nchw
            fcompute_args = (Input, Filter, stride, padding_args, dilation, out_dtype)

        elif layout == "NHWC":
            topi_scale_shift = topi.nn.scale_shift_nhwc
            fcompute_args = (Input, Filter, stride, padding_args, dilation, out_dtype)

        elif layout == "NCHWc":
            topi_scale_shift = topi.nn.scale_shift_nchwc
            in_layout = "NCHW{}c".format(input_shape[-1])
            out_layout = "NCHW{}c".format(filter_shape[-1])
            fcompute_args = (
                Input,
                Filter,
                stride,
                padding,
                dilation,
                in_layout,
                out_layout,
                out_dtype,
            )

        with autotvm.tophub.context(target):  # load tophub pre-tuned parameters
            impl_list = tvm.topi.testing.dispatch(target, _depthwise_conv2d_implement[layout])[:]
            if target == "llvm" and layout == "NCHW" and channel_multiplier == 1 and dilation == 1:
                impl_list.append(
                    (topi.x86.depthwise_conv2d_nchw, topi.x86.schedule_depthwise_conv2d_nchw)
                )

            for fcompute, fschedule in impl_list:
                with tvm.target.Target(target):
                    # Declare, build schedule
                    C = fcompute(*fcompute_args)
                    if use_scale_shift:
                        C = topi_scale_shift(C, Scale, Shift)
                    if apply_relu:
                        C = topi.nn.relu(C)

                    s = fschedule(C)

                # Build and run
                f = tvm.build(s, [Input, Filter, Scale, Shift, C], target)

                if self.run_after_compile:
                    input_np, filter_np, scale_np, shift_np, output_np = ref_data
                    if "int" in out_dtype:
                        tol = {"atol": 0, "rtol": 0}
                    elif out_dtype == "float32":
                        tol = {"rtol": 1e-4, "atol": 1e-5}
                    elif out_dtype == "float16":
                        # A summation in float16 with a single accumulator very
                        # quickly runs into large rounding errors.  At some point,
                        # this tolerance should be schedule-dependent for to avoid
                        # false negatives.
                        num_values_summed = kernel * kernel
                        gap_size = (
                            np.nextafter(output_np.max(), np.inf, dtype=output_np.dtype)
                            - output_np.max()
                        )
                        tol = {"rtol": 1e-3, "atol": num_values_summed * gap_size / 2}

                    input_tvm = tvm.nd.array(input_np, dev)
                    filter_tvm = tvm.nd.array(filter_np, dev)
                    scale_tvm = tvm.nd.array(scale_np, dev)
                    shift_tvm = tvm.nd.array(shift_np, dev)
                    output_tvm = tvm.nd.array(
                        np.zeros(shape=get_const_tuple(C.shape), dtype=C.dtype),
                        dev,
                    )

                    f(input_tvm, filter_tvm, scale_tvm, shift_tvm, output_tvm)
                    tvm.testing.assert_allclose(output_np, output_tvm.numpy(), **tol)


class TestDepthwiseConv2D(BaseDepthwiseConv2D):
    """Test variety of parameters, defined in BaseDepthwiseConv2D.  Also
    has llvm-specific tests for workload padding."""

    @tvm.testing.parametrize_targets("llvm")
    def test_workload_padding(
        self,
        out_dtype,
        layout,
        input_shape,
        filter_shape,
        target,
        ref_data,
        stride,
        padding,
        dilation,
    ):
        input_np, filter_np, scale_np, shift_np, output_np = ref_data
        if layout == "NCHW":
            _, _, out_height, out_width = output_np.shape
        elif layout == "NHWC":
            _, out_height, out_width, _ = output_np.shape
        elif layout == "NCHWc":
            _, _, out_height, out_width, _ = output_np.shape

        Input = te.placeholder(input_shape, name="Input")
        Filter = te.placeholder(filter_shape, name="Filter")
        wkl = _get_workload(Input, Filter, (stride, stride), padding, dilation, out_dtype, layout)

        # check if tile_ow candidates are the factors of the right output weight.
        with tvm.target.Target(target):
            cfg = autotvm.get_config()
            _fallback_schedule(cfg, wkl)
            ow_tile = np.prod(cfg["tile_ow"].size)

            tvm.testing.assert_allclose(ow_tile, out_width)


class TestDepthwiseConv2D_MobilenetWorkloads(BaseDepthwiseConv2D):
    """Extra tests to verify functionality for workloads used by mobilenet."""

    layout = tvm.testing.parameter("NCHW")

    batch = tvm.testing.parameter(1)
    channel_multiplier = tvm.testing.parameter(1)
    kernel = tvm.testing.parameter(3)
    padding = tvm.testing.parameter("SAME")
    dilation = tvm.testing.parameter(1)

    in_channel, in_size, stride = tvm.testing.parameters(
        (32, 112, 1),
        (64, 112, 2),
        (128, 56, 1),
        (128, 56, 2),
        (256, 28, 1),
        (256, 28, 2),
        (512, 14, 1),
        (512, 14, 2),
        (1024, 7, 1),
    )


@tvm.testing.parametrize_targets("llvm")
class TestDepthwiseConv2D_NCHWc(BaseDepthwiseConv2D):
    """Tests specific to NCHWc layouts.

    Once the implementation supports channel_multiplier>1 and GPU
    devices, this class can be merged into TestDepthwiseConv2D.
    """

    # depthwise_conv2d_NCHWc currently does not support channel multiplier > 1
    layout = tvm.testing.parameter("NCHWc")
    (batch, in_channel, in_size, channel_multiplier, kernel, stride) = tvm.testing.parameters(
        (1, 728, 32, 1, 3, 1),
    )


@tvm.testing.parametrize_targets("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")
class TestDepthwiseConv2DArmCompile(BaseDepthwiseConv2D):
    """Compile-only tests for cross-compiling to ARM."""

    layout = tvm.testing.parameter("NHWC", "NCHW")
    batch = tvm.testing.parameter(1)
    dilation = tvm.testing.parameter(1)
    in_dtype, out_dtype = tvm.testing.parameters(("int16", "int32"))
    in_channel = tvm.testing.parameter(728)
    in_size = tvm.testing.parameter(32)
    kernel = tvm.testing.parameter(1)
    channel_multiplier = tvm.testing.parameter(1, 3)
    stride = tvm.testing.parameter(1)
    padding = tvm.testing.parameter("SAME")
    use_scale_shift = tvm.testing.parameter(True, False, ids=["with_scale_shift", "no_scale_shift"])

    run_after_compile = False


if __name__ == "__main__":
    tvm.testing.main()
