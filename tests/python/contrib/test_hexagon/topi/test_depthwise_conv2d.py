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
"""Depthwise Conv2D Tests."""

import numpy as np

import tvm
from tvm.contrib.hexagon.session import Session
import tvm.testing
import tvm.topi.testing

from tvm import te, topi
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple

from ..infrastructure import get_hexagon_target


class BaseDepthwiseConv2D:
    """Provides the test_conv2d test function, to be used by other test classes.

    Test parameter sets are split out into different classes for
    readability (e.g. used for mobilenet), and for restrictions
    (e.g. implemented only for llvm).
    """

    random_seed = tvm.testing.parameter(0)

    in_dtype, out_dtype = tvm.testing.parameters(
        ("float32", "float32"),
    )

    @tvm.testing.fixture
    def input_shape(self, layout, batch, in_channel, in_size, filter_shape):
        """Returns input shape."""
        if layout == "NCHW":
            return (batch, in_channel, in_size, in_size)
        elif layout == "NHWC":
            return (batch, in_size, in_size, in_channel)
        elif layout == "NCHWc":
            oc_block = filter_shape[-1]
            ic_block = next(bn for bn in range(oc_block, 0, -1) if in_channel % bn == 0)
            return (batch, in_channel // ic_block, in_size, in_size, ic_block)
        else:
            raise RuntimeError(f"Not supported layout {layout}")

    @tvm.testing.fixture
    def filter_shape(self, layout, in_channel, channel_multiplier, kernel):
        """Returns filter shape."""
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
        else:
            raise RuntimeError(f"Not supported layout {layout}")

    @tvm.testing.fixture
    def scale_shape(self, layout, in_channel, channel_multiplier, filter_shape):
        """Returns scale shape."""
        out_channel = in_channel * channel_multiplier

        if layout in ("NCHW", "NHWC"):
            return (out_channel,)

        if layout == "NCHWc":
            oc_block = filter_shape[-1]
            return (out_channel // oc_block, oc_block)

        raise ValueError("Unknown layout {}".format(layout))

    @tvm.testing.fixture
    def shift_shape(self, scale_shape):
        """Returns shift shape."""
        return scale_shape

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(
        self,
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
        """Generate reference data."""
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

    @tvm.testing.requires_hexagon
    def test_conv2d(
        self,
        hexagon_session: Session,
        in_dtype,
        out_dtype,
        layout,
        input_shape,
        filter_shape,
        scale_shape,
        shift_shape,
        use_scale_shift,
        apply_relu,
        kernel,
        stride,
        padding,
        dilation,
        ref_data,
    ):
        """Test conv2D."""
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
        input_tensor = te.placeholder(input_shape, name="input_tensor", dtype=in_dtype)
        filter_tensor = te.placeholder(filter_shape, name="filter_tensor", dtype=in_dtype)
        scale = te.placeholder(scale_shape, name="scale", dtype=out_dtype)
        shift = te.placeholder(shift_shape, name="shift", dtype=out_dtype)

        if layout == "NCHW":
            topi_scale_shift = topi.nn.scale_shift_nchw
            fcompute_args = (input_tensor, filter_tensor, stride, padding_args, dilation, out_dtype)

        elif layout == "NHWC":
            topi_scale_shift = topi.nn.scale_shift_nhwc
            fcompute_args = (input_tensor, filter_tensor, stride, padding_args, dilation, out_dtype)

        elif layout == "NCHWc":
            topi_scale_shift = topi.nn.scale_shift_nchwc
            in_layout = "NCHW{}c".format(input_shape[-1])
            out_layout = "NCHW{}c".format(filter_shape[-1])
            fcompute_args = (
                input_tensor,
                filter_tensor,
                stride,
                padding,
                dilation,
                in_layout,
                out_layout,
                out_dtype,
            )

        with tvm.target.Target(get_hexagon_target("v68")):
            # Declare, build schedule
            if layout == "NCHW":
                fcompute = topi.nn.depthwise_conv2d_nchw
                fschedule = topi.hexagon.schedule_depthwise_conv2d_nchw
            elif layout == "NHWC":
                fcompute = topi.nn.depthwise_conv2d_nhwc
                fschedule = topi.hexagon.schedule_depthwise_conv2d_nhwc
            c_tensor = fcompute(*fcompute_args)
            if use_scale_shift:
                c_tensor = topi_scale_shift(c_tensor, scale, shift)
            if apply_relu:
                c_tensor = topi.nn.relu(c_tensor)

            schedule = fschedule([c_tensor])

            # Build and run
            f = tvm.build(
                schedule,
                [input_tensor, filter_tensor, scale, shift, c_tensor],
                get_hexagon_target("v68"),
            )
            mod = hexagon_session.load_module(f)

            input_np, filter_np, scale_np, shift_np, output_np = ref_data

            dev = hexagon_session.device
            input_tvm = tvm.nd.array(input_np, dev)
            filter_tvm = tvm.nd.array(filter_np, dev)
            scale_tvm = tvm.nd.array(scale_np, dev)
            shift_tvm = tvm.nd.array(shift_np, dev)
            output_tvm = tvm.nd.array(
                np.zeros(shape=get_const_tuple(c_tensor.shape), dtype=c_tensor.dtype),
                dev,
            )

            mod(input_tvm, filter_tvm, scale_tvm, shift_tvm, output_tvm)

            tol = {"rtol": 1e-4, "atol": 1e-5}
            tvm.testing.assert_allclose(output_np, output_tvm.numpy(), **tol)


class TestDepthwiseConv2DMobilenetWorkloads(BaseDepthwiseConv2D):
    """Extra tests to verify functionality for workloads used by mobilenet."""

    layout = tvm.testing.parameter("NCHW", "NHWC")
    use_scale_shift = tvm.testing.parameter(False, ids=["no_scale_shift"])
    apply_relu = tvm.testing.parameter(False, ids=["no_relu"])

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
    )


class TestDepthwiseConv2D(BaseDepthwiseConv2D):
    """Test depthwise conv2D class."""

    layout = tvm.testing.parameter("NCHW", "NHWC")
    use_scale_shift = tvm.testing.parameter(True, False, ids=["with_scale_shift", "no_scale_shift"])
    apply_relu = tvm.testing.parameter(True, False, ids=["with_relu", "no_relu"])

    (batch, in_channel, in_size, channel_multiplier, kernel, stride) = tvm.testing.parameters(
        (1, 64, 32, 1, 3, 1),
        (1, 128, 64, 2, 5, 2),
    )
    padding = tvm.testing.parameter("VALID")
    dilation = tvm.testing.parameter(1)


# TODO(hexagon-team): add TestDepthwiseConv2D_NCHWc test.

if __name__ == "__main__":
    tvm.testing.main()
