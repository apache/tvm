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
"""Test code for convolution."""
import numpy as np

import tvm
import tvm.testing
from tvm import topi
from tvm import te
from tvm.contrib.hexagon.session import Session
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm.topi.nn.utils import get_pad_tuple

from ..infrastructure import get_hexagon_target


class BaseConv2DTests:
    """Conv2D test class."""

    add_bias = tvm.testing.parameter(False)
    apply_relu = tvm.testing.parameter(False)
    dilation = tvm.testing.parameter(1)
    batch = tvm.testing.parameter(1)
    dtype = tvm.testing.parameter("float32")

    random_seed = tvm.testing.parameter(0)

    @tvm.testing.fixture
    def input_shape(self, batch, in_channel, in_size):
        return (batch, in_channel, in_size, in_size)

    @tvm.testing.fixture
    def weight_shape(self, num_filter, in_channel, kernel):
        return (num_filter, in_channel, kernel, kernel)

    @tvm.testing.fixture
    def bias_shape(self, num_filter):
        return (num_filter, 1, 1)

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(
        self,
        random_seed,
        input_shape,
        weight_shape,
        bias_shape,
        dtype,
        stride,
        padding,
        dilation,
        add_bias,
        apply_relu,
    ):
        """Generate reference data."""
        np.random.seed(random_seed)

        # scipy.signal.convolve2d does not support float16 data types, and
        # the python fallback is too slow for general use.  Computing
        # ref_data in float32 will have fewer rounding errors than the TVM
        # float16 compute, but those vary based on schedule anyways.
        conv_dtype = "float32" if dtype == "float16" else dtype

        a_np = np.random.uniform(size=input_shape).astype(dtype)
        w_np = np.random.uniform(size=weight_shape).astype(dtype)
        b_np = np.random.uniform(size=bias_shape).astype(dtype)
        dw_np = tvm.topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        c_np = tvm.topi.testing.conv2d_nchw_python(
            a_np.astype(conv_dtype), dw_np.astype(conv_dtype), stride, padding
        ).astype(dtype)

        if add_bias:
            c_np = c_np + b_np
        if apply_relu:
            c_np = np.maximum(c_np, 0)
        return a_np, w_np, b_np, c_np

    @tvm.testing.requires_hexagon
    def test_conv2d_nchw(
        self,
        hexagon_session: Session,
        batch,
        in_channel,
        in_size,
        num_filter,
        kernel,
        stride,
        padding,
        dtype,
        ref_data,
        dilation,
        add_bias,
        apply_relu,
    ):
        """Test Conv2d NCHW."""

        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
        padding_sum = pad_top + pad_left + pad_bottom + pad_right

        a_np, w_np, b_np, c_np = ref_data

        a_tensor = te.placeholder(a_np.shape, name="a_tensor", dtype=dtype)
        w_tensor = te.placeholder(w_np.shape, name="w_tensor", dtype=dtype)
        bias = te.placeholder(b_np.shape, name="bias", dtype=dtype)

        if "int" in dtype:
            tol = {"atol": 0, "rtol": 0}
        elif dtype == "float32":
            tol = {"rtol": 1e-4, "atol": 2e-4}
        elif dtype == "float16":
            # a_tensor summation in float16 with a single accumulator very
            # quickly runs into large rounding errors.  At some point,
            # this tolerance should be schedule-dependent for to avoid
            # false negatives.
            num_values_summed = in_channel * kernel * kernel
            gap_size = np.nextafter(c_np.max(), np.inf, dtype=c_np.dtype) - c_np.max()
            tol = {"rtol": 1e-3, "atol": num_values_summed * gap_size / 2}

        with tvm.target.Target(get_hexagon_target("v68")):
            fcompute = topi.nn.conv2d_nchw
            fschedule = topi.hexagon.schedule_conv2d_nchw
            c_tensor = fcompute(
                a_tensor, w_tensor, (stride, stride), padding, (dilation, dilation), dtype
            )
            if add_bias:
                c_tensor = topi.add(c_tensor, bias)
            if apply_relu:
                c_tensor = topi.nn.relu(c_tensor)
            s = fschedule([c_tensor])

        func_name = "conv2d_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            dtype,
            batch,
            in_channel,
            in_size,
            num_filter,
            kernel,
            stride,
            padding_sum,
            dilation,
        )
        func = tvm.build(
            s,
            [a_tensor, w_tensor, bias, c_tensor],
            get_hexagon_target("v68"),
            name=func_name,
        )
        mod = hexagon_session.load_module(func)

        dev = hexagon_session.device
        a_data = tvm.nd.array(a_np, dev)
        weight = tvm.nd.array(w_np, dev)
        b = tvm.nd.array(b_np, dev)

        c = tvm.nd.array(np.zeros(get_const_tuple(c_tensor.shape), dtype=c_tensor.dtype), dev)
        mod[func_name](a_data, weight, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, **tol)


class TestBatchSize(BaseConv2DTests):
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (32, 28, 32, 3, 1, 1),
    )
    batch = tvm.testing.parameter(1, 4, 9)


class TestBiasRelu(BaseConv2DTests):
    apply_relu = tvm.testing.parameter(True, False, ids=["relu", "no_relu"])
    add_bias = tvm.testing.parameter(True, False, ids=["bias", "no_bias"])
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (64, 56, 64, 3, 1, 1),
        (64, 8, 64, 3, 1, (1, 2, 2, 1)),
        (64, 8, 64, 5, 2, (1, 3)),
        (64, 8, 64, 3, 1, "VALID"),
        (32, 8, 32, 24, 1, "SAME"),
    )


class TestResNet18Workloads(BaseConv2DTests):
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (3, 224, 64, 7, 2, 3),
        (64, 56, 64, 3, 1, 1),
        (64, 56, 64, 1, 1, 0),
        (64, 56, 32, 3, 2, 1),
        (64, 56, 32, 1, 2, 0),
        (64, 28, 32, 3, 1, 1),
    )


class TestMobilenet(BaseConv2DTests):
    batch, in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (1, 32, 112, 32, 3, 1, 1),
    )


class TestWeirdWorkloads(BaseConv2DTests):
    batch, in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (2, 2, 2, 2, 2, 2, 2),
        (3, 3, 3, 3, 3, 3, 3),
        (4, 4, 4, 4, 4, 4, 4),
        (5, 5, 5, 5, 5, 5, 5),
        (6, 6, 6, 6, 6, 6, 6),
        (1, 1, 1, 1, 1, 1, 1),
        (2, 13, 71, 59, 3, 1, 1),
    )


class TestAsymmetricPadding(BaseConv2DTests):
    dilation = tvm.testing.parameter(1, 2)
    in_channel, in_size, num_filter, kernel, stride, padding = tvm.testing.parameters(
        (3, 35, 64, 7, 2, (0, 0, 1, 1)),
        (64, 8, 128, 3, 1, (3, 3, 2, 2)),
        (64, 8, 64, 1, 1, (1, 2, 2, 1)),
        (64, 17, 48, 1, 1, (1, 2)),
        (64, 8, 64, 3, 1, (3, 1)),
        (128, 8, 96, 3, 1, (0, 2)),
        (64, 35, 64, 3, 1, (1, 2)),
        (64, 8, 64, 1, 1, "VALID"),
        (388, 8, 64, 3, 1, "VALID"),
        (64, 10, 48, 3, 1, "VALID"),
        (64, 19, 64, 1, 1, "SAME"),
        (64, 5, 32, 2, 1, "SAME"),
        (32, 8, 32, 3, 1, "SAME"),
        (64, 8, 64, 3, 1, (1, 2, 2, 1)),
        (64, 8, 64, 5, 2, (1, 3)),
        (64, 8, 64, 3, 1, "VALID"),
        (32, 8, 32, 24, 1, "SAME"),
        (32, 35, 64, 7, 2, (0, 0, 2, 2)),
    )


if __name__ == "__main__":
    tvm.testing.main()
