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
# pylint: disable=invalid-name, unused-variable, unused-argument, disable=line-too-long, redefined-outer-name

"""Test depthwise_conv2d slice op for hexagon."""

import numpy as np

import tvm
import tvm.testing
import tvm.topi.hexagon.qnn as qn
from tvm.topi.testing import depthwise_conv2d_python_nhwc
from tvm.topi.hexagon.slice_ops.dwconv2d import dwconv2d_compute, dwconv2d_schedule
from tvm.contrib.hexagon import allocate_hexagon_array

from ...infrastructure import transform_numpy, quantize_np


@tvm.testing.fixture
def input_np(in_shape, dtype, low, high):
    if dtype in ("uint8"):
        return np.random.uniform(low=low, high=high, size=in_shape).astype("float32")
    if dtype in ("int8"):
        return np.random.uniform(low=-low, high=high, size=in_shape).astype("float32")
    return np.random.uniform(size=in_shape).astype(dtype)


@tvm.testing.fixture
def input_np_padded(input_np, in_shape, padded_in_shape):
    pad_height = padded_in_shape[1] - in_shape[1]
    pad_width = padded_in_shape[2] - in_shape[2]
    pad_channel = padded_in_shape[3] - in_shape[3]
    input_padded = np.pad(
        input_np, ((0, 0), (0, pad_height), (0, pad_width), (0, pad_channel)), "constant"
    )
    return input_padded


@tvm.testing.fixture
def in_out_layout(dtype):
    if dtype == "float16":
        return "nhwc-8h2w32c2w-2d"
    elif dtype in ("uint8", "int8"):
        return "nhwc-8h8w32c-2d"
    else:
        raise RuntimeError(f"Unsupported quantized data type '{dtype}'")


@tvm.testing.fixture
def expected_output_np(input_np, dilated_weights_np, stride, dtype):
    dilated_weights_np_t = dilated_weights_np.transpose(0, 1, 3, 2)
    ref_type = dtype
    if dtype in ("uint8", "int8"):
        # for quantized versions, return float32 output
        ref_type = "float32"
    ref_np = depthwise_conv2d_python_nhwc(
        input_np.astype("float32"), dilated_weights_np_t.astype("float32"), stride, padding=0
    ).astype(ref_type)
    return ref_np


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, in_out_layout, dtype):
    if dtype == "float16":
        return transform_numpy(expected_output_np, "nhwc", in_out_layout)
    elif dtype in ("uint8", "int8"):
        quant_arr, scale, zero_point = quantize_np(expected_output_np, dtype)
        return [transform_numpy(quant_arr, "nhwc", in_out_layout), scale, zero_point]
    else:
        raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def transformed_input_np_padded(input_np_padded, in_out_layout, dtype):
    if dtype == "float16":
        return transform_numpy(input_np_padded, "nhwc", in_out_layout)
    if dtype in ("uint8", "int8"):
        quant_arr, scale, zero_point = quantize_np(input_np_padded, dtype)
        return [transform_numpy(quant_arr, "nhwc", in_out_layout), scale, zero_point]
    raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def weights_np(filt_shape, dtype):
    if dtype == "float16":
        return np.random.uniform(size=filt_shape).astype(dtype)
    elif dtype in ("uint8", "int8"):
        weight_arr = np.random.uniform(low=-5, high=5, size=filt_shape).astype("float32")
        return weight_arr
    else:
        raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def dilated_filt_shape(filt_shape, dilation):
    """Compute the dilated filter shape when dilation > 1"""
    filt_height, filt_width, in_channel, out_channel = filt_shape
    dilation_height, dilation_width = dilation
    if dilation_height == 1 and dilation_width == 1:
        return filt_shape
    dilated_height = dilation_height * (filt_height - 1) + 1
    dilated_width = dilation_width * (filt_width - 1) + 1
    return dilated_height, dilated_width, in_channel, out_channel


@tvm.testing.fixture
def dilated_weights_np(weights_np, dilation, dilated_filt_shape, dtype):
    """Get dilated weights from original weights for testing"""
    if dtype in ["int8", "uint8"]:
        dtype = "float32"
    filt_height, filt_width, in_channels, out_channels = weights_np.shape
    dilated_weights = np.zeros(dilated_filt_shape)
    dilation_height, dilation_width = dilation
    if dilation_height == 1 and dilation_width == 1:
        return weights_np
    dilated_height, dilated_width = dilated_filt_shape[0], dilated_filt_shape[1]
    for in_channel in range(in_channels):
        for out_channel in range(out_channels):
            for dilation_i, height_i in zip(
                range(0, dilated_height, dilation_height), range(filt_height)
            ):
                for dilation_j, width_j in zip(
                    range(0, dilated_width, dilation_width), range(filt_width)
                ):
                    dilated_weights[dilation_i, dilation_j, in_channel, out_channel] = weights_np[
                        height_i, width_j, in_channel, out_channel
                    ]
    return dilated_weights


@tvm.testing.fixture
def transformed_weights_np(weights_np, dtype):
    height, width, in_channel, out_channel = weights_np.shape
    t = weights_np.reshape([height, width, in_channel, out_channel // 32, 32]).transpose(
        3, 0, 1, 2, 4
    )
    if dtype == "float16":
        return t
    if dtype in ("uint8", "int8"):
        quant_arr, scale, zero_point = quantize_np(t, dtype)
        return [quant_arr, scale, zero_point]
    raise RuntimeError(f"Unsupported data type '{dtype}'")


def generate_test_config(test_params):
    """Utility function to generate test config with meaningful ids"""
    test_config = {}

    dims = lambda vals: "x".join(map(str, vals))

    for param in test_params:
        in_shape, filt_shape, stride, dilation = param[:4]
        test_name = f"nhwc{dims(in_shape)}-hwio{dims(filt_shape)}-stride{dims(stride)}-dilation{dims(dilation)}"
        test_config[test_name] = param

    return test_config


class Testdwconv2dSlice:
    """Test class that defines the dwconv2d slice test"""

    test_params = [
        [(1, 10, 10, 32), (3, 3, 1, 32), (1, 1), (1, 1), 0.0, 10.0],
        [(1, 10, 10, 64), (3, 3, 1, 64), (1, 1), (1, 1), 0.0, 10.0],
        [(1, 12, 12, 32), (5, 5, 1, 32), (1, 1), (1, 1), 0.0, 20.0],
        [(1, 16, 16, 32), (5, 5, 1, 32), (1, 1), (2, 2), 0.0, 1.0],
        [(1, 18, 10, 32), (3, 3, 1, 32), (1, 1), (1, 1), 0.0, 10.0],
        [(1, 18, 18, 32), (3, 3, 1, 32), (2, 2), (1, 1), 0.0, 10.0],
        [(1, 18, 10, 96), (3, 3, 1, 96), (1, 1), (1, 1), 0.0, 10.0],
        [(1, 21, 21, 32), (7, 7, 1, 32), (2, 2), (1, 1), 0.0, 10.0],
        [(1, 28, 28, 32), (7, 7, 1, 32), (2, 2), (2, 2), 0.0, 10.0],
        [(1, 28, 28, 96), (7, 7, 1, 96), (2, 2), (2, 2), 0.0, 10.0],
        [(1, 10, 16, 32), (3, 1, 1, 32), (1, 1), (1, 1), 0.0, 10.0],
    ]

    test_config = generate_test_config(test_params)

    in_shape, filt_shape, stride, dilation, low, high = tvm.testing.parameters(
        *test_config.values(), ids=test_config.keys()
    )
    dtype = tvm.testing.parameter("float16", "uint8")
    working_scope = tvm.testing.parameter("global.vtcm")
    weights_layout = tvm.testing.parameter("ohwi32o-1d")

    @tvm.testing.fixture
    def padded_in_shape(self, in_shape, dtype):
        """Padding the input shape according to layout"""
        # NOTE: For float16, the input layout is always assumed to be nhwc-8h2w32c2w-2d and
        # for int8/uint8, it's nhwc-8h8w32c-2d.
        # For both nhwc-8h2w32c2w-2d and nhwc-8h8w32c-2d, the height should be a multiple
        # of 8. However, the width should be a multiple of 4 for the first case and 8 for
        # the second case.
        in_batch, in_height, in_width, in_channel = in_shape
        in_height = ((in_height + 7) // 8) * 8

        if dtype == "float16":
            in_width = ((in_width + 3) // 4) * 4
        elif dtype in ("uint8", "int8"):
            in_width = ((in_width + 7) // 8) * 8

        in_channel = ((in_channel + 31) // 32) * 32

        return in_batch, in_height, in_width, in_channel

    @tvm.testing.fixture
    def out_shape(self, in_shape, dilated_filt_shape, stride):
        in_batch, in_height, in_width, _ = in_shape
        filt_height, filt_width, _, num_filt = dilated_filt_shape
        out_height = (in_height - filt_height) // stride[0] + 1
        out_width = (in_width - filt_width) // stride[1] + 1
        out_channel = num_filt
        return in_batch, out_height, out_width, out_channel

    @tvm.testing.requires_hexagon
    def test_dwconv2d(
        self,
        dtype,
        in_out_layout,
        weights_layout,
        padded_in_shape,
        weights_np,
        filt_shape,
        stride,
        dilation,
        out_shape,
        input_np,
        input_np_padded,
        transformed_weights_np,
        expected_output_np,
        target,
        working_scope,
        transformed_input_np_padded,
        transformed_expected_output_np,
        hexagon_session,
    ):
        """Main test function that tests the dwconv2d slice op"""
        input_tensor = tvm.te.placeholder(padded_in_shape, name="InputTensor", dtype=dtype)
        weights = tvm.te.placeholder(filt_shape, name="Weights", dtype=dtype)

        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        # Construct compute and schedule based on dtype
        if dtype in ("uint8", "int8"):
            in_data_np, activation_scale, activation_zero_point = transformed_input_np_padded
            (
                weights_data_np,
                weight_scale,
                weight_zero_point,
            ) = transformed_weights_np
            out_data_np, output_scale, output_zero_point = transformed_expected_output_np

            output_tensor = qn.qdepthwise_conv2d_compute(
                input_tensor,
                weights,
                out_shape,
                stride,
                dilation,
                dtype,
                activation_zero_point,
                activation_scale,
                weight_zero_point,
                weight_scale,
                output_zero_point,
                output_scale,
            )

            tir_schedule = qn.qdepthwise_conv2d_schedule(
                output_tensor, [input_tensor, weights], in_out_layout, weights_layout
            )

        elif dtype == "float16":
            in_data_np = transformed_input_np_padded
            out_data_np = transformed_expected_output_np
            weights_data_np = transformed_weights_np
            output_tensor = dwconv2d_compute(
                input_tensor, weights, out_shape, stride, dilation, dtype
            )

            tir_schedule = dwconv2d_schedule(
                output_tensor, [input_tensor, weights], in_out_layout, weights_layout
            )
        else:
            raise RuntimeError(f"Unsupport dtype '{dtype}'")

        func_name = "depthwise_conv2d_slice"
        with tvm.transform.PassContext(opt_level=3):
            runtime_module = tvm.build(
                tir_schedule.mod,
                [input_tensor, output_tensor],
                target=target,
                name=func_name,
            )

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=in_data_np,
            axis_separators=[4],
            mem_scope=working_scope,
        )

        weights_arr = allocate_hexagon_array(
            hexagon_session.device, data=weights_data_np, mem_scope=working_scope
        )

        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            out_data_np.shape,
            dtype=dtype,
            axis_separators=[4],
            mem_scope=working_scope,
        )

        mod = hexagon_session.load_module(runtime_module)
        mod(input_arr, weights_arr, output_arr)
        n, h, w, c = out_shape

        if dtype in ("uint8", "int8"):
            output_np = output_arr.numpy().reshape([n, h // 8, w // 8, c // 32, 8, 8, 32])
            np.testing.assert_allclose(output_np, out_data_np, atol=3, rtol=0.02)
        elif dtype == "float16":
            output_np = output_arr.numpy()
            np.testing.assert_allclose(output_np, out_data_np, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    tvm.testing.main()
