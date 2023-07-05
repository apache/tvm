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

import numpy as np
from typing import *

from tvm import te
import tvm.testing
from tvm.topi.testing import poolnd_python
from tvm.contrib.hexagon.session import Session
import tvm.topi.hexagon.slice_ops as sl
import tvm.topi.hexagon.qnn as qn
from tvm.contrib.hexagon import allocate_hexagon_array
import pytest
from ...infrastructure import transform_numpy, quantize_np, get_hexagon_target
from ...pytest_util import get_multitest_ids, create_populated_numpy_ndarray, TensorContentRandom


dtype = tvm.testing.parameter("uint8", "float16")


@tvm.testing.fixture
def output_layout(output_shape, op_layout, dtype):
    if op_layout == "NHWC":
        o_b, o_h, o_w, o_c = output_shape
        if dtype == "float16":
            if o_h == 1 and o_w == 1:
                return "n11c-1024c-2d"
            else:
                return "nhwc-8h2w32c2w-2d"
        elif dtype == "int8" or "uint8":
            if o_h == 1 and o_w == 1:
                return "n11c-2048c-2d"
            else:
                return "nhwc-8h8w32c-2d"
        else:
            raise RuntimeError(f"Unsupported data type '{dtype}'")

    elif op_layout == "NCHW":
        o_b, o_c, o_h, o_w = output_shape
        if dtype == "float16":
            if o_h == 1 and o_w == 1:
                return "nc11-1024c-2d"
            else:
                return "nchw-8h2w32c2w-2d"
        elif dtype == "int8" or "uint8":
            if o_h == 1 and o_w == 1:
                return "nc11-2048c-2d"
            else:
                return "nchw-8h8w32c-2d"
        else:
            raise RuntimeError(f"Unsupported data type '{dtype}'")
    else:
        raise RuntimeError(f"Unsupported layout for qnn.avg_pool2d '{op_layout}'")


@tvm.testing.fixture
def input_layout(op_layout, dtype):
    in_layout = op_layout.lower()
    if dtype == "float16":
        return in_layout + "-8h2w32c2w-2d"
    elif dtype == "int8" or "uint8":
        return in_layout + "-8h8w32c-2d"
    else:
        raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def input_np(input_shape, dtype: str, input_tensor_populator):
    if dtype == "uint8":
        dtype = "float32"  # Use "float32" input which will be quantized later
    return create_populated_numpy_ndarray(input_shape, dtype, input_tensor_populator)


class TestAvgPool2dSlice:
    _param_descs = [
        "out_shape",  # output_shape
        "kernel",  # kernel
        "stride",  # stride
        "dil",  # dilation
        "pad",  # padding
        "ceil",  # ceil_mode
        "cnt_padded",  # count_include_pad
        "op_layout",  # input output 4D layout
        None,  # input_tensor_populator
    ]
    _multitest_params = [
        (
            [1, 7, 11, 32],
            [3, 3],
            [3, 2],
            [2, 3],
            [1, 2, 3, 4],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [4, 4],
            [2, 2],
            [2, 3],
            [0, 2, 1, 4],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        # Test default stride,dilation, and padding with different layouts
        (
            [1, 10, 10, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 12, 12, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 32, 14, 14],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NCHW",
            TensorContentRandom(),
        ),
        (
            [1, 32, 15, 15],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NCHW",
            TensorContentRandom(),
        ),
        # Test non-one stride and dilation with different layouts
        (
            [1, 18, 24, 32],
            [3, 3],
            [2, 3],
            [2, 2],
            [0, 0, 0, 0],
            False,
            True,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 32, 18, 18],
            [5, 5],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            "NCHW",
            TensorContentRandom(),
        ),
        # Test non-zero padding with count include and exclude pad and different layouts
        (
            [1, 6, 6, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1, 1, 1],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 2],
            [2, 3],
            [2, 2, 3, 3],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 32, 6, 6],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 2, 3, 4],
            False,
            False,
            "NCHW",
            TensorContentRandom(),
        ),
        (
            [1, 32, 15, 22],
            [3, 3],
            [3, 2],
            [2, 3],
            [1, 2, 3, 4],
            False,
            False,
            "NCHW",
            TensorContentRandom(),
        ),
        # Test n11c-1024c-2d layout which will require input and output to have different layout
        (
            [1, 1, 1, 2048],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [6, 6],
            [1, 1],
            [1, 1],
            [2, 2, 2, 2],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [4, 4],
            [2, 2],
            [2, 3],
            [0, 2, 1, 4],
            False,
            False,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [3, 3],
            [2, 2],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "NHWC",
            TensorContentRandom(),
        ),
        (
            [1, 2048, 1, 1],
            [4, 4],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            "NCHW",
            TensorContentRandom(),
        ),
    ]

    _param_ids = get_multitest_ids(_multitest_params, _param_descs)

    (
        output_shape,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        op_layout,
        input_tensor_populator,
    ) = tvm.testing.parameters(*_multitest_params, ids=_param_ids)

    @tvm.testing.fixture
    def expected_output_np(
        self, input_np, kernel, stride, dilation, padding, ceil_mode, count_include_pad, op_layout
    ):
        pad_before = padding[:2]
        pad_after = padding[2:]
        ref_np = poolnd_python(
            input_np,
            kernel,
            stride,
            dilation,
            pad_before,
            pad_after,
            "avg",  # pool_type
            count_include_pad,
            False,  # ceil_mode,
            layout=op_layout,
        )
        return ref_np

    @tvm.testing.fixture
    def input_shape(
        self, output_shape, kernel, padding, stride, dilation, op_layout, output_layout
    ):
        # Input shape without any padding; 'ceil' is being ignored from calculation:
        if op_layout == "NHWC":
            o_b, o_h, o_w, o_c = output_shape
        else:
            o_b, o_c, o_h, o_w = output_shape
        d_h, d_w = dilation
        s_h, s_w = stride
        k_h, k_w = kernel
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h, pad_after_w = padding[2:]

        if (
            output_layout == "n11c-2048c-2d"
            or output_layout == "nc11-2048c-2d"
            or output_layout == "n11c-1024c-2d"
            or output_layout == "nc11-1024c-2d"
        ):
            assert o_h == 1 and o_w == 1, "Output height and width must be 1"

        in_h = (o_h - 1) * s_h + d_h * (k_h - 1) + 1 - pad_before_h - pad_after_h
        in_w = (o_w - 1) * s_w + d_w * (k_w - 1) + 1 - pad_before_w - pad_after_w

        if op_layout == "NHWC":
            return [o_b, in_h, in_w, o_c]
        else:
            return [o_b, o_c, in_h, in_w]

    @tvm.testing.fixture
    def schedule_args(
        self,
        kernel,
        stride,
        padding,
        dilation,
        count_include_pad,
        output_layout,
        output_shape,
        input_np,
        input_shape,
        input_layout,
        expected_output_np,
        dtype,
        op_layout,
    ):
        """Construct schedule args based on dtype"""
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        if dtype == "float16":
            if op_layout == "NHWC":
                M = sl.avg_pool2d_NHWC(
                    A, kernel, stride, padding, dilation, count_include_pad, output_shape
                )
            elif op_layout == "NCHW":
                M = sl.avg_pool2d_NCHW(
                    A, kernel, stride, padding, dilation, count_include_pad, output_shape
                )
            else:
                raise RuntimeError(f"Unsupported layout for slice_op.avg_pool2d '{op_layout}'")
            tir_schedule = sl.avg_pool2d_schedule(M, A, output_layout, input_layout)
        elif dtype in ("uint8", "int8"):
            _, in_scale, in_zero_point = quantize_np(input_np, dtype)
            _, out_scale, out_zero_point = quantize_np(expected_output_np, dtype)
            if op_layout == "NHWC":
                M = qn.qnn_avg_pool2d_NHWC(
                    A,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    count_include_pad,
                    output_shape,
                    dtype,
                    in_scale,
                    in_zero_point,
                    out_scale,
                    out_zero_point,
                )
            elif op_layout == "NCHW":
                M = qn.qnn_avg_pool2d_NCHW(
                    A,
                    kernel,
                    stride,
                    padding,
                    dilation,
                    count_include_pad,
                    output_shape,
                    dtype,
                    in_scale,
                    in_zero_point,
                    out_scale,
                    out_zero_point,
                )
            else:
                raise RuntimeError(f"Unsupported layout for qnn.avg_pool2d '{op_layout}'")

            tir_schedule = qn.qnn_avg_pool2d_schedule(M, A, output_layout, input_layout)

        return [tir_schedule.mod, [A, M]]

    @tvm.testing.requires_hexagon
    def test_avg_pool2d_slice(
        self, dtype, input_np, expected_output_np, schedule_args, hexagon_session: Session
    ):
        print("schedule_args : ", schedule_args)
        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(*schedule_args, get_hexagon_target("v69"), name="avg_pool2d")

        input_axis_separator = []
        output_axis_separator = []

        if dtype == "float16":
            in_data_np = input_np
            out_data_np = expected_output_np
        elif dtype in ("uint8", "int8"):
            in_data_np, _, _ = quantize_np(input_np, dtype)
            out_data_np, _, _ = quantize_np(expected_output_np, dtype)
        else:
            raise RuntimeError(f"Unsupport dtype '{dtype}'")

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=in_data_np,
            axis_separators=input_axis_separator,
            mem_scope="global.ddr",
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            out_data_np.shape,
            dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.ddr",
        )

        mod = hexagon_session.load_module(func)

        mod(input_arr, output_arr)

        output_np = output_arr.numpy()
        if dtype == "float16":
            np.testing.assert_allclose(output_np, out_data_np, rtol=1e-3, atol=1e-3)
        else:
            output_np = output_arr.numpy()
            np.testing.assert_allclose(output_np, out_data_np, rtol=0, atol=2)


if __name__ == "__main__":
    tvm.testing.main()
