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

import pytest
import numpy as np
from typing import *

from tvm import te
import tvm.testing
from tvm.contrib.hexagon.session import Session
import tvm.topi.hexagon.slice_ops as sl
from ..infrastructure import allocate_hexagon_array, transform_numpy, get_hexagon_target
from ..pytest_util import (
    get_multitest_ids,
    create_populated_numpy_ndarray,
    TensorContentRandom,
)


@tvm.testing.fixture
def input_np(input_shape, dtype: str, input_tensor_populator):
    return create_populated_numpy_ndarray(input_shape, dtype, input_tensor_populator)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, "nhwc", output_layout)


@tvm.testing.fixture
def transformed_input_np_padded(input_np_padded, input_layout):
    return transform_numpy(input_np_padded, "nhwc", input_layout)


(input_layout, dtype) = tvm.testing.parameters(
    ("nhwc-8h2w32c2w-2d", "float16"),
    ("nhwc-8h8w32c-2d", "uint8"),
)


@tvm.testing.fixture
def output_layout(output_shape, dtype):
    o_b, o_h, o_w, o_c = output_shape
    if dtype == "float16":
        if o_h == 1 and o_w == 1:
            return "n11c-1024c-2d"
        else:
            assert o_h % 8 == 0 and o_w % 4 == 0, "Invalid output shape"
            return "nhwc-8h2w32c2w-2d"
    elif dtype == "int8" or "uint8":
        if o_h == 1 and o_w == 1:
            return "n11c-2048c-2d"
        else:
            assert o_h % 8 == 0 and o_w % 8 == 0, "Invalid output shape"
            return "nhwc-8h8w32c-2d"
    else:
        raise RuntimeError(f"Unsupported data type '{dtype}'")


class TestmaxPool2dSlice:
    _param_descs = [
        "out_shape",  # output_shape
        "kernel",  # kernel
        "stride",  # stride
        "dil",  # dilation
        "pad",  # padding
        "ceil",  # ceil_mode
        "cnt_padded",  # count_include_pad
        None,  # input_tensor_populator
    ]

    _multitest_params = [
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        (
            [1, 16, 16, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        (
            [1, 8, 8, 32],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        # Test non-one stride and dilation
        (
            [1, 8, 8, 32],
            [3, 3],
            [2, 3],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [2, 2],
            [2, 2],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
        # Test non-zero padding
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 1, 1, 1],
            False,
            True,
            TensorContentRandom(),
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 2, 3, 4],
            False,
            True,
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
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [6, 6],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
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
            TensorContentRandom(),
        ),
        (
            [1, 1, 1, 2048],
            [4, 4],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            TensorContentRandom(),
        ),
    ]

    _param_ids = get_multitest_ids(_multitest_params, _param_descs)

    # NOTE: input_layout is always assumed to be "nhwc-8h2w32c2w-2d" for float16
    # and "nhwc-8h8w32c-2d" for uint8
    (
        output_shape,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        input_tensor_populator,
    ) = tvm.testing.parameters(*_multitest_params, ids=_param_ids)

    @tvm.testing.fixture
    def expected_output_np(
        self,
        input_np,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
    ):
        pad_before = padding[:2]
        pad_after = padding[2:]
        ref_np = tvm.topi.testing.poolnd_python(
            input_np,
            kernel,
            stride,
            dilation,
            pad_before,
            pad_after,
            "max",  # pool_type
            count_include_pad,
            False,  # ceil_mode,
            layout="NHWC",
        )
        return ref_np

    @tvm.testing.fixture
    def input_shape(self, output_shape, kernel, padding, stride, dilation, output_layout):
        # Input shape without any padding; 'ceil' is being ignored from calculation:
        o_b, o_h, o_w, o_c = output_shape
        d_h, d_w = dilation
        s_h, s_w = stride
        k_h, k_w = kernel
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h, pad_after_w = padding[2:]

        if output_layout == "n11c-1024c-2d":
            assert (
                pad_before_w == 0 and pad_after_w == 0 and pad_before_h == 0 and pad_after_h == 0
            ), "Padding must be zero for n11c-1024c-2d layout"
            assert o_h == 1 and o_w == 1, "Output height and width must be 1"

        in_h = (o_h - 1) * s_h + d_h * (k_h - 1) + 1 - pad_before_h - pad_after_h
        in_w = (o_w - 1) * s_w + d_w * (k_w - 1) + 1 - pad_before_w - pad_after_w

        return [o_b, in_h, in_w, o_c]

    @tvm.testing.fixture
    def input_shape_padded(self, dtype, input_shape, padding, output_layout):
        # Input shape is adjusted to account for 'padding'. Also, due to the physical
        # layout of the buffer, height and width are adjusted so that they are a
        # multiple of 8 and 4 respectively.
        # NOTE: For float16, the input layout is always assumed to be nhwc-8h2w32c2w-2d and
        # for int8/uint8, it's nhwc-8h8w32c-2d.
        # For both nhwc-8h2w32c2w-2d and nhwc-8h8w32c-2d, the height should be a multiple
        # of 8. However, the width should be a multiple of 4 for the first case and 8 for
        # the second case.

        height_mult = 8
        if dtype == "float16":
            width_mult = 4  # input layout : nhwc-8h2w32c2w-2d
        elif dtype in ("uint8", "int8"):
            width_mult = 8  # input layout : nhwc-8h8w32c-2d
        else:
            raise RuntimeError(f"Unsupport dtype '{dtype}'")

        pad_before_h, pad_before_w = padding[:2]
        pad_after_h, pad_after_w = padding[2:]
        padded_input_height = (
            (input_shape[1] + pad_before_h + pad_after_h + height_mult - 1) // height_mult
        ) * height_mult
        padded_input_width = (
            (input_shape[2] + pad_before_w + pad_after_w + width_mult - 1) // width_mult
        ) * width_mult
        return [input_shape[0], padded_input_height, padded_input_width, input_shape[3]]

    @tvm.testing.fixture
    def input_np_padded(self, input_np, input_shape, input_shape_padded, padding):
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h = input_shape_padded[1] - input_shape[1] - pad_before_h
        pad_after_w = input_shape_padded[2] - input_shape[2] - pad_before_w

        input_padded = np.pad(
            input_np,
            ((0, 0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
            "constant",
        )
        return input_padded

    @tvm.testing.requires_hexagon
    def test_max_pool2d_slice(
        self,
        stride,
        kernel,
        dtype,
        dilation,
        padding,
        ceil_mode,  # only needed for manually obtaining the test id string
        input_tensor_populator,  # only needed for manually obtaining the test id string
        count_include_pad,
        input_layout,
        output_layout,
        output_shape,
        input_shape,
        input_shape_padded,
        input_np,
        input_np_padded,
        transformed_input_np_padded,
        transformed_expected_output_np,
        expected_output_np,
        hexagon_session: Session,
    ):
        A = te.placeholder(input_shape_padded, name="A", dtype=dtype)

        M = sl.max_pool2d_compute(A, output_shape, kernel, stride, dilation)

        # tir schedule
        tir_schedule = sl.max_pool2d_STIR_schedule(M, A, output_layout, input_layout)
        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout in (
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h8w32c-2d",
            "n11c-1024c-2d",
            "n11c-2048c-2d",
        ):
            output_axis_separator = [4]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [A, M],
                get_hexagon_target("v69"),
                name="max_pool2d",
            )

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np_padded,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )

        mod = hexagon_session.load_module(func)
        mod(input_arr, output_arr)
        b, h, w, c = output_shape
        if output_layout == "nhwc-8h2w32c2w-2d":
            output_np = output_arr.numpy().reshape([b, h // 8, w // 4, c // 32, 8, 2, 32, 2])
        elif output_layout == "nhwc-8h8w32c-2d":
            output_np = output_arr.numpy().reshape([b, h // 8, w // 8, c // 32, 8, 8, 32])
        elif output_layout == "n11c-2048c-2d":
            output_np = output_arr.numpy().reshape([b, 1, 1, c // 2048, 2048])
        elif output_layout == "n11c-1024c-2d":
            output_np = output_arr.numpy().reshape([b, 1, 1, c // 1024, 1024])
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        if dtype == "float16":
            np.testing.assert_allclose(
                output_np, transformed_expected_output_np, rtol=1e-3, atol=1e-3
            )
        elif dtype == "uint8":
            np.testing.assert_allclose(output_np, transformed_expected_output_np, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
