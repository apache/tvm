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

np.set_printoptions(threshold=np.inf)
from tvm import te, topi
from tvm.tir.stmt_functor import post_order_visit

import tvm.testing
from tvm.topi import testing
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.topi.hexagon.slice_ops as sl
from .infrastructure import allocate_hexagon_array, transform_numpy


input_layout = tvm.testing.parameter(
    "nhwc-8h2w32c2w",
)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return np.random.random(input_shape).astype(dtype)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, output_layout)


@tvm.testing.fixture
def transformed_input_np_padded(input_np_padded, input_layout):
    return transform_numpy(input_np_padded, input_layout)


class TestAvgPool2dSlice:
    # NOTE: input_layout is always assumed to be "nhwc-8h2w32c2w"
    (
        output_shape,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        output_layout,
        dtype,
        use_te_sched,
    ) = tvm.testing.parameters(
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 16, 16, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
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
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [2, 2],
            [2, 2],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
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
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 2, 3, 4],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [1, 2, 3, 4],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        (
            [1, 8, 8, 32],
            [3, 3],
            [3, 2],
            [2, 3],
            [1, 2, 3, 4],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            False,
        ),
        # Test n11c-1024c layout which will require input and output to have different layout
        (
            [1, 1, 1, 2048],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "n11c-1024c",
            "float16",
            False,
        ),
        (
            [1, 1, 1, 2048],
            [6, 6],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "n11c-1024c",
            "float16",
            False,
        ),
        (
            [1, 1, 1, 2048],
            [3, 3],
            [2, 2],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "n11c-1024c",
            "float16",
            False,
        ),
        (
            [1, 1, 1, 2048],
            [4, 4],
            [2, 2],
            [2, 3],
            [0, 0, 0, 0],
            False,
            True,
            "n11c-1024c",
            "float16",
            False,
        ),
        # Use 'te' schedule
        (
            [1, 8, 8, 32],
            [3, 3],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "nhwc-8h2w32c2w",
            "float16",
            True,
        ),
        (
            [1, 1, 1, 2048],
            [8, 8],
            [1, 1],
            [1, 1],
            [0, 0, 0, 0],
            False,
            True,
            "n11c-1024c",
            "float16",
            True,
        ),
    )

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
            "avg",  # pool_type
            count_include_pad,
            False,  # ceil_mode,
            layout="NHWC",
        )
        return ref_np

    @tvm.testing.fixture
    def input_shape(self, output_shape, kernel, padding, stride, dilation, output_layout):
        # Input shape without crouton padding; 'ceil' is being ignored from calculation:
        o_b, o_h, o_w, o_c = output_shape
        d_h, d_w = dilation
        s_h, s_w = stride
        k_h, k_w = kernel
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h, pad_after_w = padding[2:]

        if output_layout == "n11c-1024c":
            assert (
                pad_before_w == 0 and pad_after_w == 0 and pad_before_h == 0 and pad_after_h == 0
            ), "Padding must be zero for n11c-1024c layout!!"
            assert o_h == 1 and o_w == 1, "Output height and width must be 1!"

        in_h = (o_h - 1) * s_h + d_h * (k_h - 1) + 1 - pad_before_h - pad_after_h
        in_w = (o_w - 1) * s_w + d_w * (k_w - 1) + 1 - pad_before_w - pad_after_w

        return [o_b, in_h, in_w, o_c]

    @tvm.testing.fixture
    def input_shape_padded(self, input_shape, padding, output_layout):
        # Input shape with regular and crouton padding.
        # Input width and height are padded to a multiple of croutons.
        # NOTE: Input layout is always assumed to be nhwc-8h2w32c2w. Only the output layout can be
        # different.
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h, pad_after_w = padding[2:]
        padded_input_height = ((input_shape[1] + pad_before_h + pad_after_h + 7) // 8) * 8
        padded_input_width = ((input_shape[2] + pad_before_w + pad_after_w + 3) // 4) * 4
        return [input_shape[0], padded_input_height, padded_input_width, input_shape[3]]

    @tvm.testing.fixture
    def input_np_padded(self, input_np, input_shape, input_shape_padded, padding):
        pad_before_h, pad_before_w = padding[:2]
        pad_after_h = (
            input_shape_padded[1] - input_shape[1] - pad_before_h
        )  # pad_after for height with crouton padding
        pad_after_w = (
            input_shape_padded[2] - input_shape[2] - pad_before_w
        )  # pad_after for width with crouton padding
        input_padded = np.pad(
            input_np,
            ((0, 0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
            "constant",
        )
        return input_padded

    @tvm.testing.requires_hexagon
    def test_avg_pool2d_slice(
        self,
        use_te_sched,
        stride,
        kernel,
        dtype,
        dilation,
        padding,
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
        hexagon_session,
    ):

        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape_padded, name="A", dtype=dtype)

        M = sl.avg_pool2d_compute(
            A, output_shape, kernel, stride, dilation)
        if not use_te_sched:
            # tir schedule
            tir_schedule = sl.avg_pool2d_STIR_schedule(M, A, output_layout, input_layout)
            sch = tir_schedule.mod
        else:
            # te schedule
            te_s = sl.avg_pool2d_schedule(M, A, output_layout, input_layout)
            sch = te_s

        input_axis_separator = [4]
        if output_layout == "nhwc-8h2w32c2w":
            output_axis_separator = [4]
        elif output_layout == "n11c-1024c":
            output_axis_separator = [2]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_assert": True}):
            func = tvm.build(
                sch,
                [A, M],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="avg_pool2d",
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
        if output_layout == "nhwc-8h2w32c2w":
            output_np = output_arr.numpy().reshape([b, h // 8, w // 4, c // 32, 8, 2, 32, 2])
        elif output_layout == "n11c-1024c":
            output_np = output_arr.numpy().reshape([b, c // 1024, 1024])
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
