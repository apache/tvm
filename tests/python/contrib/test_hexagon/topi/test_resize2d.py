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

import tvm
from tvm import te
from tvm.topi.testing import resize2d_python
import tvm.topi.hexagon as s1
from ..infrastructure import allocate_hexagon_array, transform_numpy


@tvm.testing.fixture
def expected_output_np(
    input_np, in_height, in_width, out_height, out_width, layout, method, coord_trans
):
    scale_h = out_height / in_height
    scale_w = out_width / in_width
    return resize2d_python(input_np, (scale_h, scale_w), layout, method, coord_trans)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return np.random.random(input_shape).astype(dtype)


@tvm.testing.fixture
def transformed_input_np(input_np, layout, input_crouton_layout):
    return transform_numpy(input_np, layout.lower(), input_crouton_layout)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, layout, output_layout):
    return transform_numpy(expected_output_np, layout.lower(), output_layout)


@tvm.testing.fixture
def input_shape(batch, channel, in_height, in_width):
    return (batch, in_height, in_width, channel)


@tvm.testing.fixture
def output_shape(batch, channel, out_height, out_width):
    return (batch, out_height, out_width, channel)


class TestResize2d:
    (batch, channel, in_height, in_width, out_height, out_width,) = tvm.testing.parameters(
        (
            1,
            32,
            8,
            8,
            16,
            16,
        ),
        (
            1,
            32,
            48,
            48,
            8,
            8,
        ),
    )

    (layout, input_crouton_layout, output_layout, dtype,) = tvm.testing.parameters(
        ("NHWC", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", "float16"),
    )

    coord_trans = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")
    method = tvm.testing.parameter("nearest_neighbor", "linear", "cubic")

    @tvm.testing.requires_hexagon
    def test_resize2d(
        self,
        dtype,
        input_np,
        transformed_input_np,
        input_shape,
        output_shape,
        expected_output_np,
        transformed_expected_output_np,
        layout,
        input_crouton_layout,
        output_layout,
        coord_trans,
        method,
        hexagon_session,
    ):
        if hexagon_session._launcher._serial_number != "simulator":
            pytest.skip(msg="Due to https://github.com/apache/tvm/issues/11957")

        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)

        M = s1.resize2d_compute(
            A,
            [0.0] * 4,
            (output_shape[1], output_shape[2]),
            layout=layout,
            coordinate_transformation_mode=coord_trans,
            method=method,
        )

        tir_schedule = s1.tir_broadcast_schedule(M, A, input_crouton_layout, output_layout)

        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout == "nhwc-8h2w32c2w-2d":
            output_axis_separator = [4]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [A, M],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="resize2d",
            )

        A_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        M_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            dtype=dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )

        mod = hexagon_session.load_module(func)
        mod(A_data_nd, M_data_nd)

        b, h, w, c = output_shape
        # convert nd to np and reshape to fixed chunk size layout
        if output_layout == "nhwc-8h2w32c2w-2d":
            M_data_np = M_data_nd.numpy().reshape([b, h // 8, w // 4, c // 32, 8, 2, 32, 2])

        np.testing.assert_allclose(transformed_expected_output_np, M_data_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
