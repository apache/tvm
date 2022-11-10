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
"""Resize 2D tesst.
"""
import numpy as np

import tvm
from tvm import te
from tvm.topi.testing import resize2d_python
import tvm.topi.hexagon as s1
from tvm.contrib.hexagon import allocate_hexagon_array

from ..infrastructure import transform_numpy, get_hexagon_target


class TestResize2d:
    """Test resize 2D class."""

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
        ("NHWC", "nhwc-8h8w32c-2d", "nhwc-8h8w32c-2d", "uint8"),
    )

    coord_trans = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")
    method = tvm.testing.parameter("nearest_neighbor", "linear")

    @tvm.testing.fixture
    def expected_output_np(
        self,
        input_np,
        in_height,
        in_width,
        out_height,
        out_width,
        layout,
        method,
        coord_trans,
    ):
        """Generate expected output."""
        scale_h = out_height / in_height
        scale_w = out_width / in_width

        return resize2d_python(input_np, (scale_h, scale_w), layout, method, coord_trans)

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        if dtype == "float16":
            return np.random.random(input_shape).astype(dtype)
        if dtype == "uint8":
            return np.random.randint(0, 255, input_shape).astype(dtype)
        if dtype == "int8":
            return np.random.randint(-128, 127, input_shape).astype(dtype)
        raise RuntimeError(f"dtype {dtype} is not valid.")

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, layout, input_crouton_layout, dtype):
        if dtype in ["float16", "uint8", "int8"]:
            return transform_numpy(input_np, layout.lower(), input_crouton_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, layout, output_layout, dtype):
        if dtype in ["float16", "uint8", "int8"]:
            return transform_numpy(expected_output_np, layout.lower(), output_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def input_shape(self, batch, channel, in_height, in_width):
        return (batch, in_height, in_width, channel)

    @tvm.testing.fixture
    def output_shape(self, batch, channel, out_height, out_width):
        return (batch, out_height, out_width, channel)

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
        """Test resize 2D."""
        a_tensor = te.placeholder(input_shape, name="a_tensor", dtype=dtype)

        m_tensor = s1.resize2d_compute(
            a_tensor,
            [0.0] * 4,
            (output_shape[1], output_shape[2]),
            layout=layout,
            coordinate_transformation_mode=coord_trans,
            method=method,
            out_dtype=dtype,
        )

        tir_schedule = s1.tir_resize2d_schedule(
            m_tensor, a_tensor, input_crouton_layout, output_layout
        )

        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout in (
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h8w32c-2d",
        ):
            output_axis_separator = [4]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [a_tensor, m_tensor],
                get_hexagon_target("v69"),
                name="resize2d",
            )

        a_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        m_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            dtype=dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )

        mod = hexagon_session.load_module(func)
        mod(a_data_nd, m_data_nd)

        batch_size, height, width, channel = output_shape
        # convert nd to np and reshape to fixed chunk size layout
        if output_layout == "nhwc-8h2w32c2w-2d":
            m_data_np = m_data_nd.numpy().reshape(
                [batch_size, height // 8, width // 4, channel // 32, 8, 2, 32, 2]
            )
        elif output_layout == "nhwc-8h8w32c-2d":
            m_data_np = m_data_nd.numpy().reshape(
                [batch_size, height // 8, width // 8, channel // 32, 8, 8, 32]
            )

        if dtype == "float16":
            np.testing.assert_allclose(
                transformed_expected_output_np, m_data_np, rtol=1e-3, atol=1e-3
            )
        elif dtype in ["int8", "uint8"]:
            np.testing.assert_allclose(transformed_expected_output_np, m_data_np, rtol=1, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
