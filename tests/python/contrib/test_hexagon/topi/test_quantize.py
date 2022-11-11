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
"""TIR quantize schedule tests."""
import numpy as np

import tvm
from tvm import te
import tvm.topi.hexagon.qnn as s1
from tvm.contrib.hexagon import allocate_hexagon_array
from ..infrastructure import (
    transform_numpy,
    quantize_np,
    get_hexagon_target,
)

QUANTIZE_SCALE = None
QUANTIZE_ZERO_POINT = None


class TestQuantize:
    """Test quantize class."""

    @tvm.testing.fixture
    def expected_output_np(self, input_np, output_dtype):
        global QUANTIZE_SCALE, QUANTIZE_ZERO_POINT
        quant_np, QUANTIZE_SCALE, QUANTIZE_ZERO_POINT = quantize_np(input_np, output_dtype)
        return quant_np

    @tvm.testing.fixture
    def input_np(self, input_shape, input_dtype):
        return np.random.random(input_shape).astype(input_dtype)

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, input_crouton_layout):
        return transform_numpy(input_np, "nhwc", input_crouton_layout)

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, output_layout):
        return transform_numpy(expected_output_np, "nhwc", output_layout)

    input_crouton_layout, output_layout, input_dtype = tvm.testing.parameters(
        ("nhwc-4h2w32c2w-2d", "nhwc-8h8w32c-2d", "float32"),
    )

    output_dtype = tvm.testing.parameter("uint8", "int8")

    input_shape = tvm.testing.parameter(
        (1, 8, 8, 32), (1, 16, 16, 32), (1, 16, 16, 128), (1, 64, 64, 64)
    )

    @tvm.testing.requires_hexagon
    def test_quantize(
        self,
        input_dtype,
        output_dtype,
        transformed_input_np,
        input_shape,
        expected_output_np,
        transformed_expected_output_np,
        input_crouton_layout,
        output_layout,
        hexagon_session,
    ):
        """Test quantize."""
        a_tensor = te.placeholder(input_shape, name="a_tensor", dtype=input_dtype)

        m_tensor = s1.quantize_compute(a_tensor, QUANTIZE_SCALE, QUANTIZE_ZERO_POINT, output_dtype)

        tir_schedule = s1.tir_quantize_schedule(
            m_tensor, a_tensor, input_crouton_layout, output_layout
        )

        sch = tir_schedule.mod

        input_axis_separator = [4]
        output_axis_separator = [4]

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [a_tensor, m_tensor],
                get_hexagon_target("v69"),
                name="quantize",
            )

        a_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=input_dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        m_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            tensor_shape=transformed_expected_output_np.shape,
            dtype=output_dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )

        mod = hexagon_session.load_module(func)
        mod(a_data_nd, m_data_nd)

        b, h, weight, c = expected_output_np.shape

        # convert nd to np and reshape to fixed chunk size layout
        m_data_np = m_data_nd.numpy().reshape([b, h // 8, weight // 8, c // 32, 8, 8, 32])

        np.testing.assert_allclose(transformed_expected_output_np, m_data_np, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
