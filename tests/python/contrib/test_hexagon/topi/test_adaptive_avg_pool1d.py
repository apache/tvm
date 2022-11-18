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

"""Test code for specialized case of adaptive_avg_pool1d."""

import numpy as np

import tvm
from tvm import te
from tvm.topi.testing import adaptive_pool
import tvm.topi.hexagon.qnn as s1
from tvm.contrib.hexagon import allocate_hexagon_array
from ..infrastructure import transform_numpy, quantize_np


SCALE_M_VAL = None
ZERO_POINT_M_VAL = None
SCALE_VAL = None
ZERO_POINT_VAL = None


class TestAdaptivePool1D:
    """Test specialized case of adaptive_avg_pool1d."""

    (input_shape,) = tvm.testing.parameters(
        ([1, 128, 128],),
        ([1, 64, 64],),
        ([1, 64, 128],),
        ([1, 32, 64],),
        ([1, 128, 768],),
    )

    # Fixed chunk layout is set as ncw-32c64w-2d for now.
    # The adaptive_avg_pool1d implementation only handles specialized case
    # where output_size is 1 as it appears on quantized distilbert model.
    # Since output size won't be a multiple of fixed-chunk,
    # output_layout is ncw.
    # For optimization, it might get changed later.
    input_layout, output_layout, pool_type, layout, output_size, dtype, = tvm.testing.parameters(
        (
            "ncw-32c64w-2d",
            "ncw",
            "avg",
            "NCW",
            [1],
            "uint8",
        )
    )

    @tvm.testing.fixture
    def expected_output_np(
        self,
        input_np,
        output_size,
        pool_type,
        layout,
    ):
        """Generate expected output."""
        out_width = output_size[0]

        ref_np = adaptive_pool(
            input_np,
            out_width,
            pool_type,
            layout,
        )
        return ref_np

    @tvm.testing.fixture
    def input_np(self, input_shape, dtype):
        if dtype in ("uint8", "int8"):
            dtype = "float32"
        return np.random.random(input_shape).astype(dtype)

    @tvm.testing.fixture
    def quantize_input_np(self, input_np, dtype):
        if dtype in ("uint8", "int8"):
            global ZERO_POINT_VAL, SCALE_VAL
            input_np_quantized, SCALE_VAL, ZERO_POINT_VAL = quantize_np(input_np, dtype)
            return input_np_quantized

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def transformed_input_np(self, quantize_input_np, input_layout, layout, dtype):
        if dtype in ("uint8", "int8"):
            return transform_numpy(quantize_input_np, layout.lower(), input_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def quantize_expected_output_np(self, expected_output_np, dtype):
        """Generate expected output."""
        if dtype in ("uint8", "int8"):
            global ZERO_POINT_M_VAL, SCALE_M_VAL
            out_ref_quantized, SCALE_M_VAL, ZERO_POINT_M_VAL = quantize_np(
                expected_output_np, dtype
            )

            # Since output_layout is ncw, no transformation is needed.
            return out_ref_quantized

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.requires_hexagon
    def test_pool1d(
        self,
        dtype,
        output_size,
        input_layout,
        output_layout,
        input_shape,
        transformed_input_np,
        quantize_expected_output_np,
        hexagon_session,
    ):
        """Test adaptive_avg_pool1d."""
        target_hexagon = tvm.target.hexagon("v69")
        a_tensor = te.placeholder(input_shape, name="a_tensor", dtype=dtype)

        m_tensor = s1.adaptive_avg_pool1d(
            a_tensor,
            output_size,
            dtype,
            ZERO_POINT_VAL,
            SCALE_VAL,
            ZERO_POINT_M_VAL,
            SCALE_M_VAL,
        )

        tir_schedule = s1.tir_adaptive_avg_pool1d_schedule(
            m_tensor, a_tensor, output_layout, input_layout
        )

        sch = tir_schedule.mod

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [a_tensor, m_tensor],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="adaptive_pool1d",
            )

        input_axis_separator = [3]

        a_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        m_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            quantize_expected_output_np.shape,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(a_data_nd, m_data_nd)

        # Convert nd to np
        m_data_np = m_data_nd.numpy()

        np.testing.assert_allclose(quantize_expected_output_np, m_data_np, atol=2)


if __name__ == "__main__":
    tvm.testing.main()
