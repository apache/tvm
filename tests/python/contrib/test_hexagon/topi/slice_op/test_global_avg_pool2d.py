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

"""Test code for float16 and uint8 global_avg_pool2d."""

import numpy as np

import tvm
from tvm import te
from tvm.topi.testing import adaptive_pool
import tvm.topi.hexagon.qnn as qn
import tvm.topi.hexagon.slice_ops as sl
from tvm.contrib.hexagon import allocate_hexagon_array
from ...infrastructure import transform_numpy, quantize_np, get_hexagon_target


SCALE_M_VAL = None
ZERO_POINT_M_VAL = None
SCALE_VAL = None
ZERO_POINT_VAL = None


class TestGlobalPool2D:
    (input_shape,) = tvm.testing.parameters(
        ([1, 32, 8, 8],),
        ([1, 1056, 16, 16],),
    )

    # Fixed chunk layout is set as nchw-32c8h8w-2d for uint8 and nchw-32c8h4w-2d for float16.
    # For optimization, it might get changed later.
    # Since output shape will be NxCx1x1 which is not a
    # multiple of fixed-chunk, output_layout is NCHW.
    input_layout, output_layout, pool_type, layout, dtype = tvm.testing.parameters(
        ("nchw-32c8h8w-2d", "nchw", "avg", "NCHW", "uint8"),
        ("nchw-32c8h4w-2d", "nchw", "avg", "NCHW", "float16"),
    )

    @tvm.testing.fixture
    def expected_output_np(
        self,
        input_np,
        pool_type,
        layout,
    ):
        """Generate expected output."""
        ref_np = tvm.topi.testing.adaptive_pool(
            input_np,
            (1, 1),
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

    @tvm.testing.fixture
    def transformed_input_np(self, input_np, quantize_input_np, input_layout, layout, dtype):
        if dtype == "float16":
            return transform_numpy(input_np, layout.lower(), input_layout)
        if dtype in ("uint8", "int8"):
            return transform_numpy(quantize_input_np, layout.lower(), input_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def quantize_expected_output_np(self, expected_output_np, dtype):
        if dtype in ("uint8", "int8"):
            global ZERO_POINT_M_VAL, SCALE_M_VAL
            out_ref_quantized, SCALE_M_VAL, ZERO_POINT_M_VAL = quantize_np(
                expected_output_np, dtype
            )

            # Since output_layout is nchw, no transformation is needed.
            return out_ref_quantized

    @tvm.testing.requires_hexagon
    def test_global_pool2d(
        self,
        dtype,
        input_shape,
        input_layout,
        transformed_input_np,
        expected_output_np,
        quantize_expected_output_np,
        hexagon_session,
    ):
        a_tensor = te.placeholder(input_shape, name="a_tensor", dtype=dtype)

        if dtype == "float16":
            m_tensor = sl.global_avg_pool2d(a_tensor)
            tir_schedule = sl.stir_global_avg_pool2d_schedule(m_tensor, a_tensor, input_layout)
        elif dtype in ["uint8", "int8"]:
            m_tensor = qn.global_avg_pool2d_u8(
                a_tensor,
                dtype,
                ZERO_POINT_VAL,
                SCALE_VAL,
                ZERO_POINT_M_VAL,
                SCALE_M_VAL,
            )
            tir_schedule = qn.stir_global_avg_pool2d_u8_schedule(m_tensor, a_tensor, input_layout)

        sch = tir_schedule.mod

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [a_tensor, m_tensor],
                get_hexagon_target("v69"),
                name="global_pool2d",
            )

        input_axis_separator = [4]

        a_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        m_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            expected_output_np.shape,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(a_data_nd, m_data_nd)

        # Convert nd to np
        m_data_np = m_data_nd.numpy()

        if dtype == "float16":
            np.testing.assert_allclose(expected_output_np, m_data_np, rtol=1e-3, atol=1e-3)
        elif dtype in ["int8", "uint8"]:
            np.testing.assert_allclose(quantize_expected_output_np, m_data_np, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
