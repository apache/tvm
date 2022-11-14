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
"""Test code for Add, Subtract and Multiply."""
import numpy as np

import tvm
from tvm import te
import tvm.topi.hexagon.slice_ops as sl
import tvm.topi.hexagon.qnn as qn
from tvm.contrib.hexagon import allocate_hexagon_array
from ..infrastructure import (
    transform_numpy,
    quantize_np,
    get_hexagon_target,
)

ZERO_POINT_A_VAL = None
SCALE_A_VAL = None

ZERO_POINT_B_VAL = None
SCALE_B_VAL = None

ZERO_POINT_M_VAL = None
SCALE_M_VAL = None


def hexagon_wrapper_allocation(
    device,
    layout,
    axis_separators,
    tensor_shape=None,
    data_original=None,
    transformed_data=None,
    dtype=None,
):
    """Input layout can either be nhwc-8h2w32c2w-2d or nhwc"""
    if layout in ["nhwc-8h2w32c2w-2d", "nhwc-8h8w32c-2d"]:
        data_nd = allocate_hexagon_array(
            device,
            tensor_shape=tensor_shape,
            data=transformed_data,
            dtype=dtype,
            axis_separators=axis_separators,
            mem_scope="global.vtcm",
        )
    elif layout == "nhwc":
        data_nd = allocate_hexagon_array(
            device,
            data=data_original,
        )
    return data_nd


class TestAddSubtractMultiplyBroadcast2d:
    """Test Add, Subtract and Multiply class."""

    (
        input_shape_a,
        input_shape_b,
        input_a_layout,
        input_b_layout,
        output_layout,
        dtype,
    ) = tvm.testing.parameters(
        # no broadcast needed - short input
        (
            [1, 8, 4, 32],
            [1, 8, 4, 32],
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        # no broadcast needed - large input
        (
            [1, 56, 64, 128],
            [1, 56, 64, 128],
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h2w32c2w-2d",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        # one input needs broadcast
        (
            [1, 56, 64, 128],
            [1, 1, 64, 1],
            "nhwc-8h2w32c2w-2d",
            "nhwc",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        # Both input needs broadcast
        (
            [1, 56, 1, 128],
            [1, 1, 64, 1],
            "nhwc",
            "nhwc",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        # One axis in one input needs broadcast
        (
            [1, 56, 20, 128],
            [1, 56, 20, 1],
            "nhwc-8h2w32c2w-2d",
            "nhwc",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        # broadcast all axes in one input
        (
            [1, 48, 56, 32],
            [1, 1, 1, 1],
            "nhwc-8h2w32c2w-2d",
            "nhwc",
            "nhwc-8h2w32c2w-2d",
            "float16",
        ),
        (
            [1, 48, 32, 64],
            [1, 48, 32, 64],
            "nhwc-8h8w32c-2d",
            "nhwc-8h8w32c-2d",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast axis 2 in one input
        (
            [1, 48, 32, 64],
            [1, 48, 1, 64],
            "nhwc-8h8w32c-2d",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast axis 1 in one input
        (
            [1, 48, 32, 64],
            [1, 1, 32, 64],
            "nhwc-8h8w32c-2d",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast axis 3 in one input
        (
            [1, 8, 8, 32],
            [1, 8, 8, 1],
            "nhwc-8h8w32c-2d",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast both inputs
        (
            [1, 56, 1, 128],
            [1, 1, 64, 1],
            "nhwc",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast both inputs
        (
            [1, 48, 1, 1],
            [1, 1, 32, 32],
            "nhwc",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast both inputs
        (
            [1, 48, 1, 32],
            [1, 1, 32, 1],
            "nhwc",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
        # broadcast all axes in one input
        (
            [1, 48, 56, 32],
            [1, 1, 1, 1],
            "nhwc-8h8w32c-2d",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
    )

    op_name = tvm.testing.parameter("add", "subtract", "multiply")

    @tvm.testing.fixture
    def expected_output_np(self, input_np_a, input_np_b, op_name):
        """Generate expected output."""
        if op_name == "add":
            out_ref = np.add(input_np_a, input_np_b)
        elif op_name == "subtract":
            out_ref = np.subtract(input_np_a, input_np_b)
        elif op_name == "multiply":
            out_ref = np.multiply(input_np_a, input_np_b)
        return out_ref

    @tvm.testing.fixture
    def transformed_expected_output_np(self, expected_output_np, output_layout, dtype):
        """Generate expected output."""
        if dtype == "float16":
            return transform_numpy(expected_output_np, "nhwc", output_layout)
        if dtype in ["uint8", "int8"]:
            global ZERO_POINT_M_VAL, SCALE_M_VAL
            out_ref_quantized, SCALE_M_VAL, ZERO_POINT_M_VAL = quantize_np(
                expected_output_np, dtype
            )
            return transform_numpy(out_ref_quantized, "nhwc", output_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def input_np_a(self, input_shape_a, dtype):
        """Generate numpy input for variable a."""
        if dtype in ["uint8", "int8"]:
            dtype = "float32"
        return np.random.random(input_shape_a).astype(dtype)

    @tvm.testing.fixture
    def input_np_b(self, input_shape_b, dtype):
        """Generate numpy input for variable b."""
        if dtype in ["uint8", "int8"]:
            dtype = "float32"
        return np.random.random(input_shape_b).astype(dtype)

    @tvm.testing.fixture
    def quantize_input_np_a(self, input_np_a, dtype):
        if dtype in ["uint8", "int8"]:
            global ZERO_POINT_A_VAL, SCALE_A_VAL
            input_np_a_quantized, SCALE_A_VAL, ZERO_POINT_A_VAL = quantize_np(input_np_a, dtype)
            return input_np_a_quantized
        return None

    @tvm.testing.fixture
    def quantize_input_np_b(self, input_np_b, dtype):
        if dtype in ["uint8", "int8"]:
            global ZERO_POINT_B_VAL, SCALE_B_VAL
            input_np_b_quantized, SCALE_B_VAL, ZERO_POINT_B_VAL = quantize_np(input_np_b, dtype)
            return input_np_b_quantized
        return None

    @tvm.testing.fixture
    def transformed_input_np_a(self, input_np_a, quantize_input_np_a, input_a_layout, dtype):
        if dtype == "float16":
            return transform_numpy(input_np_a, "nhwc", input_a_layout)
        if dtype in ["uint8", "int8"]:
            return transform_numpy(quantize_input_np_a, "nhwc", input_a_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.fixture
    def transformed_input_np_b(self, input_np_b, quantize_input_np_b, input_b_layout, dtype):
        if dtype == "float16":
            return transform_numpy(input_np_b, "nhwc", input_b_layout)
        if dtype in ["uint8", "int8"]:
            return transform_numpy(quantize_input_np_b, "nhwc", input_b_layout)

        raise RuntimeError(f"Unsupported data type '{dtype}'")

    @tvm.testing.requires_hexagon
    def test_transform(
        self,
        dtype,
        input_shape_a,
        input_shape_b,
        input_np_a,
        input_np_b,
        quantize_input_np_a,
        quantize_input_np_b,
        transformed_input_np_a,
        transformed_input_np_b,
        expected_output_np,
        transformed_expected_output_np,
        hexagon_session,
        output_layout,
        input_a_layout,
        input_b_layout,
        op_name,
    ):
        """Test transform."""
        output_shape = expected_output_np.shape
        a_tensor = te.placeholder(input_shape_a, name="a_tensor", dtype=dtype)
        b_tensor = te.placeholder(input_shape_b, name="b_tensor", dtype=dtype)
        if dtype == "float16":
            if op_name == "add":
                m_tensor = sl.add_broadcast_compute(a_tensor, b_tensor)
            elif op_name == "subtract":
                m_tensor = sl.subtract_broadcast_compute(a_tensor, b_tensor)
            elif op_name == "multiply":
                m_tensor = sl.multiply_broadcast_compute(a_tensor, b_tensor)
            tir_schedule = sl.tir_broadcast_schedule(
                m_tensor, a_tensor, b_tensor, output_layout, input_a_layout, input_b_layout, op_name
            )
        elif dtype in ["uint8", "int8"]:
            args = [
                a_tensor,
                b_tensor,
                output_shape,
                ZERO_POINT_A_VAL,
                SCALE_A_VAL,
                ZERO_POINT_B_VAL,
                SCALE_B_VAL,
                ZERO_POINT_M_VAL,
                SCALE_M_VAL,
                dtype,
            ]
            if op_name == "add":
                m_tensor = qn.qadd_broadcast_compute(*args)
            elif op_name == "subtract":
                m_tensor = qn.qsubtract_broadcast_compute(*args)
            elif op_name == "multiply":
                m_tensor = qn.qmultiply_broadcast_compute(*args)
            tir_schedule = qn.tir_schedule_quant(
                m_tensor, a_tensor, b_tensor, output_layout, input_a_layout, input_b_layout
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
                [a_tensor, b_tensor, m_tensor],
                get_hexagon_target("v69"),
                name="slice_op_with_transform",
            )

        if dtype == "float16":
            in_data_np_a = input_np_a
            in_data_np_b = input_np_b
        elif dtype in ["int8", "uint8"]:
            in_data_np_a = quantize_input_np_a
            in_data_np_b = quantize_input_np_b
        else:
            raise RuntimeError(f"Unsupport dtype '{dtype}'")

        a_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_a_layout,
            data_original=in_data_np_a,
            transformed_data=transformed_input_np_a,
            axis_separators=input_axis_separator,
        )
        b_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_b_layout,
            data_original=in_data_np_b,
            transformed_data=transformed_input_np_b,
            axis_separators=input_axis_separator,
        )
        m_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=output_layout,
            tensor_shape=transformed_expected_output_np.shape,
            axis_separators=output_axis_separator,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(a_data_nd, b_data_nd, m_data_nd)

        batch, height, width, channel = output_shape
        # convert nd to np and reshape to fixed chunk size layout
        if output_layout == "nhwc-8h2w32c2w-2d":
            m_data_np = m_data_nd.numpy().reshape(
                [batch, height // 8, width // 4, channel // 32, 8, 2, 32, 2]
            )
        elif output_layout == "nhwc-8h8w32c-2d":
            m_data_np = m_data_nd.numpy().reshape(
                [batch, height // 8, width // 8, channel // 32, 8, 8, 32]
            )

        if dtype == "float16":
            np.testing.assert_allclose(
                transformed_expected_output_np, m_data_np, rtol=1e-3, atol=1e-3
            )
        elif dtype in ["int8", "uint8"]:
            np.testing.assert_allclose(transformed_expected_output_np, m_data_np, rtol=1, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
