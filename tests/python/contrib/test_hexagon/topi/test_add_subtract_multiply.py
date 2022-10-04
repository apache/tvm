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
import tvm.topi.hexagon.slice_ops as sl
import tvm.topi.hexagon.qnn as qn
from ..infrastructure import (
    allocate_hexagon_array,
    transform_numpy,
    quantize_np,
    get_hexagon_target,
)


@tvm.testing.fixture
def expected_output_np(input_np_A, input_np_B, op_name):
    if op_name == "add":
        out_ref = np.add(input_np_A, input_np_B)
    elif op_name == "subtract":
        out_ref = np.subtract(input_np_A, input_np_B)
    elif op_name == "multiply":
        out_ref = np.multiply(input_np_A, input_np_B)
    return out_ref


@tvm.testing.fixture
def input_np_A(input_shape_A, dtype):
    if dtype == "uint8" or dtype == "int8":
        dtype = "float32"
    return np.random.random(input_shape_A).astype(dtype)


@tvm.testing.fixture
def input_np_B(input_shape_B, dtype):
    if dtype == "uint8" or dtype == "int8":
        dtype = "float32"
    return np.random.random(input_shape_B).astype(dtype)


@tvm.testing.fixture
def quantize_input_np_A(input_np_A, dtype):
    if dtype == "uint8" or dtype == "int8":
        global zero_point_A_val, scale_A_val
        input_np_A_quantized, scale_A_val, zero_point_A_val = quantize_np(input_np_A, dtype)
        return input_np_A_quantized


@tvm.testing.fixture
def quantize_input_np_B(input_np_B, dtype):
    if dtype == "uint8" or dtype == "int8":
        global zero_point_B_val, scale_B_val
        input_np_B_quantized, scale_B_val, zero_point_B_val = quantize_np(input_np_B, dtype)
        return input_np_B_quantized


@tvm.testing.fixture
def transformed_input_np_A(input_np_A, quantize_input_np_A, input_A_layout, dtype):
    if dtype == "float16":
        return transform_numpy(input_np_A, "nhwc", input_A_layout)
    if dtype == "uint8" or dtype == "int8":
        return transform_numpy(quantize_input_np_A, "nhwc", input_A_layout)

    raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def transformed_input_np_B(input_np_B, quantize_input_np_B, input_B_layout, dtype):
    if dtype == "float16":
        return transform_numpy(input_np_B, "nhwc", input_B_layout)
    if dtype == "uint8" or dtype == "int8":
        return transform_numpy(quantize_input_np_B, "nhwc", input_B_layout)

    raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout, dtype):
    if dtype == "float16":
        return transform_numpy(expected_output_np, "nhwc", output_layout)
    if dtype == "uint8" or dtype == "int8":
        global zero_point_M_val, scale_M_val
        out_ref_quantized, scale_M_val, zero_point_M_val = quantize_np(expected_output_np, dtype)
        return transform_numpy(out_ref_quantized, "nhwc", output_layout)

    raise RuntimeError(f"Unsupported data type '{dtype}'")


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
    if layout == "nhwc-8h2w32c2w-2d" or layout == "nhwc-8h8w32c-2d":
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
    (
        input_shape_A,
        input_shape_B,
        input_A_layout,
        input_B_layout,
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

    @tvm.testing.requires_hexagon
    def test_transform(
        self,
        dtype,
        input_shape_A,
        input_shape_B,
        input_np_A,
        input_np_B,
        quantize_input_np_A,
        quantize_input_np_B,
        transformed_input_np_A,
        transformed_input_np_B,
        expected_output_np,
        transformed_expected_output_np,
        hexagon_session,
        output_layout,
        input_A_layout,
        input_B_layout,
        op_name,
    ):
        output_shape = expected_output_np.shape
        A = te.placeholder(input_shape_A, name="A", dtype=dtype)
        B = te.placeholder(input_shape_B, name="B", dtype=dtype)
        if dtype == "float16":
            if op_name == "add":
                M = sl.add_broadcast_compute(A, B)
            elif op_name == "subtract":
                M = sl.subtract_broadcast_compute(A, B)
            elif op_name == "multiply":
                M = sl.multiply_broadcast_compute(A, B)
            tir_schedule = sl.tir_broadcast_schedule(
                M, A, B, output_layout, input_A_layout, input_B_layout, op_name
            )
        elif dtype == "uint8" or dtype == "int8":
            args = [
                A,
                B,
                output_shape,
                zero_point_A_val,
                scale_A_val,
                zero_point_B_val,
                scale_B_val,
                zero_point_M_val,
                scale_M_val,
                dtype,
            ]
            if op_name == "add":
                M = qn.qadd_broadcast_compute(*args)
            elif op_name == "subtract":
                M = qn.qsubtract_broadcast_compute(*args)
            elif op_name == "multiply":
                M = qn.qmultiply_broadcast_compute(*args)
            tir_schedule = qn.tir_schedule_quant(
                M, A, B, output_layout, input_A_layout, input_B_layout
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
                [A, B, M],
                get_hexagon_target("v69"),
                name="slice_op_with_transform",
            )

        if dtype == "float16":
            in_data_np_A = input_np_A
            in_data_np_B = input_np_B
        elif dtype == "int8" or dtype == "uint8":
            in_data_np_A = quantize_input_np_A
            in_data_np_B = quantize_input_np_B
        else:
            raise RuntimeError(f"Unsupport dtype '{dtype}'")

        A_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_A_layout,
            data_original=in_data_np_A,
            transformed_data=transformed_input_np_A,
            axis_separators=input_axis_separator,
        )
        B_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_B_layout,
            data_original=in_data_np_B,
            transformed_data=transformed_input_np_B,
            axis_separators=input_axis_separator,
        )
        M_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=output_layout,
            tensor_shape=transformed_expected_output_np.shape,
            axis_separators=output_axis_separator,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(A_data_nd, B_data_nd, M_data_nd)

        b, h, w, c = output_shape
        # convert nd to np and reshape to fixed chunk size layout
        if output_layout == "nhwc-8h2w32c2w-2d":
            M_data_np = M_data_nd.numpy().reshape([b, h // 8, w // 4, c // 32, 8, 2, 32, 2])
        elif output_layout == "nhwc-8h8w32c-2d":
            M_data_np = M_data_nd.numpy().reshape([b, h // 8, w // 8, c // 32, 8, 8, 32])

        if dtype == "float16":
            np.testing.assert_allclose(
                transformed_expected_output_np, M_data_np, rtol=1e-3, atol=1e-3
            )
        elif dtype == "int8" or dtype == "uint8":
            np.testing.assert_allclose(transformed_expected_output_np, M_data_np, rtol=1, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
