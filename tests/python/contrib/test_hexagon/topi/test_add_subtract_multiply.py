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
from ..infrastructure import allocate_hexagon_array, transform_numpy


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
    return np.random.random(input_shape_A).astype(dtype)


@tvm.testing.fixture
def input_np_B(input_shape_B, dtype):
    return np.random.random(input_shape_B).astype(dtype)


@tvm.testing.fixture
def transformed_input_np_A(input_np_A, input_A_layout):
    return transform_numpy(input_np_A, "nhwc", input_A_layout)


@tvm.testing.fixture
def transformed_input_np_B(input_np_B, input_B_layout):
    return transform_numpy(input_np_B, "nhwc", input_B_layout)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, "nhwc", output_layout)


def hexagon_wrapper_allocation(
    device, layout, axis_separators, tensor_shape=None, data=None, transformed_data=None, dtype=None
):
    """Input layout can either be nhwc-8h2w32c2w-2d or nhwc"""
    if layout == "nhwc-8h2w32c2w-2d":
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
            data=data,
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
        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape_A, name="A", dtype=dtype)
        B = te.placeholder(input_shape_B, name="B", dtype=dtype)
        if op_name == "add":
            M = sl.add_broadcast_compute(A, B)
        elif op_name == "subtract":
            M = sl.subtract_broadcast_compute(A, B)
        elif op_name == "multiply":
            M = sl.multiply_broadcast_compute(A, B)

        tir_schedule = sl.tir_broadcast_schedule(
            M, A, B, output_layout, input_A_layout, input_B_layout, op_name
        )
        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout == "nhwc-8h2w32c2w-2d":
            output_axis_separator = [4]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [A, B, M],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="slice_op_with_transform",
            )

        output_shape = expected_output_np.shape

        A_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_A_layout,
            data=input_np_A,
            transformed_data=transformed_input_np_A,
            axis_separators=input_axis_separator,
        )
        B_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_B_layout,
            data=input_np_B,
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

        np.testing.assert_allclose(transformed_expected_output_np, M_data_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
