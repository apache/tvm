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
import sys

import tvm
from tvm import te
import tvm.topi.hexagon.slice_ops as sl
from ..infrastructure import allocate_hexagon_array, transform_numpy, getZeroPoint_Scale, quantize

np.set_printoptions(threshold=sys.maxsize)


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
def input_np_A(input_shape_A):
    a = np.zeros(input_shape_A, "float32")
    shape = input_shape_A
    b = shape[0]
    ch = shape[1]
    cw = shape[2]
    cc = shape[3]
    val = 0.0
    for i in range(b):
        for j in range(ch):
            for l in range(cw):
                for k in range(cc):
                    if j > 0:
                        val = l + j + 1.0
                    else:
                        val = l
                    a[i][j][l][k] = val

    return a


@tvm.testing.fixture
def input_np_B(input_shape_B):
    a = np.zeros(input_shape_B, "float32")
    shape = input_shape_B
    b = shape[0]
    ch = shape[1]
    cw = shape[2]
    cc = shape[3]
    val = 0.0
    for i in range(b):
        for j in range(ch):
            for l in range(cw):
                for k in range(cc):
                    if j > 0:
                        val = l - j - 1.01
                    else:
                        val = l
                    a[i][j][l][k] = val
    return a


@tvm.testing.fixture
def transformed_input_np_A(input_np_A, input_A_layout):
    global zero_point_A_val, scale_A_val, input_np_A_quantized
    zero_point_A_val, scale_A_val, input_np_A_quantized = quantize(input_np_A)
    return transform_numpy(input_np_A_quantized, "nhwc", input_A_layout)


@tvm.testing.fixture
def transformed_input_np_B(input_np_B, input_B_layout):
    global zero_point_B_val, scale_B_val, input_np_B_quantized
    zero_point_B_val, scale_B_val, input_np_B_quantized = quantize(input_np_B)
    return transform_numpy(input_np_B_quantized, "nhwc", input_B_layout)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    global zero_point_M_val, scale_M_val, out_ref_quantized
    zero_point_M_val, scale_M_val, out_ref_quantized = quantize(expected_output_np)
    return transform_numpy(out_ref_quantized, "nhwc", output_layout)


def hexagon_wrapper_allocation(
    device, layout, axis_separators, tensor_shape=None, data=None, transformed_data=None, dtype=None
):
    """Input layout can either be nhwc-8h8w32c-2d or nhwc"""
    if layout == "nhwc-8h8w32c-2d":
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
        # broadcast axis 2 in one input
        (
            [1, 8, 8, 32],
            [1, 8, 8, 1],
            "nhwc-8h8w32c-2d",
            "nhwc",
            "nhwc-8h8w32c-2d",
            "uint8",
        ),
    )

    op_name = tvm.testing.parameter("add")  # , "subtract", "multiply")

    @tvm.testing.requires_hexagon
    def test_transform(
        self,
        dtype,
        input_shape_A,
        input_shape_B,
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
        target_hexagon = tvm.target.hexagon("v69")

        A = te.placeholder(input_shape_A, name="A", dtype=dtype)
        B = te.placeholder(input_shape_B, name="B", dtype=dtype)

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
            ]

        if op_name == "add":
            M = sl.qadd_broadcast_compute(*args)
        elif op_name == "subtract":
            M = sl.qsubtract_broadcast_compute(*args)
        elif op_name == "multiply":
            M = sl.qmultiply_broadcast_compute(*args)

        tir_schedule = sl.tir_schedule(
            M, A, B, output_layout, input_A_layout, input_B_layout, op_name
        )
        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout == "nhwc-8h8w32c-2d":
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

        A_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_A_layout,
            data=input_np_A_quantized,
            transformed_data=transformed_input_np_A,
            axis_separators=input_axis_separator,
        )
        B_data_nd = hexagon_wrapper_allocation(
            hexagon_session.device,
            layout=input_B_layout,
            data=input_np_B_quantized,
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
        if output_layout == "nhwc-8h8w32c-2d":
            M_data_np = M_data_nd.numpy().reshape([b, h // 8, w // 8, c // 32, 8, 8, 32])

        np.testing.assert_allclose(transformed_expected_output_np, M_data_np, rtol=1, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
