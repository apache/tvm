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
from tvm.topi.testing import adaptive_pool
import tvm.topi.hexagon.qnn as s1
from ..infrastructure import allocate_hexagon_array, transform_numpy, quantize_np


@tvm.testing.fixture
def expected_output_np(
    input_np,
    output_size,
    pool_type,
    layout,
):
    out_width = output_size[0]

    ref_np = tvm.topi.testing.adaptive_pool(
        input_np,
        out_width,
        pool_type,
        layout,
    )
    return ref_np


@tvm.testing.fixture
def input_np(input_shape, dtype):
    if dtype == "uint8" or dtype == "int8":
        dtype = "float32"
    return np.random.random(input_shape).astype(dtype)


@tvm.testing.fixture
def quantize_input_np(input_np, dtype):
    if dtype == "uint8" or dtype == "int8":
        global zero_point_val, scale_val
        input_np_quantized, scale_val, zero_point_val = quantize_np(input_np, dtype)
        return input_np_quantized


@tvm.testing.fixture
def transformed_input_np(input_np, quantize_input_np, input_layout, layout, dtype):
    if dtype == "uint8" or dtype == "int8":
        return transform_numpy(quantize_input_np, layout.lower(), input_layout)

    raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def quantize_expected_output_np(expected_output_np, output_layout, layout, dtype):
    if dtype == "uint8" or dtype == "int8":
        global zero_point_M_val, scale_M_val
        out_ref_quantized, scale_M_val, zero_point_M_val = quantize_np(expected_output_np, dtype)

        # Since output_layout is ncw, no transformation is needed.
        return out_ref_quantized

    raise RuntimeError(f"Unsupported data type '{dtype}'")


# Fixed chunk layout is set as ncw-32c64w-2d for now.
# For optimization, it might get changed later.
input_layout, pool_type, layout, output_size, dtype, = tvm.testing.parameters(
    (
        "ncw-32c64w-2d",
        "avg",
        "NCW",
        [1],
        "uint8",
    )
)


@tvm.testing.fixture
def output_layout(output_size):
    # The adaptive_avg_pool1d implementation only handles specialized case
    # where output_size is 1 as it appears on quantized distilbert model.
    # Since output size won't be a multiple of fixed-chunk,
    # output_layout is ncw.
    return "ncw"


class TestAdaptivePool1D:
    (input_shape,) = tvm.testing.parameters(
        ([1, 128, 128],),
        ([1, 64, 64],),
        ([1, 64, 128],),
        ([1, 32, 64],),
        ([1, 128, 768],),
    )

    @tvm.testing.requires_hexagon
    def test_pool1d(
        self,
        dtype,
        output_size,
        input_layout,
        output_layout,
        input_shape,
        layout,
        input_np,
        transformed_input_np,
        quantize_expected_output_np,
        hexagon_session,
    ):
        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)

        out_width = [output_size[0]]

        n, c = input_shape[:2]
        oshape = (n, c) + (out_width,)

        M = s1.adaptive_avg_pool1d(
            A,
            output_size,
            dtype,
            zero_point_val,
            scale_val,
            zero_point_M_val,
            scale_M_val,
        )

        tir_schedule = s1.tir_adaptive_avg_pool1d_schedule(M, A, output_layout, input_layout)

        sch = tir_schedule.mod

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [A, M],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="adaptive_pool1d",
            )

        input_axis_separator = [3]

        A_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            dtype=dtype,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )

        M_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            quantize_expected_output_np.shape,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(A_data_nd, M_data_nd)

        # Convert nd to np
        M_data_np = M_data_nd.numpy()

        np.testing.assert_allclose(quantize_expected_output_np, M_data_np, atol=2)


if __name__ == "__main__":
    tvm.testing.main()
