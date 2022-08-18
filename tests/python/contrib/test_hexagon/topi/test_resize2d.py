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
import sys
np.set_printoptions(threshold=sys.maxsize)


@tvm.testing.fixture
def expected_output_np(
    input_np,
    in_height,
    in_width,
    out_height,
    out_width,
    layout,
    method,
    coord_trans,
    dtype,
):
    scale_h = out_height / in_height
    scale_w = out_width / in_width

    return resize2d_python(input_np, (scale_h, scale_w), layout, method, coord_trans)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    a = np.zeros(input_shape, dtype)
    shape = input_shape
    b = shape[0]
    ch = shape[1]
    cw = shape[2]
    cc = shape[3]
    val = 0
    for i in range(b):
        for j in range(ch):
            for l in range(cw):
                for k in range(cc):
                    if j > 0:
                        val = l - j - 1
                    else:
                        val = l
                    a[i][j][l][k] = val
    return a


@tvm.testing.fixture
def transformed_input_np(input_np, layout, input_crouton_layout, dtype):
    if dtype == "float16" or dtype == "uint8" or dtype == "int8":
        return transform_numpy(input_np, layout.lower(), input_crouton_layout)
    else:
        raise RuntimeError(f"Unsupported data type '{dtype}'")


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, layout, output_layout, dtype):
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
            2,
            3,
            3,
            6,
            6,
        ),
    )

    (layout, input_crouton_layout, output_layout, dtype,) = tvm.testing.parameters(
        #("NHWC", "nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-2d", "float16"),
        ("NHWC", "nhwc", "nhwc", "uint8"),
    )

    coord_trans = tvm.testing.parameter("asymmetric")#, "align_corners", "half_pixel")
    method = tvm.testing.parameter("cubic")#, "linear", "cubic")

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
        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)

        M = s1.resize2d_compute(
            A,
            [0.0] * 4,
            (output_shape[1], output_shape[2]),
            layout=layout,
            coordinate_transformation_mode=coord_trans,
            method=method,
            out_dtype=dtype,
        )

        tir_schedule = s1.tir_resize2d_schedule(M, A, input_crouton_layout, output_layout)

        sch = tir_schedule.mod

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                [A, M],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="resize2d",
            )

        A_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            data=input_np,
            dtype=dtype,
        )

        M_data_nd = allocate_hexagon_array(
            hexagon_session.device,
            expected_output_np.shape,
            dtype=dtype,
        )

        mod = hexagon_session.load_module(func)
        mod(A_data_nd, M_data_nd)

        b, h, w, c = output_shape
        # convert nd to np and reshape to fixed chunk size layout
        
        M_data_np = M_data_nd.numpy()

        print("OUT GENERATED:\n", M_data_np)
        print("OUT EXPECTED:\n", expected_output_np)
        

        np.testing.assert_allclose(expected_output_np, M_data_np, rtol=1, atol=1)


if __name__ == "__main__":
    tvm.testing.main()