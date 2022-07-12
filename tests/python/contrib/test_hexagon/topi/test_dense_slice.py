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

from tvm import te, topi

import tvm.testing
from tvm.topi import testing
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.session import Session
import tvm.topi.hexagon.slice_ops as sl
from ..infrastructure import allocate_hexagon_array, transform_numpy


input_layout = tvm.testing.parameter(
    "n11c-1024c-2d",
)


@tvm.testing.fixture
def input_np(input_shape, dtype):
    return np.random.random(input_shape).astype(dtype)

@tvm.testing.fixture
def weight_np(weight_shape, dtype):
    return np.random.random(weight_shape).astype(dtype)

@tvm.testing.fixture
def bias_np(bias_shape, bias, dtype):
    if bias:
        return np.random.random(bias_shape).astype(dtype)
    else:
        return None

@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, output_layout):
    return transform_numpy(expected_output_np, "nhwc", output_layout)

@tvm.testing.fixture
def transformed_input_np(input_np, input_layout):
    return transform_numpy(input_np, "nhwc", input_layout)


class TestDenseSlice:
    # NOTE: input_layout is always assumed to be "n11c-1024c-2d"
    (
        input_shape,
        output_shape,
        output_layout,
        bias,
        dtype,
    ) = tvm.testing.parameters(
        (
            [1, 1, 1, 1024],
            [1, 1, 1, 1024],
            "n11c-1024c-2d",
            False,
            "float16",
        ),
        (
            [1, 1, 1, 1024],
            [1, 1, 1, 1024],
            "n11c-1024c-2d",
            True,
            "float16",
        ),
        (
            [1, 1, 1, 2*1024],
            [1, 1, 1, 2*1024],
            "n11c-1024c-2d",
            False,
            "float16",
        ),
        (
            [1, 1, 1, 2*1024],
            [1, 1, 1, 2*1024],
            "n11c-1024c-2d",
            True,
            "float16",
        ),
    )

    @tvm.testing.fixture
    def expected_output_np(
        self,
        input_np,
        weight_np,
        bias_np,
        bias
    ):
        ref_np = tvm.topi.testing.dense(
            np.reshape(input_np, (input_np.shape[0], input_np.shape[-1])), # Only batch and channel
            weight_np,
            bias_np,
            use_bias=bias
        )
        return np.reshape(ref_np, (ref_np.shape[0], 1, 1, ref_np.shape[-1]))

    @tvm.testing.fixture
    def weight_shape(self, input_shape, output_shape):
        return (output_shape[-1], input_shape[-1])

    @tvm.testing.fixture
    def bias_shape(self, output_shape):
        return (output_shape[-1],)

    @tvm.testing.requires_hexagon
    def test_dense_slice(
        self,
        dtype,
        bias_np,
        input_layout,
        output_layout,
        output_shape,
        input_shape,
        input_np,
        transformed_input_np,
        weight_np,
        transformed_expected_output_np,
        expected_output_np,
        hexagon_session: Session,
    ):
        if hexagon_session._launcher._serial_number != "simulator":
            pytest.skip(msg="Due to https://github.com/apache/tvm/issues/11928")

        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        W = te.placeholder((output_shape[-1], input_shape[-1]), name="W", dtype=dtype)
        if bias_np is not None:
            B = te.placeholder((output_shape[-1],), name="B", dtype=dtype)
            args = [A,W,B]
        else:
            B = None
            args = [A,W]

        M = sl.dense_compute(*args)

        # tir schedule
        tir_schedule = sl.dense_schedule([M], args, output_layout, input_layout)
        sch = tir_schedule.mod

        input_axis_separator = [4]
        if output_layout == "n11c-1024c-2d":
            output_axis_separator = [4]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                target=tvm.target.Target(target_hexagon, host=target_hexagon),
                name="dense",
            )

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np,
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )
        weight_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=weight_np,
            axis_separators=[1],
            mem_scope="global",
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )
        if bias_np is not None:
            bias_arr = allocate_hexagon_array(
                hexagon_session.device,
                data=bias_np,
                axis_separators=None,
                mem_scope="global.vtcm",
            )
            arrs = (input_arr, weight_arr, bias_arr, output_arr)
        else:
            arrs = (input_arr, weight_arr, output_arr)

        mod = hexagon_session.load_module(func)
        mod(*arrs)
        b, h, w, c = output_shape
        if output_layout == "n11c-1024c-2d":
            output_np = output_arr.numpy().reshape([b, 1, 1, c // 1024, 1024])
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-1, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
