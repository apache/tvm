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

from tvm import te, topi, tir

import tvm.testing
from tvm.topi import testing
from tvm.contrib.hexagon.build import HexagonLauncher
from tvm.contrib.hexagon.session import Session
import tvm.topi.hexagon.slice_ops as sl
from ..infrastructure import allocate_hexagon_array, transform_numpy, quantize


input_layout = tvm.testing.parameter(
    "nc-1024c-2d",
)


@tvm.testing.fixture
def input_quant(input_shape, dtype):
    return quantize(np.random.random(input_shape), dtype)


@tvm.testing.fixture
def weight_quant(weight_shape, dtype):
    return quantize(np.random.random(weight_shape), dtype)


@tvm.testing.fixture
def bias_np(bias_shape, bias, dtype):
    if bias:
        return np.random.random(bias_shape).astype(dtype)
    else:
        return None

@tvm.testing.fixture
def quant_arr(input_quant, weight_quant, transformed_expected_output_quant):
    arr = np.empty((6,), dtype="float32")
    arr[0] = input_quant["zero"]
    arr[1] = input_quant["scale"]
    arr[2] = weight_quant["zero"]
    arr[3] = weight_quant["scale"]
    arr[4] = transformed_expected_output_quant["zero"]
    arr[5] = transformed_expected_output_quant["scale"]
    return arr

@tvm.testing.fixture
def transformed_expected_output_quant(expected_output_quant, output_layout):
    expected_output_quant["data"] = transform_numpy(expected_output_quant["data"], "nc", output_layout).astype(expected_output_quant["data"].dtype)
    return expected_output_quant


@tvm.testing.fixture
def transformed_input_quant(input_quant, input_layout):
    input_quant["data"] = transform_numpy(input_quant["data"], "nc", input_layout)
    return input_quant


class TestDenseSlice:
    # NOTE: input_layout is always assumed to be "n11c-1024c-2d"
    (input_shape, output_shape, output_layout, bias, dtype,) = tvm.testing.parameters(
        (
            [1, 1024],
            [1, 1024],
            "nc-1024c-2d",
            False,
            "uint8",
        ),
        (
            [1, 1024],
            [1, 1024],
            "nc-1024c-2d",
            True,
            "uint8",
        ),
    )

    @tvm.testing.fixture
    def expected_output_quant(self, input_quant, weight_quant, bias_np, bias):

        # dequantize
        data = (input_quant["data"]-input_quant["zero"])*input_quant["scale"]
        weight = (weight_quant["data"]-weight_quant["zero"])*weight_quant["scale"]

        ref_np = tvm.topi.testing.dense(
            np.reshape(data, (data.shape[0],data.shape[-1])),  # Only batch and channel
            weight,
            bias_np,
            use_bias=bias,
        )

        # quantize
        ref_quant = quantize(ref_np, data.dtype)

        return ref_quant

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
        input_quant,
        transformed_input_quant,
        weight_quant,
        transformed_expected_output_quant,
        expected_output_quant,
        quant_arr,
        hexagon_session: Session,
    ):
        if hexagon_session._launcher._serial_number != "simulator":
            pytest.skip(msg="Due to https://github.com/apache/tvm/issues/11928")

        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        W = te.placeholder((output_shape[-1], input_shape[-1]), name="W", dtype=dtype)
        quant = te.placeholder(quant_arr.shape, name="quant_params", dtype="float32")
        if bias_np is not None:
            B = te.placeholder((output_shape[-1],), name="B", dtype=dtype)
            args = [A, W, quant, B]
        else:
            B = None
            args = [A, W, quant]

        M = sl.qdense_compute(*args)

        # tir schedule
        tir_schedule = sl.qdense_schedule([M], args, output_layout, input_layout)
        sch = tir_schedule.mod


        input_axis_separator = [2]
        if output_layout == "nc-1024c-2d":
            output_axis_separator = [2]
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                args=args,
                target=tvm.target.Target(target_hexagon, host=target_hexagon),
                name="dense",
            )

        quant_arr_hex = allocate_hexagon_array(
            hexagon_session.device,
            data=quant_arr,
            dtype="float32",
            axis_separators=None,
            mem_scope="global",
        )
        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_quant["data"],
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )
        weight_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=weight_quant["data"],
            axis_separators=[1],
            mem_scope="global",
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_quant["data"].shape,
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
            arrs = (input_arr, weight_arr, quant_arr_hex,
                bias_arr, output_arr)
        else:
            arrs = (input_arr, weight_arr, quant_arr_hex,
                output_arr)

        mod = hexagon_session.load_module(func)
        mod(*arrs)
        b, c = output_shape
        if output_layout == "nc-1024c-2d":
            output_np = output_arr.numpy().reshape([b, c // 1024, 1024])
        else:
            raise RuntimeError(f"Unexpected layout '{output_layout}'")

        np.testing.assert_allclose(output_np, transformed_expected_output_quant["data"], rtol=0, atol=2)


if __name__ == "__main__":
    tvm.testing.main()
