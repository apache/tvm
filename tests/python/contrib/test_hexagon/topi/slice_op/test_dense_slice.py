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
import tvm.topi.hexagon.qnn as qnn
import tvm.topi.hexagon.slice_ops as sl
from ...infrastructure import transform_numpy, quantize_np
from tvm.contrib.hexagon import allocate_hexagon_array


@tvm.testing.fixture
def input_np(input_shape, dtype):
    if "int" in dtype:
        data = np.random.random(input_shape).astype("float32")
    elif "float" in dtype:
        data = np.random.random(input_shape).astype(dtype)
    return data


@tvm.testing.fixture
def weight_np(weight_shape, dtype):
    if "int" in dtype:
        weight = np.random.random(weight_shape).astype("float32")
    elif "float" in dtype:
        weight = np.random.random(weight_shape).astype(dtype)
    return weight


@tvm.testing.fixture
def input_quant(input_np, dtype):
    if "float" in dtype:
        return None
    quant, scale, zp = quantize_np(input_np, dtype)
    return {"zero": zp, "scale": scale, "data": quant}


@tvm.testing.fixture
def weight_quant(weight_np, dtype):
    if "float" in dtype:
        return None
    quant, scale, zp = quantize_np(weight_np, "int8")
    return {"zero": zp, "scale": scale, "data": quant}


@tvm.testing.fixture
def bias_np(bias_shape, bias, dtype):
    if bias:
        if "int" in dtype:
            data = np.random.randint(-128, 127, size=bias_shape).astype("int32")
        elif "float" in dtype:
            data = np.random.random(bias_shape).astype(dtype)
        return data
    else:
        return None


@tvm.testing.fixture
def quant_arr(input_quant, weight_quant):
    if input_quant is None:
        return None
    arr = np.empty((6,), dtype="float32")
    arr[0] = input_quant["zero"]
    arr[1] = input_quant["scale"]
    arr[2] = weight_quant["zero"]
    arr[3] = weight_quant["scale"]
    return arr


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, layout):
    return transform_numpy(expected_output_np, "nc", layout)


@tvm.testing.fixture
def transformed_input_np(input_np, layout):
    return transform_numpy(input_np, "nc", layout)


@tvm.testing.fixture
def transformed_input_quant(input_quant, layout):
    if input_quant is None:
        return None
    input_quant["data"] = transform_numpy(input_quant["data"], "nc", layout)
    return input_quant


class TestDenseSlice:
    (input_shape, output_shape, layout, bias, dtype,) = tvm.testing.parameters(
        (  # Float 16
            [1, 1024],
            [1, 1024],
            "nc-1024c-2d",
            False,
            "float16",
        ),
        (
            [1, 2048],
            [1, 2048],
            "nc-1024c-2d",
            True,
            "float16",
        ),
        (  # Uint 8
            [1, 2048],
            [1, 2048],
            "nc-2048c-2d",
            False,
            "uint8",
        ),
        (
            [1, 4096],
            [1, 4096],
            "nc-2048c-2d",
            True,
            "uint8",
        ),
    )

    @tvm.testing.fixture
    def expected_output_np(self, input_np, weight_np, bias_np, bias):
        ref_np = tvm.topi.testing.dense(
            np.reshape(input_np, (input_np.shape[0], input_np.shape[-1])),
            weight_np.T,  # Function expects [in_dim, out_dim]
            bias_np,
            use_bias=bias,
            out_dtype="float32" if "int" in str(input_np.dtype) else input_np.dtype,
        )
        return ref_np

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
        layout,
        output_shape,
        input_shape,
        input_np,
        input_quant,
        transformed_input_np,
        transformed_input_quant,
        weight_np,
        # transformed_weight_np,
        weight_quant,
        # transformed_weight_quant,
        transformed_expected_output_np,
        expected_output_np,
        quant_arr,
        hexagon_session: Session,
    ):

        target_hexagon = tvm.target.hexagon("v69")
        A = te.placeholder(input_shape, name="A", dtype=dtype)
        W = te.placeholder(
            (output_shape[-1], input_shape[-1]),
            name="W",
            dtype="int8" if dtype == "uint8" else dtype,
        )
        args = [A, W]
        tensors = [A, W]

        # If quantized, append the quantization params
        if "int" in dtype:
            args.append(quant_arr[0].astype("int32"))
            args.append(quant_arr[1])
            args.append(quant_arr[2].astype("int32"))
            args.append(quant_arr[3])

        if bias_np is not None:
            B = te.placeholder((output_shape[-1],), name="B", dtype=str(bias_np.dtype))
            args.append(B)
            tensors.append(B)
        else:
            B = None

        # Different compute and schedule for quant and float
        if "float" in dtype:
            M = sl.dense_compute(*args)
            tir_schedule = sl.dense_schedule([M], tensors, layout, layout)
        elif "int" in dtype:
            M = qnn.qdense_compute(*args, bias=B)
            tir_schedule = qnn.qdense_schedule([M], tensors, layout, layout)
        else:
            print("Unsupported dtype {}".format(dtype))
            exit(-1)

        sch = tir_schedule.mod

        input_axis_separator = [2]
        output_axis_separator = [2]

        with tvm.transform.PassContext(opt_level=3):
            func = tvm.build(
                sch,
                args,
                target=tvm.target.Target(target_hexagon, host=target_hexagon),
                name="dense",
            )
            func.save("dense.s" if bias_np is None else "dense_bias.s")

        input_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=transformed_input_np if "float" in dtype else transformed_input_quant["data"],
            axis_separators=input_axis_separator,
            mem_scope="global.vtcm",
        )
        weight_arr = allocate_hexagon_array(
            hexagon_session.device,
            data=weight_np if "float" in dtype else weight_quant["data"],
            axis_separators=None,
            mem_scope="global",
        )
        output_arr = allocate_hexagon_array(
            hexagon_session.device,
            transformed_expected_output_np.shape,
            "float32" if "int" in dtype else dtype,
            axis_separators=output_axis_separator,
            mem_scope="global.vtcm",
        )
        arrs = [input_arr, weight_arr]

        if bias_np is not None:
            bias_arr = allocate_hexagon_array(
                hexagon_session.device,
                data=bias_np,
                axis_separators=None,
                mem_scope="global.vtcm",
            )
            arrs.append(bias_arr)

        arrs.append(output_arr)

        mod = hexagon_session.load_module(func)
        mod(*arrs)

        # Reshape for comparison
        b, c = output_shape
        if layout == "nc-1024c-2d":
            output_np = output_arr.numpy().reshape([b, c // 1024, 1024])
        elif layout == "nc-2048c-2d":
            output_np = output_arr.numpy().reshape([b, c // 2048, 2048])
        else:
            raise RuntimeError(f"Unexpected layout '{layout}'")

        if "int" in dtype:
            np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-2, atol=0)
        elif "float" in dtype:
            np.testing.assert_allclose(output_np, transformed_expected_output_np, rtol=1e-1, atol=0)


if __name__ == "__main__":
    tvm.testing.main()
