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

import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.contrib import graph_runtime

roundings = ["UPWARD", "TONEAREST"]


def verify(mod, goldens):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
        golden_data, golden_output = goldens
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input("quantized_data", golden_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        np.testing.assert_equal(res, golden_output)


def get_mod(
    data_shape,
    data_dtype,
    out_dtype,
    input_scale,
    output_scale,
    input_zero_point=0,
    output_zero_point=0,
    rounding="TONEAREST",
    axis=0,
):
    quantized_data = relay.var("quantized_data", shape=data_shape, dtype=data_dtype)
    if isinstance(input_scale, float):
        input_scale_expr = relay.const(input_scale, "float32")
    else:
        input_scale_expr = relay.const(np.array(input_scale).astype("float32"))

    if isinstance(input_zero_point, float):
        input_zero_point_expr = relay.const(input_zero_point, "int32")
    else:
        input_zero_point_expr = relay.const(np.array(input_zero_point).astype("int32"))

    mod = relay.qnn.op.requantize(
        quantized_data,
        input_scale=input_scale_expr,
        input_zero_point=input_zero_point_expr,
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
        axis=axis,
        rounding=rounding,
        out_dtype=out_dtype,
    )

    mod = relay.Function(relay.analysis.free_vars(mod), mod)
    mod = tvm.IRModule.from_expr(mod)
    return mod


def test_same_scale():
    # Have same scales, everything within range
    golden_data = np.arange(-100, 100, 1).astype("int32")
    golden_output = golden_data

    for rounding in roundings:
        mod = get_mod(
            data_shape=(200,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=0.5,
            output_scale=0.5,
            rounding=rounding,
        )
        assert "right_shift" not in mod.astext()
        verify(mod, (golden_data, golden_output))


def test_downscale():
    for rounding in roundings:
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=1,
            output_scale=16,
            rounding=rounding,
        )

        # Try positive values
        # 8 corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.repeat([0, 1, 2], [8, 16, 8])
        verify(mod, (golden_data, golden_output))

        # Try negative values
        # -8 corresponds to -0.5. For UPWARD, this is 0
        golden_data = np.arange(0, -32, -1).astype("int32")
        if rounding == "UPWARD":
            golden_output = np.repeat([0, -1, -2], [9, 16, 7])
        else:
            golden_output = np.repeat([0, -1, -2], [8, 16, 8])
        verify(mod, (golden_data, golden_output))

        # Try a different scale
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=1,
            output_scale=4,
            rounding=rounding,
        )

        # Try positive values
        # 2I corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 4, 4, 4, 4, 4, 4, 2])
        verify(mod, (golden_data, golden_output))

        # Try negative values
        # -8 corresponds to -0.5. For UPWARD, this is 0
        golden_data = np.arange(0, -32, -1).astype("int32")
        if rounding == "UPWARD":
            golden_output = np.repeat(
                [0, -1, -2, -3, -4, -5, -6, -7, -8], [3, 4, 4, 4, 4, 4, 4, 4, 1]
            )
        else:
            golden_output = np.repeat(
                [0, -1, -2, -3, -4, -5, -6, -7, -8], [2, 4, 4, 4, 4, 4, 4, 4, 2]
            )
        verify(mod, (golden_data, golden_output))

        # Try uint8 out_dtype
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="uint8",
            input_scale=1,
            output_scale=16,
            rounding=rounding,
        )

        # Try positive values
        # 8 corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.repeat([0, 1, 2], [8, 16, 8])
        verify(mod, (golden_data, golden_output))

        # Try uint8 in_dtyope and uint8 out_dtype
        mod = get_mod(
            data_shape=(32,),
            data_dtype="uint8",
            out_dtype="uint8",
            input_scale=1,
            output_scale=16,
            rounding=rounding,
        )

        # Try positive values
        # 8 corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.repeat([0, 1, 2], [8, 16, 8])
        verify(mod, (golden_data, golden_output))


def test_upscale():
    for rounding in roundings:
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=2,
            output_scale=1,
            rounding=rounding,
        )

        # Try positive values
        # 8 corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.multiply(2, golden_data)
        verify(mod, (golden_data, golden_output))

        # Try negative values
        # -8 corresponds to -0.5. For UPWARD, this is 0
        golden_data = np.arange(0, -32, -1).astype("int32")
        golden_output = np.multiply(2, golden_data)
        verify(mod, (golden_data, golden_output))


def test_saturation():
    for rounding in roundings:
        mod = get_mod(
            data_shape=(16,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=0.5,
            output_scale=0.5,
            rounding=rounding,
        )
        golden_data = np.arange(0, 16, 1).astype("int32")
        golden_data = np.add(120, golden_data)
        output = np.array(
            [120, 121, 122, 123, 124, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127]
        )
        golden_output = output
        verify(mod, (golden_data, golden_output))

        # Try negative numbers
        golden_data = np.arange(0, -16, -1).astype("int32")
        golden_data = np.add(-120, golden_data)
        output = np.array(
            [
                -120,
                -121,
                -122,
                -123,
                -124,
                -125,
                -126,
                -127,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
                -128,
            ]
        )
        golden_output = output
        verify(mod, (golden_data, golden_output))


def test_zero_point():
    # Output zero point
    for rounding in roundings:
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=1,
            output_scale=16,
            output_zero_point=1,
            rounding=rounding,
        )

        # Try positive values
        # 8 corresponds to 0.5, resulting in 1
        golden_data = np.arange(0, 32, 1).astype("int32")
        golden_output = np.repeat([0, 1, 2], [8, 16, 8])
        golden_output = np.add(1, golden_output)
        verify(mod, (golden_data, golden_output))

        # Try negative values
        # -8 corresponds to -0.5. For UPWARD, this is 0
        golden_data = np.arange(-32, -64, -1).astype("int32")
        if rounding == "UPWARD":
            golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
        else:
            golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
        golden_output = np.add(1, golden_output)
        verify(mod, (golden_data, golden_output))

    # Input zero point
    for rounding in roundings:
        mod = get_mod(
            data_shape=(32,),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=1,
            output_scale=16,
            input_zero_point=16,
            rounding=rounding,
        )

        # Try positive values
        golden_data = np.arange(32, 64, 1).astype("int32")
        golden_output = np.repeat([2, 3, 4], [8, 16, 8])
        golden_output = np.subtract(golden_output, 1)
        verify(mod, (golden_data, golden_output))

        # Try negative values
        golden_data = np.arange(-32, -64, -1).astype("int32")
        if rounding == "UPWARD":
            golden_output = np.repeat([-2, -3, -4], [9, 16, 7])
        else:
            golden_output = np.repeat([-2, -3, -4], [8, 16, 8])
        golden_output = np.subtract(golden_output, 1)
        verify(mod, (golden_data, golden_output))


def test_per_channel_same_scale():
    # Have same scales, everything within range
    golden_data = np.arange(-5, 5, 1).astype("int32").reshape((5, 2))
    golden_output = golden_data

    for rounding in roundings:
        mod = get_mod(
            data_shape=(5, 2),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=[0.5, 0.5],
            output_scale=0.5,
            axis=1,
            rounding=rounding,
        )
        verify(mod, (golden_data, golden_output))

    # Change axis
    golden_data = np.arange(-10, 10, 1).astype("int32").reshape((2, 2, 5))
    golden_output = golden_data

    for rounding in roundings:
        mod = get_mod(
            data_shape=(2, 2, 5),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=[0.5, 0.5],
            output_scale=0.5,
            axis=1,
            rounding=rounding,
        )
        verify(mod, (golden_data, golden_output))


def test_per_channel_different_scale():
    # Have same scales, everything within range
    golden_data = np.arange(-5, 5, 1).astype("int32").reshape((5, 2))
    golden_output = np.array([-5, -2, -3, -1, -1, 0, 1, 1, 3, 2]).reshape((5, 2))

    for rounding in roundings:
        mod = get_mod(
            data_shape=(5, 2),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=[0.5, 0.25],
            output_scale=0.5,
            axis=1,
            rounding=rounding,
        )
        verify(mod, (golden_data, golden_output))

    # Change axis
    golden_data = np.arange(-20, 20, 2).astype("int32").reshape((2, 2, 5))
    golden_output = np.array(
        [-20, -18, -16, -14, -12, -5, -4, -3, -2, -1, 0, 2, 4, 6, 8, 5, 6, 7, 8, 9]
    ).reshape((2, 2, 5))

    for rounding in roundings:
        mod = get_mod(
            data_shape=(2, 2, 5),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=[0.5, 0.25],
            output_scale=0.5,
            axis=1,
            rounding=rounding,
        )
        verify(mod, (golden_data, golden_output))

    # Have input scale > output scale
    golden_data = np.arange(-5, 5, 1).astype("int32").reshape((5, 2))
    golden_output = np.array([-10, -2, -6, -1, -2, 0, 2, 1, 6, 2]).reshape((5, 2))

    for rounding in roundings:
        mod = get_mod(
            data_shape=(5, 2),
            data_dtype="int32",
            out_dtype="int8",
            input_scale=[1.0, 0.25],
            output_scale=0.5,
            axis=1,
            rounding=rounding,
        )
        verify(mod, (golden_data, golden_output))


if __name__ == "__main__":
    test_same_scale()
    test_downscale()
    test_upscale()
    test_saturation()
    test_zero_point()
    test_per_channel_same_scale()
    test_per_channel_different_scale()
