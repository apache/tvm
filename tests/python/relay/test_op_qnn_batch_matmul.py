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
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.testing.temp_op_attr import TempOpAttr

# We use llvm target for testing functionality. `llvm` points to an older Intel
# generation machine, that legalizes to a simple lowering. Therefore, the
# legalization is overwritten such that it can be skipped and we use the
# QNNCanonicalizeOps lowering for the testing.
def legalize_qnn_batch_matmul(attrs, inputs, types):
    return None


def make_requantize_params(input_scale, output_scale, output_zero_point, out_dtype):
    config = {
        "input_scale": input_scale,
        "output_scale": output_scale,
        "output_zero_point": output_zero_point,
        "out_dtype": out_dtype,
    }
    return config


def make_configuration(
    quantized_x,
    quantized_y,
    dtype,
    x_shape,
    y_shape,
    x_zero_point,
    y_zero_point,
    x_scale,
    y_scale,
    output,
    out_dtype="int32",
    requantize=None,
):
    config = {
        "quantized_x": quantized_x,
        "quantized_y": quantized_y,
        "dtype": dtype,
        "x_shape": x_shape,
        "y_shape": y_shape,
        "x_zero_point": x_zero_point,
        "y_zero_point": y_zero_point,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "output": output,
        "out_dtype": out_dtype,
        "requantize": requantize,
    }
    return config


def make_int_configuration(
    xzero_point_zero=True,
    yzero_point_zero=True,
    requantize_output=False,
    per_channel=False,
    batch_size=1,
):
    x_shape, y_shape, output_shape = (batch_size, 4, 5), (batch_size, 3, 5), (batch_size, 4, 3)
    if xzero_point_zero == True:
        x_zero_point = 0
    else:
        x_zero_point = -123

    if yzero_point_zero == True:
        y_zero_point = 0
    else:
        y_zero_point = -123

    in_dtype = "int8"
    out_dtype = "int32" if not requantize_output else "int8"

    quantized_x_np = (
        np.array(
            [
                1,
                3,
                5,
                7,
                9,  # sum = 25
                11,
                13,
                15,
                -19,
                -21,  # sum = -1
                1,
                3,
                5,
                7,
                9,  # sum = 25
                11,
                13,
                -17,
                17,
                -21,
            ]
        )[  # sum = 3
            np.newaxis, np.newaxis, :
        ]
        .repeat(batch_size, axis=1)
        .astype(in_dtype)
        .reshape(x_shape)
    )
    quantized_y_np = (
        np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 1, 3, 5, 7, 9])[np.newaxis, np.newaxis, :]
        .repeat(batch_size, axis=1)
        .astype(in_dtype)
        .reshape(y_shape)
    )
    x_scale = 0.5
    y_scale = 0.5
    output_scale = 2.0

    if requantize_output:
        assert xzero_point_zero is True
        assert yzero_point_zero is True
        output = np.array([20, 51, 20, -26, -27, -26, 20, 51, 20, -14, -10, -14])
    elif xzero_point_zero is False and yzero_point_zero is False:
        output = np.array(
            [81960, 88360, 81960, 78400, 84540, 78400, 81960, 88360, 81960, 78984, 85164, 78984]
        )
    elif xzero_point_zero is True and yzero_point_zero is False:
        output = np.array([3240, 3490, 3240, -320, -330, -320, 3240, 3490, 3240, 264, 294, 264])
    elif xzero_point_zero is False and yzero_point_zero is True:
        output = np.array([3240, 9640, 3240, 2878, 9018, 2878, 3240, 9640, 3240, 2970, 9150, 2970])
    else:
        output = np.array([165, 415, 165, -197, -207, -197, 165, 415, 165, -105, -75, -105])

    requant_params = (
        make_requantize_params(x_scale * y_scale, output_scale, -1, "int8")
        if requantize_output
        else None
    )
    # Outputs are for batch size 1, make batch size n version
    output = (
        output[np.newaxis, np.newaxis, :]
        .repeat(batch_size, axis=1)
        .astype(out_dtype)
        .reshape(output_shape)
    )
    return make_configuration(
        quantized_x=quantized_x_np,
        quantized_y=quantized_y_np,
        dtype=in_dtype,
        x_shape=x_shape,
        y_shape=y_shape,
        x_zero_point=x_zero_point,
        y_zero_point=y_zero_point,
        x_scale=x_scale,
        y_scale=y_scale,
        output=output,
        requantize=requant_params,
    )


def qnn_batch_matmul_driver(test_configuration):
    in_dtype = test_configuration["dtype"]
    out_dtype = test_configuration["out_dtype"]
    quantized_x_name = "quantized_x"
    quantized_y_name = "quantized_y"
    expected_out_dtype = test_configuration["out_dtype"]
    quantized_x = relay.var(quantized_x_name, shape=test_configuration["x_shape"], dtype=in_dtype)
    quantized_y = relay.var(quantized_y_name, shape=test_configuration["y_shape"], dtype=in_dtype)
    mod = relay.qnn.batch_matmul(
        quantized_x,
        quantized_y,
        relay.const(test_configuration["x_zero_point"], "int32"),
        relay.const(test_configuration["y_zero_point"], "int32"),
        relay.const(test_configuration["x_scale"], "float32"),
        relay.const(test_configuration["y_scale"], "float32"),
    )
    if test_configuration["requantize"] is not None:
        requantize_config = test_configuration["requantize"]
        mod = relay.qnn.requantize(
            mod,
            input_scale=relay.const(requantize_config["input_scale"], "float32"),
            input_zero_point=relay.const(0, "int32"),
            output_scale=relay.const(requantize_config["output_scale"], "float32"),
            output_zero_point=relay.const(requantize_config["output_zero_point"], "int32"),
            out_dtype=requantize_config["out_dtype"],
        )
        expected_out_dtype = requantize_config["out_dtype"]

    mod = relay.Function(relay.analysis.free_vars(mod), mod)
    mod = tvm.IRModule.from_expr(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    with tvm.transform.PassContext(opt_level=2):
        graph, lib, params = relay.build(mod, "llvm", params=None)
        mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
        mod.set_input(quantized_x_name, test_configuration[quantized_x_name])
        mod.set_input(quantized_y_name, test_configuration[quantized_y_name])
        mod.set_input(**params)
        mod.run()
        res = mod.get_output(0).numpy()
        np.testing.assert_equal(res, test_configuration["output"])
        assert res.dtype == expected_out_dtype


def test_qnn_batch_matmul_xzp0_yzp0():
    with TempOpAttr("qnn.batch_matmul", "FTVMQnnLegalize", legalize_qnn_batch_matmul):
        for batch_size in [1, 4, 7]:
            int32_output_params = make_int_configuration(
                xzero_point_zero=True, yzero_point_zero=True, batch_size=batch_size
            )
            qnn_batch_matmul_driver(int32_output_params)


def test_qnn_batch_matmul_xzp0():
    with TempOpAttr("qnn.batch_matmul", "FTVMQnnLegalize", legalize_qnn_batch_matmul):
        for batch_size in [1, 4, 7]:
            int32_output_params = make_int_configuration(
                xzero_point_zero=True, yzero_point_zero=False, batch_size=batch_size
            )
            qnn_batch_matmul_driver(int32_output_params)


def test_qnn_batch_matmul_yzp0():
    with TempOpAttr("qnn.batch_matmul", "FTVMQnnLegalize", legalize_qnn_batch_matmul):

        for batch_size in [1, 4, 7]:
            int32_output_params = make_int_configuration(
                xzero_point_zero=False, yzero_point_zero=True, batch_size=batch_size
            )
            qnn_batch_matmul_driver(int32_output_params)


def test_qnn_batch_matmul():
    with TempOpAttr("qnn.batch_matmul", "FTVMQnnLegalize", legalize_qnn_batch_matmul):
        for batch_size in [1, 4, 7]:

            int32_output_params = make_int_configuration(
                xzero_point_zero=False, yzero_point_zero=False, batch_size=batch_size
            )
            qnn_batch_matmul_driver(int32_output_params)


def test_qnn_batch_matmul_with_requantized_output():
    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", legalize_qnn_batch_matmul):
        for batch_size in [1, 4, 7]:
            int8_requantized_output_params = make_int_configuration(
                requantize_output=True, batch_size=batch_size
            )
            qnn_batch_matmul_driver(int8_requantized_output_params)


if __name__ == "__main__":
    test_qnn_batch_matmul_xzp0_yzp0()
    test_qnn_batch_matmul_xzp0()
    test_qnn_batch_matmul_yzp0()
    test_qnn_batch_matmul()
    test_qnn_batch_matmul_with_requantized_output()
