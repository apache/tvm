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
from tvm.contrib import graph_executor
from tvm.relay.testing.temp_op_attr import TempOpAttr


# We use llvm target for testing functionality. `llvm` points to an older Intel
# generation machine, that legalizes to a simple lowering. Therefore, the
# legalization is overwritten such that it can be skipped and we use the
# QNNCanonicalizeOps lowering for the testing.
def legalize_qnn_dense(attrs, inputs, types):
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
    quantized_data,
    quantized_kernel,
    dtype,
    input_shape,
    kernel_shape,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    units,
    output,
    out_dtype="int32",
    bias=None,
    requantize=None,
):
    if requantize is not None:
        assert bias is not None
    config = {
        "quantized_data": quantized_data,
        "quantized_kernel": quantized_kernel,
        "dtype": dtype,
        "input_shape": input_shape,
        "kernel_shape": kernel_shape,
        "input_zero_point": input_zero_point,
        "kernel_zero_point": kernel_zero_point,
        "input_scale": input_scale,
        "kernel_scale": kernel_scale,
        "units": units,
        "output": output,
        "out_dtype": out_dtype,
        "bias": bias,
        "requantize": requantize,
    }
    return config


def make_int_configuration(use_bias=False, requantize_output=False, per_channel=False):
    input_shape, kernel_shape, output_shape = (2, 10), (3, 10), (2, 3)
    input_zero_point, kernel_zero_point = -1, -1
    in_dtype = "int8"
    out_dtype = "int32" if not requantize_output else "int8"
    units = 3
    quantized_data_np = (
        np.array([1, 3, 5, 7, 9, 11, 13, 15, -19, -21, 1, 3, 5, 7, 9, 11, 13, -17, 17, -21])
        .astype(in_dtype)
        .reshape(input_shape)
    )
    quantized_kernel_np = (
        np.array(
            [
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
            ]
        )
        .astype(in_dtype)
        .reshape(kernel_shape)
    )
    input_scale = 0.5
    kernel_scale = 0.5
    output_scale = 1.0
    bias = np.array([4, 8, 12]).astype(out_dtype).reshape((units,)) if use_bias else None

    if per_channel:
        assert use_bias and requantize_output
        kernel_scale = np.array([0.5, 0.3, 0.4], dtype=np.float32)
        output = np.array([23, 14, 20, 57, 34, 47])
    elif requantize_output:
        assert use_bias
        output = np.array([23, 24, 25, 57, 58, 59])
    elif use_bias:
        output = np.array([96, 100, 104, 232, 236, 240])
    else:
        output = np.array([92, 92, 92, 228, 228, 228])

    requant_params = (
        make_requantize_params(input_scale * kernel_scale, output_scale, -1, "int8")
        if requantize_output
        else None
    )

    output = output.astype(out_dtype).reshape(output_shape)
    return make_configuration(
        quantized_data=quantized_data_np,
        quantized_kernel=quantized_kernel_np,
        dtype=in_dtype,
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        input_zero_point=input_zero_point,
        kernel_zero_point=kernel_zero_point,
        input_scale=input_scale,
        kernel_scale=kernel_scale,
        units=units,
        output=output,
        bias=bias,
        requantize=requant_params,
    )


def qnn_dense_driver(test_configuration):
    in_dtype = test_configuration["dtype"]
    out_dtype = test_configuration["out_dtype"]
    quantized_data_name = "quantized_data"
    quantized_kernel_name = "quantized_kernel"
    expected_out_dtype = test_configuration["out_dtype"]
    bias_name = "bias"
    quantized_data = relay.var(
        quantized_data_name, shape=test_configuration["input_shape"], dtype=in_dtype
    )
    quantized_kernel = relay.var(
        quantized_kernel_name, shape=test_configuration["kernel_shape"], dtype=in_dtype
    )
    mod = relay.qnn.dense(
        quantized_data,
        quantized_kernel,
        relay.const(test_configuration["input_zero_point"], "int32"),
        relay.const(test_configuration["kernel_zero_point"], "int32"),
        relay.const(test_configuration["input_scale"], "float32"),
        relay.const(test_configuration["kernel_scale"], "float32"),
        test_configuration["units"],
    )
    if test_configuration[bias_name] is not None:
        bias = relay.var(bias_name, shape=test_configuration["bias"].shape, dtype=out_dtype)
        mod = relay.nn.bias_add(mod, bias)
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
        mod.set_input(quantized_data_name, test_configuration[quantized_data_name])
        mod.set_input(quantized_kernel_name, test_configuration[quantized_kernel_name])
        if test_configuration[bias_name] is not None:
            mod.set_input(bias_name, test_configuration[bias_name])
        mod.set_input(**params)
        mod.run()
        res = mod.get_output(0).numpy()
        np.testing.assert_equal(res, test_configuration["output"])
        assert res.dtype == expected_out_dtype


def test_qnn_dense_without_bias():
    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", legalize_qnn_dense):

        int32_output_without_bias_params = make_int_configuration(use_bias=False)
        qnn_dense_driver(int32_output_without_bias_params)


def test_qnn_dense_with_bias():
    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", legalize_qnn_dense):

        int32_output_with_bias_params = make_int_configuration(use_bias=True)
        qnn_dense_driver(int32_output_with_bias_params)


def test_qnn_dense_with_requantized_output():
    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", legalize_qnn_dense):

        int8_requantized_output_with_bias_params = make_int_configuration(
            use_bias=True, requantize_output=True
        )
        qnn_dense_driver(int8_requantized_output_with_bias_params)


def test_per_channel_weight_scale():
    with TempOpAttr("qnn.dense", "FTVMQnnLegalize", legalize_qnn_dense):
        config = make_int_configuration(use_bias=True, requantize_output=True, per_channel=True)
        qnn_dense_driver(config)


if __name__ == "__main__":
    test_qnn_dense_without_bias()
    test_qnn_dense_with_bias()
    test_qnn_dense_with_requantized_output()
    test_per_channel_weight_scale()
