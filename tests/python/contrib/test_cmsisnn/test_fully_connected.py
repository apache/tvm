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

"""CMSIS-NN integration tests: Fully Connected"""
import itertools
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn


from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    AOT_USMP_CORSTONE300_RUNNER,
    AOT_DEFAULT_RUNNER,
    generate_ref_data,
    compile_and_run,
)
from utils import (
    skip_if_no_reference_system,
    make_module,
    get_range_for_dtype_str,
    get_same_padding,
    get_conv2d_qnn_params,
    make_qnn_relu,
    assert_partitioned_function,
    assert_no_external_function,
)


def make_model(
    in_shape,  # [batchsize, in_channels]
    kernel_shape,  # [out_channels, num_inputs]
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    output_zero_point,
    output_scale,
    dtype,
    kernel_dtype,
    out_channels,
    enable_bias,
    relu_type="NONE",
):
    """Return a model and any parameters it may have"""
    a = relay.var("input", shape=in_shape, dtype=dtype)
    rng = np.random.default_rng(12321)
    w = tvm.nd.array(
        rng.integers(
            np.iinfo(kernel_dtype).min,
            high=np.iinfo(kernel_dtype).max,
            size=kernel_shape,
            dtype=kernel_dtype,
        )
    )
    weight_const = relay.const(w, kernel_dtype)
    fc = relay.qnn.op.dense(
        a,
        weight_const,
        input_zero_point=relay.const(input_zero_point, "int32"),
        kernel_zero_point=relay.const(kernel_zero_point, "int32"),
        input_scale=relay.const(input_scale, "float32"),
        kernel_scale=relay.const(kernel_scale, "float32"),
        units=out_channels,
        out_dtype="int32",
    )

    b = tvm.nd.array(rng.integers(0, high=10, size=(out_channels,), dtype="int32"))
    bias_const = relay.const(b, "int32")
    last_op = relay.nn.bias_add(fc, bias_const) if enable_bias else fc
    requant_input_sc = input_scale * kernel_scale
    last_op = relay.qnn.op.requantize(
        last_op,
        relay.const(requant_input_sc, "float32"),
        relay.const(0, "int32"),
        relay.const(output_scale, "float32"),
        relay.const(output_zero_point, "int32"),
        out_dtype=dtype,
    )
    last_op = make_qnn_relu(last_op, relu_type, output_scale, output_zero_point, dtype)
    params = {"w": w, "b": b}
    return last_op, params


@tvm.testing.requires_cmsisnn
@pytest.mark.xfail(strict=False, reason="Flaky test: https://github.com/apache/tvm/issues/10213")
@pytest.mark.parametrize("in_shape", [(2, 28), (1, 64)])
@pytest.mark.parametrize("out_channels", [12, 128])
@pytest.mark.parametrize("enable_bias", [False, True])
@pytest.mark.parametrize("relu_type", ["RELU"])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale",
    [(10, 0.0128, 0.11), (-64, 0.0256, 1.37)],
)
def test_op_int8(
    in_shape,
    enable_bias,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
    relu_type,
):
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    dtype = "int8"
    kernel_zero_point = 0
    kernel_shape = [out_channels, in_shape[1]]
    conv2d_kernel_shape = (1, 1, kernel_shape[0], kernel_shape[1])
    in_min, in_max = get_range_for_dtype_str(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        conv2d_kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        dtype,
    )

    model, params = make_model(
        in_shape,
        kernel_shape,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        output_zero_point,
        output_scale,
        dtype,
        dtype,
        out_channels,
        enable_bias,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=in_shape, dtype=dtype)}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            output_tolerance=1,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


def parameterize_for_invalid_model(test):
    in_dtype = ["uint8", "int8"]
    kernel_dtype = ["uint8", "int8"]
    kernel_zero_point = [-33, 10, 0]
    all_combinations = itertools.product(in_dtype, kernel_dtype, kernel_zero_point)
    all_combinations = filter(
        lambda parameters: not (
            parameters[0] == "int8" and parameters[1] == "int8" and parameters[2] == 0
        ),
        all_combinations,
    )
    return pytest.mark.parametrize(
        ["in_dtype", "kernel_dtype", "kernel_zero_point"],
        all_combinations,
    )(test)


@tvm.testing.requires_cmsisnn
@parameterize_for_invalid_model
def test_invalid_parameters(
    in_dtype,
    kernel_dtype,
    kernel_zero_point,
):
    in_shape = (2, 28)
    out_channels = 2
    input_scale = 1
    input_zero_point = 24
    kernel_scale = [0.11, 0.0237]
    in_min, in_max = get_range_for_dtype_str(in_dtype)

    kernel_shape = [out_channels, in_shape[1]]
    conv2d_kernel_shape = [1, 1, kernel_shape[0], kernel_shape[1]]
    output_scale, output_zero_point = get_conv2d_qnn_params(
        conv2d_kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        in_dtype,
        kernel_dtype,
        in_dtype,
    )
    model, params = make_model(
        in_shape=in_shape,
        kernel_shape=kernel_shape,
        input_zero_point=input_zero_point,
        kernel_zero_point=kernel_zero_point,
        input_scale=input_scale,
        kernel_scale=kernel_scale,
        output_zero_point=output_zero_point,
        output_scale=output_scale,
        dtype=in_dtype,
        kernel_dtype=kernel_dtype,
        out_channels=out_channels,
        enable_bias=True,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_no_external_function(cmsisnn_mod)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
