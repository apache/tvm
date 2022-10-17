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

"""CMSIS-NN integration tests: Conv2D"""
import itertools
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from tvm.testing.aot import (
    generate_ref_data,
    AOTTestModel,
    compile_models,
    compile_and_run,
    run_and_check,
)
from tvm.micro.testing.aot_test_utils import AOT_USMP_CORSTONE300_RUNNER
from .utils import (
    make_module,
    get_range_for_dtype_str,
    get_same_padding,
    get_conv2d_qnn_params,
    get_kernel_bias_dtype,
    make_qnn_relu,
    assert_partitioned_function,
    assert_no_external_function,
    create_test_runner,
    CheckForPadsWithinCompositeFunc,
)


def make_model(
    shape,
    kernel_shape,
    input_zero_point,
    input_scale,
    kernel_zero_point,
    kernel_scale,
    output_zero_point,
    output_scale,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    kernel_dtype,
    bias_dtype,
    out_channels,
    kernel_layout,
    enable_bias,
    relu_type,
    input_op=None,
):
    """Return a model and any parameters it may have"""
    if input_op:
        op = input_op
    else:
        op = relay.var("input", shape=shape, dtype=dtype)

    h_index = kernel_layout.index("H")
    w_index = kernel_layout.index("W")
    kernel_h = kernel_shape[h_index]
    kernel_w = kernel_shape[w_index]
    p = (0, 0, 0, 0)
    if padding == "SAME":
        p = get_same_padding((shape[1], shape[2]), (kernel_h, kernel_w), dilation, strides)

    rng = np.random.default_rng(12321)
    kernel = tvm.nd.array(
        rng.integers(
            np.iinfo(kernel_dtype).min,
            high=np.iinfo(kernel_dtype).max,
            size=kernel_shape,
            dtype=kernel_dtype,
        )
    )
    kernel_const = relay.const(kernel, kernel_dtype)
    conv2d_kernel_sc = kernel_scale[0] if out_channels == 1 else kernel_scale
    conv = relay.qnn.op.conv2d(
        op,
        kernel_const,
        input_zero_point=relay.const(input_zero_point, "int32"),
        kernel_zero_point=relay.const(kernel_zero_point, "int32"),
        input_scale=relay.const(input_scale, "float32"),
        kernel_scale=relay.const(conv2d_kernel_sc, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout=kernel_layout,
        dilation=dilation,
        strides=strides,
        groups=groups,
        channels=out_channels,
        padding=p,
        out_dtype=bias_dtype,
    )
    bias = tvm.nd.array(rng.integers(0, high=10, size=(out_channels,), dtype=bias_dtype))
    bias_const = relay.const(bias, bias_dtype)
    last_op = relay.nn.bias_add(conv, bias_const, axis=3) if enable_bias else conv
    requant_input_sc = [sc * input_scale for sc in kernel_scale]
    requant_input_sc = requant_input_sc[0] if out_channels == 1 else requant_input_sc
    last_op = relay.qnn.op.requantize(
        last_op,
        relay.const(requant_input_sc, "float32"),
        relay.const(0, "int32"),
        relay.const(output_scale, "float32"),
        relay.const(output_zero_point, "int32"),
        out_dtype=dtype,
    )
    last_op = make_qnn_relu(last_op, relu_type, output_scale, output_zero_point, dtype)
    params = {"w": kernel, "b": bias}
    return last_op, params


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale, out_channels",
    [(10, 0.0128, [0.11, 0.22], 2)],
)
def test_conv2d_number_primfunc_args(
    padding,
    enable_bias,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
):
    """Tests number of arguments in Conv2D primfunc"""
    interface_api = "c"
    use_unpacked_api = True

    ifm_shape = (1, 64, 100, 4)
    kernel_size = (3, 3)
    strides = (1, 1)
    dilation = (1, 1)
    dtype = "int8"
    groups = 1
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)
    relu_type = "RELU"

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # compile the model
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)

    compiled_models = compile_models(
        AOTTestModel(module=cmsisnn_mod, inputs=inputs, outputs=output_list, params=params),
        interface_api,
        use_unpacked_api,
    )

    # validate number of TIR primfunc args
    expected_num_params = 6 if enable_bias else 5
    cmsisnn_tir_mod = None
    for target, mod in compiled_models[0].executor_factory.lowered_ir_mods.items():
        if target.kind.name == "cmsis-nn":
            cmsisnn_tir_mod = mod

    cmsisnn_func = cmsisnn_tir_mod["tvmgen_default_cmsis_nn_main_0"]
    assert (
        len(cmsisnn_func.params) == expected_num_params
    ), "Generated unexpected number of function arguments."


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("dtype", ["int8", "int16"])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("relu_type", ["RELU"])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale, out_channels",
    [(10, 0.0128, [0.11, 0.22], 2), (-64, 1, [1, 0.0256, 1.37], 3)],
)
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
def test_conv2d_symmetric_padding(
    dtype,
    padding,
    enable_bias,
    relu_type,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
    compiler_cpu,
    cpu_flags,
):
    """Tests QNN Conv2D where the padding is symmetric on both sides of input"""
    interface_api = "c"
    use_unpacked_api = True

    ifm_shape = (1, 64, 100, 4)
    kernel_size = (3, 3)
    strides = (1, 1)
    dilation = (1, 1)
    groups = 1
    # input_zero_point is not handled by TFLM when int16
    input_zero_point = input_zero_point if dtype == "int8" else 0
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            output_tolerance=1,
        ),
        create_test_runner(compiler_cpu, cpu_flags),
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale, out_channels",
    [(10, 0.0128, [0.11, 0.22], 2), (-64, 1, [1, 0.0256, 1.37], 3)],
)
def test_conv2d_asymmetric_padding(
    padding,
    enable_bias,
    relu_type,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
):
    """Tests QNN Conv2D where the padding is asymmetric on different sides of input"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    dtype = "int8"
    ifm_shape = (1, 25, 25, 12)
    kernel_size = (5, 5)
    strides = (2, 2)
    dilation = (1, 1)
    groups = 1
    input_zero_point = input_zero_point if dtype == "int8" else 0
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)
    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
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


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("ifm_shape", [(1, 25, 25, 12), (1, 64, 100, 4)])
@pytest.mark.parametrize(
    "pad_width",
    [
        ((0, 0), (0, 1), (1, 2), (0, 0)),
        ((0, 0), (1, 1), (1, 1), (0, 0)),
        ((0, 0), (2, 2), (3, 4), (0, 0)),
    ],
)
def test_pad_conv2d_fusion_int8(
    ifm_shape,
    pad_width,
):
    """Tests QNN Conv2D where the padding is asymmetric on different sides of input"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    ifm_shape = (1, 25, 25, 12)
    kernel_size = (5, 5)
    strides = (2, 2)
    dilation = (1, 1)
    padding = "SAME"
    dtype = "int8"
    enable_bias = True
    relu_type = "NONE"
    input_zero_point = 10
    input_scale = 0.0128
    kernel_scale = [0.11, 0.22]
    out_channels = 2
    groups = 1
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)
    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
    )

    invar = relay.var("input", shape=ifm_shape, dtype=dtype)
    pad = relay.nn.pad(
        invar,
        pad_width=pad_width,  # ((), (top, bottom), (left, right), ())
        pad_value=input_zero_point,
        pad_mode="constant",
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
        input_op=pad,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod, False)

    # check pad is not present inside CMSIS-NN partitioned function
    cmsisnn_func = None
    for var in cmsisnn_mod.get_global_vars():
        if "cmsis_nn_main_0" in var.name_hint:
            cmsisnn_func = cmsisnn_mod[var]
            pad_verifier = CheckForPadsWithinCompositeFunc()
            pad_verifier.visit_function(cmsisnn_func)
            pad_verifier.assert_no_pads_within_func()

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
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


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize(
    "ifm_shape, pad_width, conv2d_padding",
    [
        [(1, 25, 25, 12), ((0, 0), (0, 2), (1, 2), (0, 0)), "SAME"],
        [(1, 64, 100, 4), ((0, 0), (1, 3), (1, 1), (0, 0)), "VALID"],
        [(1, 55, 55, 3), ((0, 0), (2, 1), (3, 5), (0, 0)), "SAME"],
    ],
)
def test_invalid_pad_conv2d_fusion_int8(
    ifm_shape,
    pad_width,
    conv2d_padding,
):
    """Tests QNN Conv2D where the padding is asymmetric on different sides of input"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    ifm_shape = (1, 25, 25, 12)
    kernel_size = (5, 5)
    strides = (2, 2)
    dilation = (1, 1)
    dtype = "int8"
    enable_bias = True
    relu_type = "NONE"
    input_zero_point = 10
    input_scale = 0.0128
    kernel_scale = [0.11, 0.22]
    out_channels = 2
    groups = 1
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
    )

    invar = relay.var("input", shape=ifm_shape, dtype=dtype)
    pad = relay.nn.pad(
        invar,
        pad_width=pad_width,  # ((), (top, bottom), (left, right), ())
        pad_value=input_zero_point,
        pad_mode="constant",
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        conv2d_padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
        input_op=pad,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # check pad is only present inside main function
    cmsisnn_func = None
    for var in cmsisnn_mod.get_global_vars():
        if "cmsis_nn_main_0" in var.name_hint:
            cmsisnn_func = cmsisnn_mod[var]
            pad_verifier = CheckForPadsWithinCompositeFunc()
            pad_verifier.visit_function(cmsisnn_func)
            pad_verifier.assert_no_pads_within_func()
        else:
            main_func = cmsisnn_mod[var]
            pad_verifier = CheckForPadsWithinCompositeFunc()
            pad_verifier.visit_function(main_func)
            pad_verifier.assert_pads_within_func()

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
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


# pylint: disable=import-outside-toplevel
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("strides, dilation", [((3, 2), (1, 1))])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("activation", ["NONE", "RELU"])
def test_conv2d_int8_tflite(ifm_shape, kernel_shape, strides, dilation, padding, activation):
    """Compares TVM output against TFLite output"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER
    dtype = "int8"

    from tvm.relay.testing.tflite import TFLiteModel

    tfl_model = TFLiteModel(dtype)
    conv2d_function = tfl_model.create_conv2d_single(
        kernel_shape, strides, padding, dilation, activation
    )
    tfl_model.create_tflite_model(conv2d_function, [ifm_shape])
    relay_mod, relay_params = tfl_model.convert_to_relay()

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(relay_mod, relay_params)

    # validate pattern matching
    assert_partitioned_function(relay_mod, cmsisnn_mod)

    # validate CMSIS-NN output against TFLite output
    input_map, output_map, output_tolerance = tfl_model.generate_reference_data()
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=input_map,
            outputs=output_map,
            params=relay_params,
            output_tolerance=output_tolerance,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("dtype", ["int8", "int16"])
@pytest.mark.parametrize("ifm_shape", [(1, 28, 28, 12), (1, 64, 100, 4)])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (1, 1))])
@pytest.mark.parametrize("relu_type", ["RELU"])
@pytest.mark.parametrize(
    "depth_multiplier, enable_bias",
    [(1, True), (3, True)],
)
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale, out_channels",
    [(10, 0.0128, [0.11, 0.22], 2), (-64, 1, [1, 0.0256, 1.37], 3)],
)
@pytest.mark.parametrize(
    "compiler_cpu, cpu_flags", [("cortex-m55", "+nomve"), ("cortex-m55", ""), ("cortex-m7", "")]
)
def test_depthwise(
    dtype,
    ifm_shape,
    kernel_size,
    padding,
    strides,
    dilation,
    enable_bias,
    relu_type,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
    depth_multiplier,
    compiler_cpu,
    cpu_flags,
):
    """Tests QNN Depthwise int8 op via CMSIS-NN"""
    interface_api = "c"
    use_unpacked_api = True

    groups = 1
    input_zero_point = input_zero_point if dtype == "int8" else 0
    kernel_layout = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)
    kernel_zero_point = 0
    in_min, in_max = get_range_for_dtype_str(dtype)

    groups = ifm_shape[3]
    kernel_layout = "HWOI"
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3], depth_multiplier)
    out_channels = ifm_shape[3] * depth_multiplier
    ks_len = len(kernel_scale)
    kernel_scale = [kernel_scale[i % ks_len] for i in range(out_channels)]

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
        is_depthwise=True,
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # validate the output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            output_tolerance=1,
        ),
        create_test_runner(compiler_cpu, cpu_flags),
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (1, 1))])
@pytest.mark.parametrize("relu_type", ["RELU", "NONE"])
@pytest.mark.parametrize("depth_multiplier", [1, 3])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale",
    [
        (
            10,
            0.0128,
            [0.11, 0.22],
        ),
        (
            -64,
            1,
            [1, 0.0256, 1.37],
        ),
    ],
)
def test_relay_conv2d_cmsisnn_depthwise_int8(
    padding,
    strides,
    dilation,
    relu_type,
    input_zero_point,
    input_scale,
    kernel_scale,
    depth_multiplier,
):
    """Tests QNN Depthwise int8 op via CMSIS-NN"""
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_USMP_CORSTONE300_RUNNER

    dtype = "int8"
    in_min, in_max = get_range_for_dtype_str(dtype)

    ifm_shape = (1, 24, 24, 1)
    groups = ifm_shape[3]
    input_zero_point = input_zero_point if dtype == "int8" else 0
    kernel_layout = "HWIO"
    (kernel_h, kernel_w) = (3, 3)
    kernel_shape = (kernel_h, kernel_w, ifm_shape[3], depth_multiplier)
    out_channels = ifm_shape[3] * depth_multiplier
    enable_bias = True
    ks_len = len(kernel_scale)
    kernel_zero_point = 0
    kernel_scale = [kernel_scale[i % ks_len] for i in range(out_channels)]

    kernel_dtype, bias_dtype = get_kernel_bias_dtype(dtype)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        input_dtype=dtype,
        kernel_dtype=kernel_dtype,
        output_dtype=dtype,
        is_depthwise=True,
    )

    model, params = make_model(
        ifm_shape,
        kernel_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        kernel_dtype,
        bias_dtype,
        out_channels,
        kernel_layout,
        enable_bias,
        relu_type,
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate pattern matching
    assert_partitioned_function(orig_mod, cmsisnn_mod)

    # generate reference output
    rng = np.random.default_rng(12345)
    inputs = {"input": rng.integers(in_min, high=in_max, size=ifm_shape, dtype=dtype)}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)

    # validate presence of depthwise convolution
    compiled_models = compile_models(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            output_tolerance=1,
        ),
        interface_api,
        use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    cmsisnn_tir_mod = None
    for target, mod in compiled_models[0].executor_factory.lowered_ir_mods.items():
        if target.kind.name == "cmsis-nn":
            cmsisnn_tir_mod = mod

    cmsisnn_func = cmsisnn_tir_mod["tvmgen_default_cmsis_nn_main_0"]
    call_extern = None
    # This happens when context buffer is init in case depthM != 1
    if isinstance(cmsisnn_func.body, tvm.tir.stmt.Evaluate):
        call_extern = cmsisnn_func.body.value
    else:
        call_extern = cmsisnn_func.body.body.value
    assert (
        call_extern.args[0].value == "arm_depthwise_conv_wrapper_s8"
    ), "Relay Conv2D should be mapped to CMSIS-NN Depthwise Convolution."

    # validate the output
    run_and_check(
        models=compiled_models,
        runner=test_runner,
        interface_api=interface_api,
    )


def parameterize_for_invalid_model(test):
    """Generates non-int8 non-int16 inputs"""
    in_dtype = ["uint8", "int8", "int16"]
    kernel_dtype = ["uint8", "int8"]
    kernel_zero_point = [-33, 10, 0]
    input_zero_point = [64, 0]
    all_combinations = itertools.product(
        in_dtype, kernel_dtype, kernel_zero_point, input_zero_point
    )
    all_combinations = filter(
        lambda parameters: not (
            (parameters[0] == "int8" or (parameters[0] == "int16" and parameters[3] == 0))
            and parameters[1] == "int8"
            and parameters[2] == 0
        ),
        all_combinations,
    )
    return pytest.mark.parametrize(
        ["in_dtype", "kernel_dtype", "kernel_zero_point", "input_zero_point"],
        all_combinations,
    )(test)


@tvm.testing.requires_cmsisnn
@parameterize_for_invalid_model
def test_invalid_parameters(
    in_dtype,
    kernel_dtype,
    kernel_zero_point,
    input_zero_point,
):
    """Tests Depthwise op for non int8 inputs"""
    ifm_shape = (1, 28, 28, 12)
    out_channels = 2
    input_scale = 1
    kernel_scale = [0.11, 0.0237]

    kernel_layout = "HWIO"
    kernel_shape = [3, 3, ifm_shape[3], out_channels]
    _, bias_dtype = get_kernel_bias_dtype(in_dtype)
    output_scale, output_zero_point = get_conv2d_qnn_params(
        kernel_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        in_dtype,
        kernel_dtype,
        in_dtype,
        is_depthwise=False,
    )
    model, params = make_model(
        shape=ifm_shape,
        kernel_shape=kernel_shape,
        input_zero_point=input_zero_point,
        input_scale=input_scale,
        kernel_zero_point=kernel_zero_point,
        kernel_scale=kernel_scale,
        output_zero_point=output_zero_point,
        output_scale=output_scale,
        padding="SAME",
        strides=(1, 1),
        dilation=(1, 1),
        groups=1,
        dtype=in_dtype,
        kernel_dtype=kernel_dtype,
        bias_dtype=bias_dtype,
        out_channels=out_channels,
        kernel_layout=kernel_layout,
        enable_bias=True,
        relu_type="NONE",
    )
    orig_mod = make_module(model)
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)
    assert_no_external_function(cmsisnn_mod)


if __name__ == "__main__":
    tvm.testing.main()
