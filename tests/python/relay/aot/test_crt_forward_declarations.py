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

"""test forward function declarations codegen by CodegenCHost."""

from collections import OrderedDict
import pytest
import numpy as np

import tvm.testing
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib import cmsisnn
from tvm.testing.aot import AOTTestModel, compile_models, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
    AOT_USMP_CORSTONE300_RUNNER,
    parametrize_aot_options,
    AOTTestRunner,
)


def _change_ndarray_layout(arr, src_layout, dst_layout):
    """Makes a copy of an ndarray, reshaping it to a new data layout.

    Parameter
    ---------
    arr : numpy.ndarray
        The ndarray to be reformatted.

    src_layout : str
        The current layout of the Relay constant. Must be alphabetic (e.g. NHWC
        or OIHW, but not NCHW2c).

    dst_layout : str
        The desired layout of new the Relay constant. Must be alphabetic (e.g. NHWC
        or OIHW, but not NCHW2c).

    Returns
    -------
    dst_shape : numpy.ndarray
        A copy of the ndarray with the new layout.
    """
    assert src_layout.isalpha() and dst_layout.isalpha()
    axis_order = [src_layout.index(c) for c in dst_layout]
    return np.transpose(arr, axis_order)


@tvm.testing.requires_package("tflite")
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("test_runner", [AOT_CORSTONE300_RUNNER, AOT_USMP_CORSTONE300_RUNNER])
def test_external_calls(test_runner):
    """Download a small network and partition for CMSIS-NN to test forward declarations for external
    calls outside of __tvm_main__."""
    # download the model
    base_url = (
        "https://github.com/ARM-software/ML-zoo/raw/"
        "48a22ee22325d15d2371a6df24eb7d67e21dcc97"
        "/models/keyword_spotting/cnn_small/tflite_int8"
    )
    file_to_download = "cnn_s_quantized.tflite"
    file_saved = "cnn_s_quantized_15Dec2021.tflite"
    model_file = download_testdata("{}/{}".format(base_url, file_to_download), file_saved)

    # convert the tflite network into relay model
    # pylint: disable=import-outside-toplevel
    from tvm.relay.testing.tflite import TFLiteModel

    input_shape = (1, 490)
    dtype = "int8"
    tfl_model = TFLiteModel(dtype)
    tfl_model.load_from_file(model_file, [input_shape])
    relay_mod, relay_params = tfl_model.convert_to_relay()
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(relay_mod, relay_params)

    # obtain the executor factory post relay compilation.
    input_map, output_map, output_tolerance = tfl_model.generate_reference_data()
    interface_api = "c"
    use_unpacked_api = True
    compiled_models = compile_models(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=input_map,
            outputs=output_map,
            params=None,
            output_tolerance=output_tolerance,
        ),
        interface_api,
        use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    # Validate frquency of function appearances in the Host C file after forward declarations.
    lib_mod = compiled_models[0].executor_factory.lib.imported_modules[0]
    main_source = lib_mod.get_source()
    assert (
        main_source.count("TVMBackendAllocWorkspace") == 3
        or main_source.count("TVMBackendAllocWorkspace") == 0
    )
    assert main_source.count("tvmgen_default_fused_reshape") == 3
    assert main_source.count("tvmgen_default_cmsis_nn_main") == 12
    cmsisnn_source = lib_mod.imported_modules[0].get_source()
    assert cmsisnn_source.count("arm_convolve_wrapper") == 1
    assert cmsisnn_source.count("arm_fully_connected") == 3
    assert cmsisnn_source.count("arm_softmax") == 1


@parametrize_aot_options
def test_internal_calls(interface_api, use_unpacked_api, test_runner):
    """Test for all internal function calls. No forward declarations are expected here."""
    dtype = "float32"
    groups = 32
    weight_shape = 1
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    test_runner = AOTTestRunner(
        makefile=test_runner.makefile,
        prologue=test_runner.prologue,
        epilogue=test_runner.epilogue,
        includes=test_runner.includes,
        parameters=test_runner.parameters,
        pass_config=pass_config,
    )

    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = tvm.relay.transform.InferType()(mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)

    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])

    output_list = generate_ref_data(mod, inputs)
    compiled_models = compile_models(
        models=AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        interface_api=interface_api,
        use_unpacked_api=use_unpacked_api,
        pass_config=test_runner.pass_config,
    )

    lib_mod = compiled_models[0].executor_factory.lib.imported_modules[0]
    main_source = lib_mod.get_source()
    assert main_source.count("int32_t tvmgen_default_fused_nn_contrib_depthwise_conv2d_NCHWc") == 2
    assert main_source.count("int32_t tvmgen_default_fused_layout_transform") == 6


@tvm.testing.requires_corstone300
def test_tensorized_calls():
    """Test a subgraph with a mix of internal and tensorized calls."""
    data_shape, kernel_size, num_filter, groups, strides, padding, dilation = (
        (1, 32, 32, 16),
        (3, 3),
        16,
        1,
        1,
        (0, 2, 2, 0),
        1,
    )
    in_dtype = "int8"
    data_layout = "NHWC"
    kernel_layout = "HWOI"
    ref_kernel_layout = "HWIO"
    out_layout = "NHWC"
    schedule_name = "conv2d_nhwc_dsp.arm_cpu"

    ref_input_data = np.random.randint(low=-128, high=127, size=data_shape, dtype=in_dtype)
    ref_input_var = relay.var("input", relay.TensorType(data_shape, in_dtype))  # NHWC layout
    kernel_shape = (*kernel_size, data_shape[-1] // groups, num_filter)  # HWIO layout
    ref_kernel_data = np.random.randint(low=-10, high=10, size=kernel_shape, dtype=in_dtype)

    ref_relay_op = relay.op.nn.conv2d(
        ref_input_var,
        relay.const(_change_ndarray_layout(ref_kernel_data, "HWIO", ref_kernel_layout)),
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        dilation=(dilation, dilation),
        data_layout="NHWC",
        kernel_layout=ref_kernel_layout,
        out_dtype="int32",
        out_layout="NHWC",
    )
    ref_module = tvm.IRModule.from_expr(relay.Function([ref_input_var], ref_relay_op))
    ref_outputs = generate_ref_data(ref_module, {"input": ref_input_data})

    # Reshape output dictionary to match out_layout
    assert len(ref_outputs) == 1
    output_tensor_name, output_tensor = next(iter(ref_outputs.items()))
    ref_outputs[output_tensor_name] = _change_ndarray_layout(output_tensor, "NHWC", out_layout)

    test_input_data = _change_ndarray_layout(ref_input_data, "NHWC", data_layout)
    test_input_var = relay.var("input", relay.TensorType(test_input_data.shape, in_dtype))
    test_kernel_data = _change_ndarray_layout(ref_kernel_data, "HWIO", kernel_layout)

    test_relay_op = relay.op.nn.conv2d(
        test_input_var,
        relay.const(test_kernel_data),
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        groups=groups,
        dilation=(dilation, dilation),
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_dtype="int32",
        out_layout=out_layout,
    )
    test_function = relay.Function([test_input_var], test_relay_op)
    test_model = AOTTestModel(
        module=tvm.IRModule.from_expr(test_function),
        inputs={"input": test_input_data},
        outputs=ref_outputs,
    )
    compiled_models = compile_models(
        test_model,
        interface_api="c",
        use_unpacked_api=True,
        pass_config=AOT_CORSTONE300_RUNNER.pass_config,
        target="c -keys=arm_cpu -mcpu=cortex-m7",
        schedule_name=schedule_name,
    )

    lib_mod = compiled_models[0].executor_factory.lib.imported_modules[0]
    main_source = lib_mod.get_source()
    assert main_source.count("tvmgen_default_fused_nn_conv2d") == 3
    assert main_source.count("gemm_") == 15


if __name__ == "__main__":
    tvm.testing.main()
