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
"""AOT with C Device API Tests"""

import re
from collections import OrderedDict

import numpy as np
import pytest
import tvm.testing
from tvm import relay
from tvm.ir.module import IRModule
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
from tvm.testing.aot import AOTTestModel, compile_models, generate_ref_data


@pytest.fixture(name="device_api_main_func")
def fixture_device_api_main_func():
    """Test function generator which generates C Device API calls"""

    # Ideally we should have a sample Target registered here
    # but we're going to re-use this for now
    pytest.importorskip("ethosu.vela")

    # pylint: disable=import-outside-toplevel
    import tensorflow as tf
    import tflite.Model
    from tvm.relay.op.contrib.ethosu import partition_for_ethosu

    from tests.python.contrib.test_ethosu.infra import (
        create_test_runner,
        generate_ref_data_tflite,
    )

    # pylint: enable=import-outside-toplevel

    tf.config.run_functions_eagerly(True)

    class Model(tf.Module):
        @tf.function
        def tf_function(self, x):
            return tf.nn.max_pool(x, [1, 2], [1, 2], "SAME")

    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 3, 4, 3)
            yield [data.astype(np.float32)]

    model = Model()
    concrete_func = model.tf_function.get_concrete_function(
        tf.TensorSpec([1, 3, 4, 3], dtype=tf.float32)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_graph = converter.convert()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": [1, 3, 4, 3]},
        dtype_dict={"x": "int8"},
    )
    mod = partition_for_ethosu(relay_module, params)

    # Generate reference data
    input_data, output_data = generate_ref_data_tflite(tflite_graph)

    def compile_to_main_func(interface_api="c", use_unpacked_api=True):
        test_runner = create_test_runner()
        compiled_models = compile_models(
            models=AOTTestModel(
                module=mod,
                inputs=input_data,
                outputs=output_data,
            ),
            interface_api=interface_api,
            use_unpacked_api=use_unpacked_api,
            workspace_byte_alignment=16,
            pass_config=test_runner.pass_config,
        )
        main_ir_module = compiled_models[0].executor_factory.lowered_ir_mods.items()[0][1]
        main_func = main_ir_module["__tvm_main__"]
        return main_func

    return compile_to_main_func


@pytest.fixture(name="non_device_api_main_func")
def fixture_non_device_api_main_func():
    """Test function generator which does not generate C Device API calls"""
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")

    inputs = OrderedDict([("x", x_data), ("y", y_data)])
    output_list = generate_ref_data(func, inputs)

    def compile_to_main_func(interface_api="c", use_unpacked_api=True):
        test_runner = AOT_DEFAULT_RUNNER
        compiled_models = compile_models(
            models=AOTTestModel(
                module=IRModule.from_expr(func),
                inputs=inputs,
                outputs=output_list,
            ),
            interface_api=interface_api,
            use_unpacked_api=use_unpacked_api,
            workspace_byte_alignment=16,
            pass_config=test_runner.pass_config,
        )
        main_ir_module = list(compiled_models[0].executor_factory.lowered_ir_mods.values())[0]
        main_func = main_ir_module["__tvm_main__"]
        return main_func

    return compile_to_main_func


def test_device_api_hooks_unpacked_api(device_api_main_func):
    """Check for Device API hooks with unpacked internal calls"""
    main_func = device_api_main_func(interface_api="c", use_unpacked_api=True)

    # Activate Device
    assert (
        str(main_func.body[0].value)
        == "T.tvm_check_return(0, -1, T.call_extern("
        + '"int32",'
        + ' "TVMDeviceEthosUActivate",'
        + " device_context_ethos_u))"
    )
    # Open Device
    print("main func", repr(main_func.body))
    assert (
        str(main_func.body[1].value)
        == "T.tvm_check_return(0, -1, T.call_extern("
        + '"int32",'
        + ' "TVMDeviceEthosUOpen",'
        + " device_context_ethos_u))"
    )
    # Device Call
    # We dont need to check exact input and output var names in this test.
    # Hence, using a regex to cover any legal I/O name.
    regex = re.compile(
        r"T\.tvm_check_return\("
        r"0, -1, "
        r'T\.call_extern\("int32", "tvmgen_default_ethos_u_main_0", '
        r"\w+, \w+, device_context_ethos_u\)\)"
    )
    assert regex.match(str(main_func.body[2].value))
    # Close Device
    assert (
        str(main_func.body[3].value)
        == "T.tvm_check_return(0, -1, T.call_extern("
        + '"int32",'
        + ' "TVMDeviceEthosUClose",'
        + " device_context_ethos_u))"
    )
    # Deactivate Device
    assert (
        str(str(main_func.body[4].value))
        == "T.tvm_check_return(0, -1, T.call_extern("
        + '"int32",'
        + ' "TVMDeviceEthosUDeactivate",'
        + " device_context_ethos_u))"
    )


@pytest.mark.skip(
    "Skipping this test as this is incorrectly using Arm(R) Ethos(TM)-U NPU "
    "with packed calling convention which is not supported by the NPU codegen's "
    "TIR to Runtime Hook. We need to use a different target to test this feature"
)
def test_device_api_hooks_packed_api(device_api_main_func):
    """Check for Device API hooks with packed internal calls"""
    main_func = device_api_main_func(interface_api="packed", use_unpacked_api=False)

    # Activate Device
    assert (
        str(main_func.body[0][0].value)
        == "@tir.tvm_check_return(0, -1, tir.call_extern("
        + '"TVMDeviceEthosUActivate",'
        + " device_context_ethos_u: handle,"
        + " dtype=int32))"
    )
    # Open Device
    assert (
        str(main_func.body[1].body.body[0][0][0].value)
        == "@tir.tvm_check_return(0, -1, tir.call_extern("
        + '"TVMDeviceEthosUOpen",'
        + " device_context_ethos_u: handle,"
        + " dtype=int32))"
    )
    # Device Call
    assert (
        str(main_func.body[1].body.body[0][0][1][0].value)
        == "@tir.tvm_call_cpacked("
        + '"tvmgen_default_ethos_u_main_0",'
        + " input: handle, output: handle,"
        + " device_context_ethos_u: handle,"
        + " dtype=int32)"
    )
    # Close Device
    assert (
        str(main_func.body[1].body.body[0][0][2].value)
        == "@tir.tvm_check_return(0, -1, tir.call_extern("
        + '"TVMDeviceEthosUClose",'
        + " device_context_ethos_u: handle,"
        + " dtype=int32))"
    )
    # Deactivate Device
    assert (
        str(main_func.body[2][0].value)
        == "@tir.tvm_check_return(0, -1, tir.call_extern("
        + '"TVMDeviceEthosUDeactivate",'
        + " device_context_ethos_u: handle,"
        + " dtype=int32))"
    )


def test_without_device_api_unpacked_api(non_device_api_main_func):
    """Test a graph without the Device API with the unpacked internal calls"""

    main_func = non_device_api_main_func(interface_api="c", use_unpacked_api=True)
    body = main_func.body.value
    assert (
        repr(body)
        == 'T.tvm_check_return(0, -1, T.call_extern("int32", '
        + '"tvmgen_default_fused_multiply",'
        + " x_buffer_var, y_buffer_var, output_buffer_var))"
    )


def test_without_device_api_packed_api(non_device_api_main_func):
    """Test a graph without the Device API with the packed internal calls"""

    main_func = non_device_api_main_func(interface_api="packed", use_unpacked_api=False)

    body = main_func.body.value
    assert repr(body) == (
        'T.call_cpacked("tvmgen_default_fused_multiply", '
        "T.tvm_stack_make_array(x_buffer_var, T.tvm_stack_make_shape(10, 10), "
        'T.reinterpret("handle", T.uint64(0)), T.uint32(2), T.Cast("float32", 0), 0), '
        "T.tvm_stack_make_array(y_buffer_var, T.tvm_stack_make_shape(1, 10), "
        'T.reinterpret("handle", T.uint64(0)), T.uint32(2), T.Cast("float32", 0), 0), '
        "T.tvm_stack_make_array(output_buffer_var, T.tvm_stack_make_shape(10, 10), "
        'T.reinterpret("handle", T.uint64(0)), T.uint32(2), T.Cast("float32", 0), 0), '
        'T.reinterpret("handle", T.uint64(0)))'
    )


if __name__ == "__main__":
    tvm.testing.main()
