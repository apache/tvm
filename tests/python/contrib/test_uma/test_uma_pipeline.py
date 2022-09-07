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

pytest.importorskip("tflite")
pytest.importorskip("tensorflow")

import os
import tensorflow as tf
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
from tvm.relay import transform, testing
from tvm.testing.aot import (
    AOTTestModel,
    AOTTestRunner,
    generate_ref_data,
    compile_and_run,
    create_relay_module_and_inputs_from_tflite_file,
)

import tvm
from test_uma_vanilla_accelerator import VanillaAcceleratorBackend
from tvm import relay
import numpy as np
from collections import OrderedDict

from tvm.relay.backend.contrib.uma.api.utils import uma_available

pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


@pytest.mark.parametrize(
    "interface_api,use_unpacked_api,test_runner,groups,weight_shape",
    [("c", True, AOT_DEFAULT_RUNNER, 1, 32)],
)
def test_conv2d(interface_api, use_unpacked_api, test_runner, groups, weight_shape):
    """Test a subgraph with a single conv2d operator."""
    mod, inputs, output_list, test_runner = create_conv2d(groups, test_runner, weight_shape)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))

    compile_and_run(
        AOTTestModel(module=mod, inputs=inputs, outputs=output_list),
        test_runner,
        interface_api,
        use_unpacked_api,
        target=target,
    )


def create_conv2d(groups=1, test_runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "float32"
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
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, test_runner


def _generate_runtime_data(input_shapes: dict, output_shapes: dict) -> [OrderedDict, OrderedDict]:
    assert len(input_shapes) == 1
    assert len(output_shapes) == 1

    iname = list(input_shapes.keys())[0]
    oname = list(output_shapes.keys())[0]
    ishape = input_shapes[iname]
    oshape = output_shapes[oname]
    i_data = np.random.uniform(0, 1, ishape).astype("float32")
    o_data = np.random.uniform(0, 1, oshape).astype("float32")
    oname = "output"  # name set by relay.build in executor_codegen_metadata.outputs
    inputs = OrderedDict([(iname, i_data)])
    outputs = OrderedDict([(oname, o_data)])
    return inputs, outputs


def test_mobilenet():
    """Full network test with Mobilenet"""
    use_unpacked_api = True
    interface_api = "c"
    test_runner = AOT_DEFAULT_RUNNER

    mod, params = testing.mobilenet.get_workload(batch_size=1)

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    input_list = {"data": data}
    output_list = generate_ref_data(mod, input_list, params)
    mod = uma_backend.partition(mod)
    aot_test_model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list, params=params)

    compile_and_run(
        aot_test_model,
        test_runner,
        interface_api,
        use_unpacked_api,
        workspace_byte_alignment=1,
        debug_calculated_workspaces=False,
        target=[target_c, target],
    )


def test_tflite_model():
    """
    End-to-end test of TF-Lite file using UMA
    """
    tflite_file = "/tmp/model.tflite"
    if os.path.exists(tflite_file):
        os.remove(tflite_file)
    generate_tflite_file(tflite_file)

    pytest.importorskip("tflite")

    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    tf_model_details = interpreter.get_input_details()
    mod, _, params = create_relay_module_and_inputs_from_tflite_file(
        tflite_file, bind_params_by_name=False
    )

    uma_backend = VanillaAcceleratorBackend()
    uma_backend.register()
    target = tvm.target.Target("vanilla_accelerator", host=tvm.target.Target("c"))
    target_c = tvm.target.Target("c")

    # Generation of test input and output
    data_shape = [int(x) for x in mod["main"].params[0].type_annotation.shape]
    data = np.random.uniform(size=data_shape).astype("float32")
    input_list = {str(tf_model_details[0]["name"]): data}
    output_list = generate_ref_data(mod, input_list, params)

    # UMA partitioning (needs to be done after generate_ref_data)
    mod = uma_backend.partition(mod)

    aot_test_model = AOTTestModel(module=mod, inputs=input_list, outputs=output_list, params=params)
    test_runner = AOTTestRunner(
        pass_config={"tir.usmp.enable": True, "tir.usmp.algorithm": "greedy_by_size"}
    )

    compile_and_run(
        aot_test_model,
        test_runner,
        interface_api="c",
        use_unpacked_api=True,
        workspace_byte_alignment=1,
        debug_calculated_workspaces=False,
        target=[target_c, target],
    )


def generate_tflite_file(tflite_filename):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    tf_model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    output = tf_model(x_train[:1])
    output = output.numpy()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss(y_train[:1], output).numpy()
    tf_model.compile(metrics=["accuracy"], optimizer="adam", loss=loss)
    tf_model.fit(x_train, y_train, epochs=1)

    tflite_converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = tflite_converter.convert()
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    tvm.testing.main()
