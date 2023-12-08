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
"""Configure pytest"""
import pytest
import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import utils, tflite_runtime


def _create_tflite_model():
    """Functions of creating a tflite model"""
    if not tvm.runtime.enabled("tflite"):
        print("skip because tflite runtime is not enabled...")
        return None
    if not tvm.get_global_func("tvm.tflite_runtime.create", True):
        print("skip because tflite runtime is not enabled...")
        return None

    try:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
    except ImportError:
        print("skip because tensorflow not installed...")
        return None

    root = tf.Module()
    root.const = tf.constant([1.0, 2.0], tf.float32)
    root.f = tf.function(lambda x: root.const * x)

    input_signature = tf.TensorSpec(
        shape=[
            2,
        ],
        dtype=tf.float32,
    )
    concrete_func = root.f.get_concrete_function(input_signature)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()
    return tflite_model


@pytest.mark.skip("skip because accessing output tensor is flakey")
def test_local():
    """Local tests of tflite model"""
    if not tvm.runtime.enabled("tflite"):
        print("skip because tflite runtime is not enabled...")
        return
    if not tvm.get_global_func("tvm.tflite_runtime.create", True):
        print("skip because tflite runtime is not enabled...")
        return

    try:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
    except ImportError:
        print("skip because tensorflow not installed...")
        return

    tflite_fname = "model.tflite"
    tflite_model = _create_tflite_model()
    temp = utils.tempdir()
    tflite_model_path = temp.relpath(tflite_fname)
    open(tflite_model_path, "wb").write(tflite_model)

    # inference via tflite interpreter python apis
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], tflite_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    # inference via tvm tflite runtime
    with open(tflite_model_path, "rb") as model_fin:
        runtime = tflite_runtime.create(model_fin.read(), tvm.cpu(0))
        runtime.set_input(0, tvm.nd.array(tflite_input))
        runtime.invoke()
        out = runtime.get_output(0)
        np.testing.assert_equal(out.numpy(), tflite_output)


def test_remote():
    """Remote tests of tflite model"""
    if not tvm.runtime.enabled("tflite"):
        print("skip because tflite runtime is not enabled...")
        return
    if not tvm.get_global_func("tvm.tflite_runtime.create", True):
        print("skip because tflite runtime is not enabled...")
        return

    try:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
    except ImportError:
        print("skip because tensorflow not installed...")
        return

    tflite_fname = "model.tflite"
    tflite_model = _create_tflite_model()
    temp = utils.tempdir()
    tflite_model_path = temp.relpath(tflite_fname)
    open(tflite_model_path, "wb").write(tflite_model)

    # inference via tflite interpreter python apis
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], tflite_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]["index"])

    # inference via remote tvm tflite runtime
    def check_remote(server):
        remote = rpc.connect(server.host, server.port)

        with open(tflite_model_path, "rb") as model_fin:
            runtime = tflite_runtime.create(model_fin.read(), remote.cpu(0))
            runtime.set_input(0, tvm.nd.array(tflite_input, remote.cpu(0)))
            runtime.invoke()
            out = runtime.get_output(0)
            np.testing.assert_equal(out.numpy(), tflite_output)

    check_remote(rpc.Server("127.0.0.1"))


if __name__ == "__main__":
    test_local()
    test_remote()
