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
import os
import tvm
from tvm import te
import numpy as np
from tvm import rpc
from tvm.contrib import utils, tflite_runtime

# import tflite_runtime.interpreter as tflite

# NOTE: This script was tested on tensorflow/tflite (v2.4.1)


def skipped_test_tflite_runtime():
    def get_tflite_model_path(target_edgetpu):
        # Return a path to the model
        edgetpu_path = os.getenv("EDGETPU_PATH", "/home/mendel/edgetpu")
        # Obtain mobilenet model from the edgetpu repo path
        if target_edgetpu:
            model_path = os.path.join(
                edgetpu_path, "test_data/mobilenet_v1_1.0_224_quant_edgetpu.tflite"
            )
        else:
            model_path = os.path.join(edgetpu_path, "test_data/mobilenet_v1_1.0_224_quant.tflite")
        return model_path

    def init_interpreter(model_path, target_edgetpu):
        # Initialize interpreter
        if target_edgetpu:
            edgetpu_path = os.getenv("EDGETPU_PATH", "/home/mendel/edgetpu")
            libedgetpu = os.path.join(edgetpu_path, "libedgetpu/direct/aarch64/libedgetpu.so.1")
            interpreter = tflite.Interpreter(
                model_path=model_path, experimental_delegates=[tflite.load_delegate(libedgetpu)]
            )
        else:
            interpreter = tflite.Interpreter(model_path=model_path)
        return interpreter

    def check_remote(server, target_edgetpu=False):
        tflite_model_path = get_tflite_model_path(target_edgetpu)

        # inference via tflite interpreter python apis
        interpreter = init_interpreter(tflite_model_path, target_edgetpu)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]["shape"]
        tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
        interpreter.set_tensor(input_details[0]["index"], tflite_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]["index"])

        # inference via remote tvm tflite runtime
        remote = rpc.connect(server.host, server.port)
        dev = remote.cpu(0)
        if target_edgetpu:
            runtime_target = "edge_tpu"
        else:
            runtime_target = "cpu"

        with open(tflite_model_path, "rb") as model_fin:
            runtime = tflite_runtime.create(model_fin.read(), dev, runtime_target)
            runtime.set_input(0, tvm.nd.array(tflite_input, dev))
            runtime.invoke()
            out = runtime.get_output(0)
            np.testing.assert_equal(out.numpy(), tflite_output)

    # Target CPU on coral board
    check_remote(rpc.Server("127.0.0.1"))
    # Target EdgeTPU on coral board
    check_remote(rpc.Server("127.0.0.1"), target_edgetpu=True)


if __name__ == "__main__":
    # skipped_test_tflite_runtime()
    pass
