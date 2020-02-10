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
from tvm import rpc
from tvm.contrib import util, tflite_runtime
# import tensorflow as tf
# import tflite_runtime.interpreter as tflite


def skipped_test_tflite_runtime():

    def create_tflite_model():
        root = tf.Module()
        root.const = tf.constant([1., 2.], tf.float32)
        root.f = tf.function(lambda x: root.const * x)
        
        input_signature = tf.TensorSpec(shape=[2,  ], dtype=tf.float32)
        concrete_func = root.f.get_concrete_function(input_signature)
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        tflite_model = converter.convert()
        return tflite_model


    def check_local():
        tflite_fname = "model.tflite"
        tflite_model = create_tflite_model()
        temp = util.tempdir()
        tflite_model_path = temp.relpath(tflite_fname)
        open(tflite_model_path, 'wb').write(tflite_model)

        # inference via tflite interpreter python apis
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # inference via tvm tflite runtime
        with open(tflite_model_path, 'rb') as model_fin:
            runtime = tflite_runtime.create(model_fin.read(), tvm.cpu(0))
            runtime.set_input(0, tvm.nd.array(tflite_input))
            runtime.invoke()
            out = runtime.get_output(0)
            np.testing.assert_equal(out.asnumpy(), tflite_output)


    def check_remote():
        tflite_fname = "model.tflite"
        tflite_model = create_tflite_model()
        temp = util.tempdir()
        tflite_model_path = temp.relpath(tflite_fname)
        open(tflite_model_path, 'wb').write(tflite_model)

        # inference via tflite interpreter python apis
        interpreter = tflite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        tflite_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        # inference via remote tvm tflite runtime
        server = rpc.Server("localhost")
        remote = rpc.connect(server.host, server.port)
        ctx = remote.cpu(0)
        a = remote.upload(tflite_model_path)

        with open(tflite_model_path, 'rb') as model_fin:
            runtime = tflite_runtime.create(model_fin.read(), remote.cpu(0))
            runtime.set_input(0, tvm.nd.array(tflite_input, remote.cpu(0)))
            runtime.invoke()
            out = runtime.get_output(0)
            np.testing.assert_equal(out.asnumpy(), tflite_output)

    check_local()
    check_remote()

if __name__ == "__main__":
    # skipped_test_tflite_runtime()
    pass
