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
"""Common utilities for creating TFLite models"""
from distutils.version import LooseVersion
import numpy as np
import pytest
import tvm

pytest.importorskip("tflite")
pytest.importorskip("tensorflow")
import tflite.Model
import tensorflow as tf


class TFLiteModel:
    """Creates TFLite Model and facilitates reference data generation"""

    def __init__(self, dtype):
        self.serial_model = None  # This is what TFLite convert() provides
        self.dtype = dtype  # This is the dtype of graph inputs
        self.shape_dict = {}
        self.dtype_dict = {}

    @tf.function
    def conv2d_single_function(self, ifm_tensor, args):
        """Returns TFLite Conv2d layer"""
        assert len(args) == 6, "Conv2D needs (ifm_shape, kernel_shape, strides, padding, dilation)"
        _, kernel_shape, strides, padding, dilation, activation = args
        op = tf.nn.conv2d(
            ifm_tensor,
            filters=tf.constant(
                np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 3, 3]),
                dtype=tf.float32,
            ),
            strides=[1, strides[0], strides[1], 1],
            padding=padding,
            dilations=dilation,
        )
        if activation == "RELU":
            op = tf.nn.relu(op)
        elif activation == "NONE":
            pass
        else:
            assert False, "Unsupported activation {}".format(activation)
        return op

    def create_tflite_model(self, op_type, *args):
        """Returns TFLite serial graph, Relay module, Relay params based on op_type"""
        concrete_func = None
        input_shape = None
        if op_type == "conv2d_single":
            input_shape = args[0]
            ifm_tensor = tf.TensorSpec(input_shape, dtype=tf.float32, name="input")
            concrete_func = self.conv2d_single_function.get_concrete_function(ifm_tensor, args)
        else:
            assert False, "Unsupported op_type {}".format(op_type)

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(input_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        self.serial_model = converter.convert()
        self.shape_dict = {"input": input_shape}
        self.dtype_dict = {"input": self.dtype}

    def convert_to_relay(self):
        """Converts TFLite serialized graph into Relay"""
        assert self.serial_model is not None, "TFLite model is empty!"

        tflite_model = tflite.Model.Model.GetRootAsModel(self.serial_model, 0)
        relay_module, relay_params = tvm.relay.frontend.from_tflite(
            tflite_model, self.shape_dict, self.dtype_dict
        )
        return relay_module, relay_params

    def generate_randomized_input_data(self, seed, shape, dtype):
        """Generates randomized input numpy arrays based on shape and dtype."""
        random_state = np.random.RandomState(seed)
        random_data = None
        if dtype == np.float32:
            random_data = random_state.uniform(-1, 1, size).astype(dtype)
        else:
            low = np.iinfo(dtype).min
            high = np.iinfo(dtype).max + 1
            random_data = random_state.randint(low, high, shape, dtype)
        return random_data

    # pylint: disable=import-outside-toplevel
    def generate_reference_data(self):
        """
        This method uses TFLite reference kernels to generate reference output.
        It returns randomized inputs and reference outputs.
        """
        assert self.serial_model is not None, "TFLite model was not created."

        output_tolerance = None
        if tf.__version__ < LooseVersion("2.5.0"):
            output_tolerance = 1
            interpreter = tf.lite.Interpreter(model_content=self.serial_model)
        else:
            output_tolerance = 0
            interpreter = tf.lite.Interpreter(
                model_content=self.serial_model,
                experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                experimental_preserve_all_tensors=False,
            )

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Generate predictable randomized input
        seed = 0
        input_data = {}
        for input_detail in input_details:
            input_values = self.generate_randomized_input_data(
                seed, input_detail["shape"], input_detail["dtype"]
            )
            interpreter.set_tensor(input_detail["index"], input_values)
            input_data.update({input_detail["name"]: input_values})

        interpreter.invoke()

        # Obtain the expected output from interpreter
        expected_output_data = {}
        for output_detail in output_details:
            expected_output_data.update(
                {output_detail["name"]: interpreter.get_tensor(output_detail["index"])}
            )

        return input_data, expected_output_data, output_tolerance
