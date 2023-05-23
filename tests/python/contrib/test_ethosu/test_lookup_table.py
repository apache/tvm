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
# pylint: disable=invalid-name, unused-argument
import pytest

pytest.importorskip("ethosu.vela")
import numpy as np
import tflite.Model

import tvm
import tensorflow as tf
from tvm import relay
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.relay.build_module import bind_params_by_name  # type: ignore

from . import infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_tflite_lut_activations(accel_type):

    dtype = "int8"
    ifm_shape = (1, 55, 55, 3)

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_func(self, x):
                weight_shape = (3, 3, ifm_shape[3], 4)
                weight = tf.constant(
                    np.random.uniform(low=0, high=0.3, size=weight_shape), dtype=tf.float32
                )
                # The input strides to the TensorFlow API needs to be of shape 1x4
                op = tf.nn.conv2d(x, weight, strides=(1, 2, 2, 1), padding="SAME", dilations=(1, 1))
                op = tf.nn.tanh(op)
                op = tf.nn.tanh(op)

                weight_shape2 = (2, 3, 4, 1)
                weight2 = tf.constant(
                    np.random.uniform(low=0, high=0.3, size=weight_shape2), dtype=tf.float32
                )
                op = tf.nn.depthwise_conv2d(
                    op, weight2, strides=(1, 1, 1, 1), padding="VALID", dilations=(2, 2)
                )
                op = tf.nn.sigmoid(op)
                op = tf.nn.max_pool(op, (1, 1), strides=(1, 1, 1, 1), padding="SAME")
                op = tf.nn.tanh(op)
                return op

        model = Model()
        concrete_func = model.tf_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = 0.7 * np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    relay_module, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = partition_for_ethosu(relay_module, params)

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    test_runner = infra.create_test_runner(accel_type)
    compiled_models = infra.build_source(
        mod,
        input_data,
        output_data,
        test_runner,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
    compilation_artifacts = get_artifacts(ethosu_module)
    cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
    infra.print_payload(cmms)
    infra.verify_source(compiled_models, test_runner)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_random_lut(accel_type):

    dtype = "int8"
    ifm_shape = (1, 55, 55, 3)

    lut_data = np.random.randint(-128, high=127, size=[256])
    lut_data_map = {idx: lut_data[idx + 128] for idx in range(-128, 128)}

    in_data = np.random.randint(-128, high=127, size=ifm_shape, dtype=dtype)
    out_data = np.array([lut_data_map[i] for i in in_data.ravel()]).reshape(ifm_shape).astype(dtype)

    ifm = relay.var("ifm", shape=ifm_shape, dtype=dtype)
    ifm0 = relay.var("ifm0", shape=ifm_shape, dtype=dtype)
    lut1 = relay.var("lut1", shape=(256,), dtype="uint8")

    identity = infra.make_ethosu_identity(ifm0, lut=lut1, activation="LUT")
    glb_ethosu = relay.GlobalVar("tvmgen_default_ethos_u_main_0")

    func = (
        relay.Function([ifm0, lut1], identity)
        .with_attr("Inline", 1)
        .with_attr("Compiler", "ethos-u")
        .with_attr("global_symbol", "tvmgen_default_ethos_u_main_0")
        .with_attr("Primitive", 1)
    )

    params = {"lut1": tvm.nd.array(lut_data.astype("uint8"))}
    func = bind_params_by_name(func, params)

    mod = tvm.IRModule()
    mod[glb_ethosu] = func
    mod = relay.transform.InferType()(mod)

    call = relay.Call(glb_ethosu, [ifm])
    mod["main"] = relay.Function([ifm], call)
    mod = relay.transform.InferType()(mod)

    test_runner = infra.create_test_runner(accel_type)
    compiled_models = infra.build_source(
        mod,
        {"ifm": in_data},
        {"output": out_data},
        test_runner,
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
    compilation_artifacts = get_artifacts(ethosu_module)
    cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
    infra.print_payload(cmms)
    infra.verify_source(compiled_models, test_runner)


if __name__ == "__main__":
    tvm.testing.main()
