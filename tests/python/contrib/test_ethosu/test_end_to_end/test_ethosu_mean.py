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

from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.backend.contrib.ethosu import preprocess

from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tests.python.relay.aot.aot_test_utils import generate_ref_data

from tests.python.contrib.test_ethosu import infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize(
    "accel_type",
    ACCEL_TYPES,
)
@pytest.mark.parametrize(
    "ifm_shape, axis, keep_dims, use_same_quantization",
    [
        # mean to depthwise + multiply
        [(1, 8, 16, 16), (1, 2), True, False],
        [(1, 3, 4), (0, 1), True, False],
        [(1, 65, 2, 1), (1, 2), True, False],  # special case when h > 64
        # mean to average pool
        [(1, 8, 16, 16), (2,), False, True],
        [(3, 3, 4), (0,), True, True],
        [(8, 5), (0,), False, True],
        # mean to depthwise
        [(1, 8, 16, 16), (2,), True, False],
        [(1, 8, 16, 16), (2, 1), False, False],
        [(8, 4), (0,), False, False],
    ],
)
def test_mean(accel_type, ifm_shape, axis, keep_dims, use_same_quantization):
    dtype = "int8"

    def create_mod_from_tflite():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                op = tf.math.reduce_mean(x, axis=axis, keepdims=keep_dims)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_graph = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"ifm": ifm_shape},
            dtype_dict={"ifm": dtype},
        )
        input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)
        return mod, input_data, output_data

    def create_mod_from_relay():
        ifm = relay.var("input", shape=ifm_shape, dtype=dtype)
        cast = relay.cast(ifm, dtype="int32")
        mean = relay.mean(cast, axis=axis, keepdims=keep_dims)
        requantize = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )

        func = relay.Function(relay.analysis.free_vars(requantize), requantize)
        mod = tvm.IRModule.from_expr(func)

        input_data = {"input": np.random.randint(low=-127, high=128, size=ifm_shape, dtype=dtype)}
        output_data = generate_ref_data(mod, input_data)
        return mod, input_data, output_data

    mod, input_data, output_data = (
        create_mod_from_relay() if use_same_quantization else create_mod_from_tflite()
    )
    mod = partition_for_ethosu(mod)

    # TODO(lhutton1) For now output is not bit exact with TFLite.
    # This is because TFLite reference kernels are not being used.
    # For this, TFLite will need upgrading to 2.6.
    compiled_models = infra.build_source(
        mod, input_data, output_data, accel_type, output_tolerance=1
    )

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    get_artifacts = tvm._ffi.get_global_func("runtime.module.ethos-u.get_artifacts")
    compilation_artifacts = get_artifacts(ethosu_module)
    cmms = bytes.fromhex(compilation_artifacts[0].command_stream)
    infra.print_payload(cmms)
    infra.verify_source(compiled_models, accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
