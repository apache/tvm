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
from tests.python.contrib.test_ethosu.test_end_to_end import comparison_infra

ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


def test_ethosu_section_name():
    @tf.function
    def depthwise_conv2d(x):
        weight_shape = [3, 3, 3, 1]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        tf_strides = [1, 1, 1, 1]
        op = tf.nn.depthwise_conv2d(x, weight, strides=tf_strides, padding="SAME", dilations=(2, 2))
        return op

    mod, tflite_graph = comparison_infra._get_tflite_graph(depthwise_conv2d, [(1, 55, 55, 3)])

    # Generate reference data
    input_data, output_data = infra.generate_ref_data_tflite(tflite_graph)

    compiled_models = infra.build_source(mod, input_data, output_data)

    # Assumes only two runtime.Modules are created -- i.e. single offload module
    ethosu_module = compiled_models[0].executor_factory.lib.imported_modules[0].imported_modules[0]

    # Verify generated C source
    source = ethosu_module.get_source()
    assert (
        '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t tvmgen_default_ethos_u_main_0_cms_data_data'
        in source
    )
    assert (
        '__attribute__((section(".rodata.tvm"), aligned(16))) static int8_t tvmgen_default_ethos_u_main_0_weights'
        in source
    )


if __name__ == "__main__":
    pytest.main([__file__])
