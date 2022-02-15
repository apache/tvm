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
from tests.python.contrib.test_ethosu.end_to_end import comparison_infra


@pytest.mark.parametrize("weight_min, weight_max", [(0.0, 1e-11), (-1e10, 1e10)])
def test_out_of_range_scaling(weight_min, weight_max):
    ifm_shape = (1, 6, 6, 2)
    strides = (1, 1)
    kernel_shape = (1, 1)
    dilation = (1, 1)
    padding = "SAME"
    activation = "RELU"
    accel_type = "ethos-u55-128"

    @tf.function
    def conv_invalid_scale(x):
        # Use tf.nn API to create the model
        tf_strides = [1, strides[0], strides[1], 1]
        weights = np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 2, 2])
        # Overwrite to force quantization that produces out of range shift values
        weights[0][0][0][0] = weight_min
        weights[0][0][1][0] = weight_max
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                weights,
                dtype=tf.float32,
            ),
            strides=tf_strides,
            padding=padding,
            dilations=dilation,
        )
        if activation:
            op = tf.nn.relu(op)
        return op

    comparison_infra._compare_tvm_with_tflite(conv_invalid_scale, [ifm_shape], accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
