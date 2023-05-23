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

pytest.importorskip("ethosu.vela")
import tvm
from tvm.relay.backend.contrib.ethosu.tir.scheduler import OperatorCompute
import tvm.relay.backend.contrib.ethosu.codegen as codegen
import tensorflow as tf
from . import infra


@pytest.mark.parametrize(
    "axis, ifm_shape, pool_shape",
    [
        (1, (1, 12, 1, 2), (3, 1)),
        (1, (1, 12, 12, 2), (3, 3)),
        (2, (1, 1, 12, 2), (1, 3)),
        (2, (1, 12, 12, 2), (3, 3)),
    ],
)
def test_rolling_buffer_2_layers(axis, ifm_shape, pool_shape):
    accel_type = "ethos-u55-256"
    strides = (1, 1)

    @tf.function
    def tf_model(x):
        padding = "VALID"
        pool_0 = tf.nn.max_pool(x, pool_shape, strides, padding)
        pool_1 = tf.nn.max_pool(pool_0, pool_shape, strides, padding)
        return pool_1

    def _cascader(cached_func, const_dict, sch):
        pool_b_out = cached_func.outputs[0]
        pool_b_compute = OperatorCompute.from_output(pool_b_out)

        pool_a_out = pool_b_compute.read.op.input_tensors[0]
        pool_a_compute = OperatorCompute.from_output(pool_a_out)

        outer = pool_b_compute.split(sch, axis=axis, val=4)
        pool_a_compute.compute_at(sch, stage=sch[pool_b_out], axis=outer)
        pool_a_compute.rolling_buffer(sch)

    codegen.SCHEDULER = lambda: _cascader
    infra.compare_tvm_with_tflite(tf_model, [ifm_shape], accel_type)


@pytest.mark.parametrize(
    "axis, ifm_shape, pool_shape",
    [
        (1, (1, 12, 1, 2), (3, 1)),
        (1, (1, 12, 1, 17), (3, 1)),
        (1, (1, 12, 12, 2), (3, 3)),
        (1, (1, 12, 12, 17), (3, 3)),
        (2, (1, 1, 12, 2), (1, 3)),
        (2, (1, 1, 12, 17), (1, 3)),
        (2, (1, 12, 12, 2), (3, 3)),
        (2, (1, 12, 12, 17), (3, 3)),
    ],
)
def test_rolling_buffer_3_layers(axis, ifm_shape, pool_shape):
    accel_type = "ethos-u55-256"
    strides = (1, 1)

    @tf.function
    def tf_model(x):
        padding = "VALID"
        pool_0 = tf.nn.max_pool(x, pool_shape, strides, padding)
        pool_1 = tf.nn.max_pool(pool_0, pool_shape, strides, padding)
        pool_2 = tf.nn.max_pool(pool_1, pool_shape, strides, padding)
        return pool_2

    def _cascader(cached_func, const_dict, sch):
        pool_b_out = cached_func.outputs[0]
        pool_b_compute = OperatorCompute.from_output(pool_b_out)

        pool_a_out = pool_b_compute.read.op.input_tensors[0]
        pool_a_compute = OperatorCompute.from_output(pool_a_out)

        outer = pool_b_compute.split(sch, axis=axis, val=4)
        pool_a_compute.compute_at(sch, stage=sch[pool_b_out], axis=outer)
        pool_a_compute.rolling_buffer(sch)

    codegen.SCHEDULER = lambda: _cascader
    infra.compare_tvm_with_tflite(tf_model, [ifm_shape], accel_type)


if __name__ == "__main__":
    tvm.testing.main()
