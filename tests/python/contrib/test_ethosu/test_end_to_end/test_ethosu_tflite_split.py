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


# This codegen test checks both, split and split_v
@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape, num_or_size_splits, axis",
    [
        ((1, 4, 6, 8), (1, 3, 4), 3),
        ((4, 6, 8), 2, 0),
        ((50,), 25, 0),
        ((5, 11), 1, 1),
        ((13,), (13,), 0),
        ((22, 7), (4, -1), 1),
    ],
)
def test_tflite_split(accel_type, ifm_shape, num_or_size_splits, axis):
    @tf.function
    def split_func(x):
        op = tf.split(x, num_or_size_splits, axis=axis)
        return op

    comparison_infra._compare_tvm_with_tflite(split_func, [ifm_shape], accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
