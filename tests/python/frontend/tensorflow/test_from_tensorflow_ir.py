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
"""Unit tests for converting TensorFlow graph to Relay ir."""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
import numpy as np
import tvm
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow
from tensorflow.python.ops import nn

SEMVER = '#[version = "0.0.5"]\n'

def run_from_tensorflow(graph):
    mod, _ = from_tensorflow(graph.as_graph_def(add_shapes=True))
    return mod

def test_moments():
    g = tf.Graph()
    shape = [4, 176, 8, 8]
    dtype = "float32"
    with g.as_default():
         A = tf.placeholder(shape=shape, dtype=dtype, name="A")
         B = tf.placeholder(shape=shape, dtype=dtype, name="B")
         mean, variance = tf.nn.moments(A, [1], keep_dims=True)
         normalised_input = (A - mean) / tf.sqrt(variance + 0.0005)
         
    mod = run_from_tensorflow(g)
    program = """
    def @main(%A: Tensor[(4, 176, 8, 8), float32]) {
        %527 = mean(%A, axis=[1], keepdims=True) /* moments/mean */;
        %528 = subtract(%A, %527) /* sub */;
        %529 = subtract(%A, %527);
        %530 = multiply(%529, %529) /* moments/SquaredDifference */;
        %531 = mean(%530, axis=[1], keepdims=True) /* moments/variance */;
        %532 = add(%531, 0.0005f) /* add */;
        %533 = sqrt(%532) /* Sqrt */;
        divide(%528, %533) /* truediv */
    }
    """
    mod_golden = tvm.parser.parse(SEMVER + program)
    tvm.ir.assert_structural_equal(mod["main"].body, mod_golden["main"].body, map_free_vars=True)

if __name__ == "__main__":
    test_moments()
