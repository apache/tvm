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
"""Unit tests for converting TensorFlow debugging ops to Relay."""
try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
import numpy as np
from tvm import relay, ir, testing
from tvm.relay.frontend.tensorflow import from_tensorflow


def run_relay(graph):
    with testing.disable_span_filling():
        mod, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    with testing.enable_span_filling():
        mod_with_span, _ = relay.frontend.from_tensorflow(graph.as_graph_def(add_shapes=True))
    assert ir.structural_equal(mod["main"], mod_with_span["main"])

    return relay.create_executor("debug", mod=mod).evaluate()(**params)


def test_no_op():
    g = tf.Graph()
    with g.as_default():
        no_op = tf.no_op()
        with tf.Session() as sess:
            # In TF, the type of a no-op is None.
            assert sess.run(no_op) is None

        # In TVM, no-op is currently translated to 0, though it should
        # probably be none or an empty tuple.
        np.testing.assert_allclose(0, run_relay(g).numpy())


if __name__ == "__main__":
    test_no_op()
