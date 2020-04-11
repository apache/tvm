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
except ImportError:
    import tensorflow as tf
import numpy as np
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow

def run_relay(graph):
    mod, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('debug', mod=mod)
    return ex.evaluate()(**params)

def test_no_op():
    g = tf.Graph()
    with g.as_default():
        no_op = tf.no_op()
        with tf.Session() as sess:
            # In TF, the type of a no-op is None.
            assert sess.run(no_op) is None

        # In TVM, no-op is currently translated to 0, though it should
        # probably be none or an empty tuple.
        np.testing.assert_allclose(0, run_relay(g).asnumpy())


if __name__ == "__main__":
    test_no_op()

