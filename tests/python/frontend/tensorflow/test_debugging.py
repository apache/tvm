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


def run_relay(graph, shape_dict=None, *vars):
    with testing.disable_span_filling():
        mod, params = from_tensorflow(graph.as_graph_def(add_shapes=True), shape=shape_dict)
    with testing.enable_span_filling():
        mod_with_span, _ = relay.frontend.from_tensorflow(
            graph.as_graph_def(add_shapes=True), shape=shape_dict
        )
    assert ir.structural_equal(mod["main"], mod_with_span["main"])

    return relay.create_executor("debug", mod=mod).evaluate()(*vars)


def test_assert_true():
    g = tf.Graph()
    shape = (1, 2)
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=shape, name="input")
        assert_op = tf.Assert(tf.reduce_all(tf.less_equal(x, x)), ["it failed"])

        with tf.Session() as sess:
            x_value = np.random.rand(*shape)
            assert sess.run(assert_op, feed_dict={x: x_value}) is None

        # In TVM, tf.assert is converted to a no-op which is actually a 0,
        # though it should probably be none or an empty tuple.
        #
        # ToDo: It appears that the frontend converter gets confused here and
        # entirely eliminates all operands from main(). Likely because x <= x
        # is always true, so the placeholder can be eliminated. But TF doesn't
        # do that, it's happening in Relay, and that optimization shouldn't
        # affect the arity of the main function. We should have to pass in
        # x_value here.
        np.testing.assert_allclose(0, run_relay(g, {"input": shape}).numpy())


def test_assert_true_var_capture():
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, shape=())

        # It turns out that tf.assert() creates a large and complex subgraph if
        # you capture a variable as part of the error message. So we need to
        # test that, too.
        assert_op = tf.Assert(tf.less_equal(x, x), ["it failed", x])

        with tf.Session() as sess:
            x_value = np.random.rand()
            assert sess.run(assert_op, feed_dict={x: x_value}) is None

        # TODO: The frontend converter notes the output of
        # the graph as a boolean, which is not correct - as you can see above,
        # TF believes that the value of this graph is None.
        np.testing.assert_allclose(True, run_relay(g, None, x_value).numpy())


def test_assert_false():
    g = tf.Graph()
    with g.as_default():
        assert_op = tf.Assert(tf.constant(False), ["it failed"])

        with tf.Session() as sess:
            try:
                print(sess.run(assert_op))
                assert False  # TF should have thrown an exception
            except tf.errors.InvalidArgumentError as e:
                assert "it failed" in e.message

        # In TVM, tf.assert is converted to a no-op which is actually a 0,
        # though it should probably be none or an empty tuple. For the same
        # reason, there should not be an error here, even though the assertion
        # argument is false.
        np.testing.assert_allclose(0, run_relay(g).numpy())


if __name__ == "__main__":
    test_assert_true()
    test_assert_true_var_capture()
    test_assert_false()
