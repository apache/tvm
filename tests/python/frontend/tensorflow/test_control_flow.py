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
"""Unit tests for converting TensorFlow control flow op to Relay."""
import pytest
import tensorflow as tf
import numpy as np
from tvm import nd
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow


def check_equal(graph, tf_out):
    mod, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('vm', mod=mod)
    relay_out = ex.evaluate()(**params)
    if isinstance(relay_out, nd.NDArray):
        np.testing.assert_allclose(tf_out, relay_out.asnumpy())
    else:
        if not isinstance(tf_out, list):
            tf_out = [tf_out]
        for x, y in zip(tf_out, [r.asnumpy() for r in relay_out]):
            np.testing.assert_allclose(x, y)


def test_vanilla_loop():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(0)

        def c(i): return tf.less(i, 10)

        def b(i): return tf.add(i, 1)

        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def test_callnode_loop_vars():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.add(tf.constant(0), 1)

        def c(i): return tf.less(i, 10)

        def b(i): return tf.add(i, 1)

        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def test_loop_2_vars():
    graph = tf.Graph()
    with graph.as_default():
        i0 = tf.constant(0)
        j0 = tf.ones([2, 2])

        def c(i, j): return i < 10

        def b(i, j): return [tf.add(i, 1), j]

        i1, i2 = tf.while_loop(c, b, loop_vars=[i0, j0])
        i1 += tf.constant(1337)

        with tf.Session() as sess:
            tf_out = sess.run(i1)

    check_equal(graph, tf_out)


def test_loop_3_vars():
    graph = tf.Graph()
    with graph.as_default():
        i0 = tf.constant(1)
        j0 = tf.constant(2)
        k0 = tf.constant(4)

        def c(i, j, k): return i < 10

        def b(i, j, k): return [i+1, j * k, k + i]
        r = tf.while_loop(c, b, loop_vars=[i0, j0, k0])

        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_loop_conditions():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(1)
        j = tf.constant(1)
        k = tf.constant(5)

        def c(i, j, k): return \
            tf.equal(tf.not_equal(tf.less(i + j, 10),
                                  tf.less(j * k, 100)),
                     tf.greater_equal(k, i + j))

        def b(i, j, k): return [i+j, j+k, k+1]
        r = tf.while_loop(c, b, loop_vars=[i, j, k])
        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


@pytest.mark.skip
def test_loop_bodies():
    graph = tf.Graph()
    with graph.as_default():
        def body(x):
            a = tf.constant(np.array([[5, 6], [7, 8]]), dtype=tf.int32)
            b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
            c = a + b
            return tf.nn.relu(x + c)

        def condition(x):
            return tf.reduce_sum(x) < 100
        x = tf.constant(0, shape=[2, 2])
        r = tf.while_loop(condition, body, [x])
        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_nested_loop():
    graph = tf.Graph()
    with graph.as_default():

        def body(x):
            def nest_body(c):
                return tf.multiply(c, 2)
            def cd(c): return tf.less(c, 10)
            c = tf.constant(2)
            res = tf.while_loop(cd, nest_body, loop_vars=[c])
            return tf.nn.relu(x + res)

        def condition(x):
            return tf.greater(x, 100)
        x = tf.constant(3)
        r = tf.while_loop(condition, body, loop_vars=[x])

        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_vanilla_cond():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(1)
        j = tf.constant(4)

        def f1():
            return tf.multiply(1, 17)

        def f2():
            return tf.add(4, 23)
        r = tf.cond(tf.less(i, j), f1, f2)

    with tf.Session(graph=graph) as sess:
        tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_multiple_cond_vars():
    graph = tf.Graph()
    with graph.as_default():
        x1 = tf.constant(7)
        x2 = tf.constant(12)
        z = tf.constant(20)
        r = tf.cond(tf.less(tf.add(x1, x2), 10),
                    lambda: tf.add(10, 2), lambda: tf.square(5))

        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_cond_fn_parameters():
    graph = tf.Graph()
    with graph.as_default():
        def fn1(x, y):
            return tf.multiply(5, 6)

        def fn2(x, y):
            return tf.add(3, 4)

        i = tf.constant(1)
        j = tf.constant(2)
        k = tf.constant(3)
        r = tf.cond(tf.less(i, j), lambda: fn1(i, k), lambda: fn2(j, k))

        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={i: 1, j: 2, k: 3})

    check_equal(graph, tf_out)


def test_nested_cond():
    graph = tf.Graph()
    with graph.as_default():
        def fn1(a, b):
            def nest_fn1():
                return tf.add(1, 2)

            def nest_fn2():
                return tf.subtract(10, 5)

            res = tf.cond(tf.less(1, 2), nest_fn1, nest_fn2)
            return tf.multiply(tf.add(87, res), 10)

        def fn2(a, b):
            return tf.add(10, 10)

        x = tf.constant(5)
        y = tf.constant(6)
        z = tf.constant(7)
        pred = tf.less(x, y)
        r = tf.cond(pred, lambda: fn1(x, y), lambda: fn2(y, z))

        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={x: 1, y: 2, z: 3, pred: True})

    check_equal(graph, tf_out)


def test_loop_in_cond():
    graph = tf.Graph()
    with graph.as_default():
        def fn1(a, b):
            i = tf.constant(0)

            def cd(i): return tf.less(i, 10)

            def bd(i): return tf.add(i, 1)
            res = tf.while_loop(cd, bd, [i])
            return tf.multiply(tf.add(20, res), 10)

        def fn2(a, b):
            return tf.add(10, 20)

        x = tf.constant(7)
        y = tf.constant(20)
        z = tf.constant(10)
        pred = tf.less(x, y)
        r = tf.cond(pred, lambda: fn1(x, y), lambda: fn2(y, z))

        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={x: 1, y: 2, z: 3, pred: True})

    check_equal(graph, tf_out)


def test_cond_in_loop():
    graph = tf.Graph()
    with graph.as_default():
        def body(x):
            x = tf.constant(7)
            z = tf.constant(20)
            res = tf.cond(tf.less(x, 10), lambda: tf.add(
                10, 20), lambda: tf.square(10))
            return tf.multiply(res, x)

        x = tf.constant(21)
        def condition(x):
            return tf.less(x, 100)

        r = tf.while_loop(condition, body, loop_vars=[x])
        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


if __name__ == "__main__":

    # tf.while_loop
    test_vanilla_loop()
    test_loop_2_vars()
    test_loop_3_vars()
    test_loop_conditions()
    # TODO(@jroesch): Need to fix memory alloc to support closure
    # test_loop_bodies()
    test_callnode_loop_vars()

    # tf.cond
    test_vanilla_cond()
    test_multiple_cond_vars()
    test_cond_fn_parameters()

    # nested cases
    test_nested_loop()
    test_nested_cond()
    test_loop_in_cond()
    test_cond_in_loop()
