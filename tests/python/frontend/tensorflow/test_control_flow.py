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

try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
from tvm import nd, relay, ir, testing
from tvm.relay.frontend.tensorflow import from_tensorflow


def check_equal(graph, tf_out, input_map=None):
    with testing.disable_span_filling():
        mod, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    with testing.enable_span_filling():
        mod_with_span, _ = from_tensorflow(graph.as_graph_def(add_shapes=True))
    assert ir.structural_equal(mod["main"], mod_with_span["main"])

    if input_map is not None:
        params.update(input_map)
    relay_out = relay.create_executor("vm", mod=mod).evaluate()(**params)
    if isinstance(relay_out, nd.NDArray):
        np.testing.assert_allclose(tf_out, relay_out.numpy())
    else:
        if not isinstance(tf_out, (list, tuple)):
            tf_out = [tf_out]
        for x, y in zip(tf_out, [r.numpy() for r in relay_out]):
            np.testing.assert_allclose(x, y)


def test_vanilla_loop():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(0, name="while/constant")

        def c(i):
            return tf.less(i, 10)

        def b(i):
            return tf.add(i, 1)

        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def test_callnode_loop_vars():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.add(tf.constant(0), 1)

        def c(i):
            return tf.less(i, 10)

        def b(i):
            return tf.add(i, 1)

        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def test_loop_2_vars():
    graph = tf.Graph()
    with graph.as_default():
        i0 = tf.constant(0)
        j0 = tf.ones([2, 2])

        def c(i, j):
            return i < 10

        def b(i, j):
            return [tf.add(i, 1), j]

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

        def c(i, j, k):
            return i < 10

        def b(i, j, k):
            return [i + 1, j * k, k + i]

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

        def c(i, j, k):
            return tf.equal(
                tf.not_equal(tf.less(i + j, 10), tf.less(j * k, 100)), tf.greater_equal(k, i + j)
            )

        def b(i, j, k):
            return [i + j, j + k, k + 1]

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

            def cd(c):
                return tf.less(c, 10)

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
        r = tf.cond(tf.less(tf.add(x1, x2), 10), lambda: tf.add(10, 2), lambda: tf.square(5))

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

            def cd(i):
                return tf.less(i, 10)

            def bd(i):
                return tf.add(i, 1)

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
            res = tf.cond(tf.less(x, 10), lambda: tf.add(10, 20), lambda: tf.square(10))
            return tf.multiply(res, x)

        x = tf.constant(21)

        def condition(x):
            return tf.less(x, 100)

        r = tf.while_loop(condition, body, loop_vars=[x])
        with tf.Session() as sess:
            tf_out = sess.run(r)

    check_equal(graph, tf_out)


def test_vanilla_loop_bound():
    graph = tf.Graph()
    with graph.as_default():
        dshape = (2, 10)
        dtype = "float32"
        dname = "data"
        np_data = np.random.uniform(size=dshape).astype(dtype)
        data = tf.placeholder(shape=dshape, dtype=dtype, name=dname)
        x = tf.slice(data, [1, 4], [1, 4])
        outer = x + 5.0

        def body(x, y):
            res = tf.cond(tf.less(y, 10), lambda: tf.add(10.0, 20.0), lambda: tf.square(10.0))
            z = tf.constant(7)
            res = tf.cond(tf.less(z, 10), lambda: res * 5, lambda: res + 10)
            return tf.multiply(res, x * outer), y + 1

        y = tf.constant(0)

        def condition(x, y):
            return tf.less(y, 20)

        r = tf.while_loop(condition, body, loop_vars=[x, y])
        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={"%s:0" % dname: np_data})

    check_equal(graph, tf_out, {dname: np_data})


def test_nested_loop_bound():
    graph = tf.Graph()
    with graph.as_default():
        dshape = (2, 10)
        dtype = "float32"
        dname = "data"
        np_data = np.random.uniform(size=dshape).astype(dtype)
        data = tf.placeholder(shape=dshape, dtype=dtype, name=dname)
        x = tf.slice(data, [1, 4], [1, 4])
        outer = x + 5.0

        def body(x, y):
            res = tf.cond(tf.less(y, 10), lambda: tf.add(10.0, 20.0), lambda: tf.square(10.0))

            def nested_body(nx, ny):
                return nx + 1, res + 2.0

            def nested_cond(nx, ny):
                return tf.less(nx, 15)

            nx = tf.constant(0)
            ny = tf.constant(0.0)
            nested_res = tf.while_loop(nested_cond, nested_body, loop_vars=[nx, ny])
            res = res + nested_res[1]
            z = tf.constant(7)
            res = tf.cond(tf.less(z, 10), lambda: res * 5, lambda: res + 10)
            return tf.multiply(res, x * outer), y + 1

        y = tf.constant(0)

        def condition(x, y):
            return tf.less(y, 20)

        r = tf.while_loop(condition, body, loop_vars=[x, y])
        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={"%s:0" % dname: np_data})

    check_equal(graph, tf_out, {dname: np_data})


def test_switch():
    graph = tf.Graph()

    with graph.as_default():
        data_np = np.random.uniform(0, 5, size=(2, 4, 5, 1)).astype("float32")
        dname = "data"
        flag_name = "flag"
        data = tf.placeholder(shape=data_np.shape, dtype=data_np.dtype, name=dname)
        split = tf.split(data, 2, axis=0)
        flag = tf.placeholder(shape={}, dtype=tf.bool, name=flag_name)
        output_false, output_true = control_flow_ops.switch(split[1], flag)
        with tf.Session() as sess:
            tf_out = sess.run(output_false, feed_dict={data.name: data_np, flag.name: False})

    check_equal(graph, tf_out, {dname: data_np, flag_name: False})


def test_loop_tuple_input():
    graph = tf.Graph()

    with graph.as_default():
        data_np = np.random.uniform(0, 5, size=(2, 4, 5, 1)).astype("float32")
        dname = "data"
        data = tf.placeholder(shape=data_np.shape, dtype=data_np.dtype, name=dname)
        split = tf.split(data, 2, axis=0)

        def body(x, y):
            return x + 2, y + 1

        start = tf.constant(0)

        def condition(x, y):
            return tf.less(y, 20)

        r = tf.while_loop(condition, body, loop_vars=[split[1], start])
        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={data.name: data_np})

    check_equal(graph, tf_out, {dname: data_np})


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
    test_vanilla_loop_bound()
    test_nested_loop_bound()

    test_switch()
    test_loop_tuple_input()
