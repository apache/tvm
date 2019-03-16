"""Unit tests for converting TensorFlow control flow op to Relay."""
import tensorflow as tf
import numpy as np
from tvm import relay
from tvm.relay.frontend.tensorflow import from_tensorflow


def check_equal(graph, tf_out):
    expr, params = from_tensorflow(graph.as_graph_def(add_shapes=True))
    ex = relay.create_executor('debug')
    relay_out = ex.evaluate(expr)(**params)
    if isinstance(relay_out, relay.backend.interpreter.TensorValue):
        np.testing.assert_allclose(tf_out, relay_out.asnumpy())
    else:
        if not isinstance(tf_out, list):
            tf_out = [tf_out]
        for x, y in zip(tf_out, [r.asnumpy() for r in relay_out]):
            np.testing.assert_allclose(x, y)


def vanilla_loop():
    graph = tf.Graph()
    with graph.as_default():
        i = tf.constant(0)

        def c(i): return tf.less(i, 10)

        def b(i): return tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

        with tf.Session() as sess:
            tf_out = sess.run(r)

        check_equal(graph, tf_out)


def loop_2_vars():
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


def loop_3_vars():
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


def loop_conditions():
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


def loop_bodies():
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


def nested_loop():
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


def vanilla_cond():
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


def multiple_cond_vars():
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


def cond_fn_parameters():
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


def nested_cond():
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


def loop_in_cond():
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


def cond_in_loop():
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


def loop_lambda_placeholder():
    graph = tf.Graph()
    with graph.as_default():
        c = lambda i, j: tf.equal(tf.less(i, 17), tf.greater(j, 7))
        b = lambda i, j: [i + 3, j - 13]

        i = tf.placeholder(tf.float32)
        j = tf.placeholder(tf.float32)
        r = tf.while_loop(c, b, loop_vars=[i, j])

        with tf.Session() as sess:
            tf_out = sess.run(r, feed_dict={i: -203, j: 107})

    check_equal(graph, tf_out)


if __name__ == "__main__":

    # tf.while_loop
    vanilla_loop()
    loop_2_vars()
    loop_3_vars()
    loop_conditions()
    loop_bodies()

    # tf.cond
    vanilla_cond()
    multiple_cond_vars()
    cond_fn_parameters()

    # nested cases
    nested_loop()
    nested_cond()
    loop_in_cond()
    cond_in_loop()

    # w/ placeholder and lambda
    loop_lambda_placeholder()
