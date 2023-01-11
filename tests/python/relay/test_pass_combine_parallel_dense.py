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
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform


def run_combine_parallel(expr, min_num_branches=3, to_batch=True):
    mod = tvm.IRModule.from_expr(expr)
    mod = transform.CombineParallelDense(min_num_branches, to_batch)(mod)
    return mod["main"]


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    return mod["main"]


def test_combine_parallel_dense():
    """Simple testcase. One dense cannot be combined due to shape mismatch"""

    def before(x, w1, w2, w3, w4):
        args = [x, w1, w2, w3, w4]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)

        # y3 cannot be combined
        y3 = relay.nn.dense(x, w3)

        y4 = relay.nn.dense(x, w4)
        y = relay.Tuple((y1, y2, y3, y4))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3, w4):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1, w2, w3, w4]
        x_stacked = relay.stack((x, x, x), axis=0)
        w = relay.stack((w1, w2, w4), axis=0)
        y = relay.nn.batch_matmul(x_stacked, w)
        (y1, y2, y4) = relay.split(y, 3)
        y1 = relay.squeeze(y1, [0])
        y2 = relay.squeeze(y2, [0])
        y4 = relay.squeeze(y4, [0])

        # y3 cannot be combined
        y3 = relay.nn.dense(x, w3)

        y = relay.Tuple((y1, y2, y3, y4))
        return relay.Function(args, y)

    def check(i, j, k):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))
        w3 = relay.var("w3", shape=(j + 1, k))
        w4 = relay.var("w4", shape=(j, k))

        y_before = before(x, w1, w2, w3, w4)
        y = run_opt_pass(y_before, transform.CombineParallelDense(min_num_branches=2))
        y_expected = expected(x, w1, w2, w3, w4)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4)
    check(100, 200, 300)


def test_combine_parallel_dense_biasadd():
    """Testcase of combining dense + 1d biasadd"""

    def before(x, w1, w2, b1, b2):
        args = [x, w1, w2, b1, b2]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, b1, b2, is_2d_bias):
        args = [x, w1, w2, b1, b2]
        x_stacked = relay.stack((x, x), axis=0)
        w = relay.stack((w1, w2), axis=0)
        y = relay.nn.batch_matmul(x_stacked, w)

        if not is_2d_bias:
            b1 = relay.expand_dims(b1, 0)
            b2 = relay.expand_dims(b2, 0)

        b = relay.stack((b1, b2), axis=0)
        y = relay.add(y, b)
        (y1, y2) = relay.split(y, 2)
        y1 = relay.squeeze(y1, [0])
        y2 = relay.squeeze(y2, [0])
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(i, j, k, is_2d_bias):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))

        if is_2d_bias:
            b1 = relay.var("b1", shape=(i, j))
            b2 = relay.var("b2", shape=(i, j))
        else:
            b1 = relay.var("b1", shape=(j,))
            b2 = relay.var("b2", shape=(j,))

        y_before = before(x, w1, w2, b1, b2)
        y = run_opt_pass(y_before, transform.CombineParallelDense(min_num_branches=2))
        y_expected = expected(x, w1, w2, b1, b2, is_2d_bias)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4, False)
    check(100, 200, 300, False)
    check(3, 5, 4, True)
    check(100, 200, 300, True)


def test_combine_parallel_dense_biasadd_scale_reshape():
    """Testcase of combining dense + 1d biasadd + multiply with non-fused reshape"""

    def before(x, w1, w2, b1, b2, scale1, scale2, newshape):
        args = [x, w1, w2, b1, b2, scale1, scale2]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y1 = relay.multiply(y1, scale1)
        y2 = relay.multiply(y2, scale2)
        y1 = relay.reshape(y1, newshape=newshape)
        y2 = relay.reshape(y2, newshape=newshape)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, b1, b2, scale1, scale2, newshape):
        args = [x, w1, w2, b1, b2, scale1, scale2]
        x_stacked = relay.stack((x, x), axis=0)
        w = relay.stack((w1, w2), axis=0)
        y = relay.nn.batch_matmul(x_stacked, w)
        b1 = relay.expand_dims(b1, 0)
        b2 = relay.expand_dims(b2, 0)
        b = relay.stack((b1, b2), axis=0)
        y = relay.add(y, b)
        scale1 = relay.expand_dims(scale1, 0)
        scale2 = relay.expand_dims(scale2, 0)
        scale = relay.stack((scale1, scale2), axis=0)
        y = relay.multiply(y, scale)
        (y1, y2) = relay.split(y, 2)
        y1 = relay.squeeze(y1, [0])
        y2 = relay.squeeze(y2, [0])
        y1 = relay.reshape(y1, newshape=newshape)
        y2 = relay.reshape(y2, newshape=newshape)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(i, j, k, scale1, scale2, newshape):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))
        b1 = relay.var("b1", shape=(j,))
        b2 = relay.var("b2", shape=(j,))
        scale1 = relay.var("scale1", shape=(1,))
        scale2 = relay.var("scale2", shape=(1,))

        y_before = before(x, w1, w2, b1, b2, scale1, scale2, newshape)
        y = run_opt_pass(y_before, transform.CombineParallelDense(min_num_branches=2))
        y_expected = expected(x, w1, w2, b1, b2, scale1, scale2, newshape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4, 0.5, 0.25, (1, 1, 15))
    check(100, 200, 300, 0.5, 0.25, (1, 1, 20000))


def test_combine_parallel_dense_flat():
    """Simple testcase. All matmul of different output dim can be combined"""

    def before(x, w1, w2, w3):
        args = [x, w1, w2, w3]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y3 = relay.nn.dense(x, w3)
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3, j):
        args = [x, w1, w2, w3]
        w_stacked = relay.concatenate((w1, w2, w3), axis=0)
        y = relay.nn.dense(x, w_stacked, units=6 * j)
        strides = [1, 1]
        y1 = relay.strided_slice(y, begin=[0, 0], end=[-1, j], strides=strides, slice_mode="size")
        y2 = relay.strided_slice(
            y, begin=[0, j], end=[-1, 2 * j], strides=strides, slice_mode="size"
        )
        y3 = relay.strided_slice(
            y, begin=[0, 3 * j], end=[-1, 3 * j], strides=strides, slice_mode="size"
        )
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def check(i, j, k):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(2 * j, k))
        w3 = relay.var("w3", shape=(3 * j, k))

        y_before = before(x, w1, w2, w3)
        combine_pass = transform.CombineParallelDense(min_num_branches=3, to_batch=False)
        y = run_opt_pass(y_before, combine_pass)
        y_expected = expected(x, w1, w2, w3, j)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4)
    check(100, 200, 300)


def test_combine_parallel_dense_flat_biasadd():
    """Testcase of combining dense + 1d biasadd with different out dims"""

    def before(x, w1, w2, b1, b2):
        args = [x, w1, w2, b1, b2]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, b1, b2, j, bias_shape1, bias_shape2):
        args = [x, w1, w2, b1, b2]
        w_stacked = relay.concatenate((w1, w2), axis=0)
        y = relay.nn.dense(x, w_stacked, units=3 * j)
        n_out_dims = max(len(bias_shape1), 2)
        if len(bias_shape1) == 0:
            b1 = relay.repeat(relay.expand_dims(b1, -1), j, 0)
        elif bias_shape1[-1] == 1:
            b1 = relay.repeat(b1, j, len(bias_shape1) - 1)
        if len(bias_shape2) == 0:
            b2 = relay.repeat(relay.expand_dims(b2, -1), 2 * j, 0)
        elif bias_shape2[-1] == 1:
            b2 = relay.repeat(b2, 2 * j, len(bias_shape2) - 1)
        b = relay.concatenate((b1, b2), axis=max(0, len(bias_shape1) - 1))
        y = relay.add(y, b)
        begin = [0 for _ in range(n_out_dims - 1)]
        end = [-1 for _ in range(n_out_dims - 1)]
        strides = [1 for _ in range(n_out_dims)]
        y1 = relay.strided_slice(
            y, begin=begin + [0], end=end + [j], strides=strides, slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=begin + [j], end=end + [2 * j], strides=strides, slice_mode="size"
        )
        return relay.Function(args, relay.Tuple((y1, y2)))

    def check(i, j, k, bias_shape1, bias_shape2):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(2 * j, k))
        b1 = relay.var("b1", shape=bias_shape1)
        b2 = relay.var("b2", shape=bias_shape2)

        y_before = before(x, w1, w2, b1, b2)
        combine_pass = transform.CombineParallelDense(min_num_branches=2, to_batch=False)
        y = run_opt_pass(y_before, combine_pass)
        y_expected = expected(x, w1, w2, b1, b2, j, bias_shape1, bias_shape2)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4, (), ())
    check(3, 5, 4, (1,), (1,))
    check(3, 5, 4, (5,), (1,))
    check(3, 5, 4, (1,), (10,))
    check(3, 5, 4, (3, 1), (3, 1))
    check(3, 5, 4, (3, 5), (3, 10))
    check(3, 5, 4, (3, 1), (3, 10))
    check(3, 5, 4, (3, 5), (3, 1))
    check(3, 5, 4, (9, 3, 5), (9, 3, 10))
    check(3, 5, 4, (9, 3, 5), (9, 3, 1))
    check(3, 5, 4, (9, 3, 1), (9, 3, 10))


def test_combine_parallel_dense_flat_biasadd_scale_reshape():
    """Testcase of combining dense with different out dims
    following bias add, scale, reshape ops
    """

    def before(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2):
        args = [x, w1, w2, b1, b2, scale1, scale2]
        y1 = relay.nn.dense(x, w1)
        y2 = relay.nn.dense(x, w2)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y1 = relay.multiply(y1, scale1)
        y2 = relay.multiply(y2, scale2)
        y1 = relay.reshape(y1, newshape=newshape1)
        y2 = relay.reshape(y2, newshape=newshape2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2, j):
        args = [x, w1, w2, b1, b2, scale1, scale2]
        w_stacked = relay.concatenate((w1, w2), axis=0)
        y = relay.nn.dense(x, w_stacked, units=3 * j)
        b = relay.concatenate((b1, b2), axis=0)
        y = relay.add(y, b)
        scale1 = relay.repeat(scale1, j, 0)
        scale2 = relay.repeat(scale2, 2 * j, 0)
        scale = relay.concatenate((scale1, scale2), axis=0)
        y = relay.multiply(y, scale)
        strides = [1, 1]
        y1 = relay.strided_slice(y, begin=[0, 0], end=[-1, j], strides=strides, slice_mode="size")
        y2 = relay.strided_slice(
            y, begin=[0, j], end=[-1, 2 * j], strides=strides, slice_mode="size"
        )
        y1 = relay.reshape(y1, newshape=newshape1)
        y2 = relay.reshape(y2, newshape=newshape2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(i, j, k, scale1, scale2, newshape1, newshape2):
        x = relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(2 * j, k))
        b1 = relay.var("b1", shape=(j,))
        b2 = relay.var("b2", shape=(2 * j,))
        scale1 = relay.var("scale1", shape=(1,))
        scale2 = relay.var("scale2", shape=(1,))

        y_before = before(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2)
        combine_pass = transform.CombineParallelDense(min_num_branches=2, to_batch=False)
        y = run_opt_pass(y_before, combine_pass)
        y_expected = expected(x, w1, w2, b1, b2, scale1, scale2, newshape1, newshape2, j)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4, 0.5, 0.25, (1, 1, 15), (1, 1, 30))
    check(100, 200, 300, 0.5, 0.25, (1, 1, 20000), (1, 1, 40000))


def test_combine_parallel_dense_expand_dims():
    """Verify that the correct slice axis is selected after the combined dense."""

    def before(x, w1, w2):
        args = [x, w1, w2]
        y1 = relay.nn.dense(x, w1)
        y1 = relay.expand_dims(y1, axis=2)

        y2 = relay.nn.dense(x, w2)
        y2 = relay.expand_dims(y2, axis=2)

        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2):
        args = [x, w1, w2]
        w_stacked = relay.concatenate((w1, w2), axis=0)
        y = relay.nn.dense(x, w_stacked, units=24)
        y = relay.expand_dims(y, axis=2)

        strides = [1, 1, 1]
        y1 = relay.strided_slice(
            y, begin=[0, 0, 0], end=[-1, 16, -1], strides=strides, slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, 16, 0], end=[-1, 8, -1], strides=strides, slice_mode="size"
        )
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    x = relay.var("x", shape=(2, 32))
    w1 = relay.var("w1", shape=(16, 32))
    w2 = relay.var("w2", shape=(8, 32))

    y_before = before(x, w1, w2)
    combine_pass = transform.CombineParallelDense(min_num_branches=2, to_batch=False)
    y = run_opt_pass(y_before, combine_pass)
    y_expected = expected(x, w1, w2)
    y_expected = run_opt_pass(y_expected, transform.InferType())
    tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
