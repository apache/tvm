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
from tvm import te
from tvm import relay
from tvm.relay import transform


def run_combine_parallel(expr, min_num_branches=3):
    mod = tvm.IRModule.from_expr(expr)
    mod = transform.CombineParallelDense(min_num_branches)(mod)
    return mod["main"]

def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
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
        x =  relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))
        w3 = relay.var("w3", shape=(j + 1, k))
        w4 = relay.var("w4", shape=(j, k))

        y_before = before(x, w1, w2, w3, w4)
        y = run_opt_pass(y_before,
                         transform.CombineParallelDense(min_num_branches=2))
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
        x =  relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))

        if is_2d_bias:
            b1 = relay.var("b1", shape=(i, j))
            b2 = relay.var("b2", shape=(i, j))
        else:
            b1 = relay.var("b1", shape=(j,))
            b2 = relay.var("b2", shape=(j,))

        y_before = before(x, w1, w2, b1, b2)
        y = run_opt_pass(y_before,
                         transform.CombineParallelDense(min_num_branches=2))
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
        x =  relay.var("x", shape=(i, k))
        w1 = relay.var("w1", shape=(j, k))
        w2 = relay.var("w2", shape=(j, k))
        b1 = relay.var("b1", shape=(j,))
        b2 = relay.var("b2", shape=(j,))
        scale1 = relay.var("scale1", shape=(1,))
        scale2 = relay.var("scale2", shape=(1,))

        y_before = before(x, w1, w2, b1, b2, scale1, scale2, newshape)
        y = run_opt_pass(y_before,
                         transform.CombineParallelDense(min_num_branches=2))
        y_expected = expected(x, w1, w2, b1, b2, scale1, scale2, newshape)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(3, 5, 4, 0.5, 0.25, (1, 1, 15))
    check(100, 200, 300, 0.5, 0.25, (1, 1, 200))


if __name__ == "__main__":
    test_combine_parallel_dense()
    test_combine_parallel_dense_biasadd()
    test_combine_parallel_dense_biasadd_scale_reshape()
