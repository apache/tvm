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
# pylint: disable=invalid-name,too-many-locals,too-many-arguments,missing-module-docstring

import tvm
from tvm import relay
from tvm.relay import transform


def run_opt_pass(expr, opt_pass):
    "runs the opt_pass on the expr of a function the function"
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    return mod["main"]


def test_combine_parallel_batch_matmul():
    """Simple testcase."""

    def before(x, w1, w2, w3):
        args = [x, w1, w2, w3]
        y1 = relay.nn.batch_matmul(x, w1)
        y2 = relay.nn.batch_matmul(x, w2)
        y3 = relay.nn.batch_matmul(x, w3)
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3):
        # use a fixed order of args so alpha equal check can pass
        s1 = w1.type_annotation.shape[1]
        s2 = w2.type_annotation.shape[1]
        s3 = w3.type_annotation.shape[1]
        args = [x, w1, w2, w3]
        w = relay.concatenate((w1, w2, w3), axis=1)
        y = relay.nn.batch_matmul(x, w)
        y1 = relay.strided_slice(
            y, begin=[0, 0, 0], end=[-1, -1, s1], strides=[1, 1, 1], slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, 0, s1], end=[-1, -1, s2], strides=[1, 1, 1], slice_mode="size"
        )
        y3 = relay.strided_slice(
            y, begin=[0, 0, s1 + s2], end=[-1, -1, s3], strides=[1, 1, 1], slice_mode="size"
        )
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def check(b, i, j, k):
        x = relay.var("x", shape=(b, i, k))
        w1 = relay.var("w1", shape=(b, j, k))
        w2 = relay.var("w2", shape=(b, j, k))
        w3 = relay.var("w3", shape=(b, j, k))

        y_before = before(x, w1, w2, w3)
        y = run_opt_pass(y_before, transform.CombineParallelBatchMatmul(min_num_branches=2))
        y_expected = expected(x, w1, w2, w3)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(2, 3, 5, 4)
    check(1, 100, 200, 300)


def test_combine_parallel_batch_matmul_biasadd():
    """Simple testcase with bias"""

    def before(x, w1, w2, w3, b1, b2, b3):
        args = [x, w1, w2, w3, b1, b2, b3]
        y1 = relay.nn.batch_matmul(x, w1)
        y2 = relay.nn.batch_matmul(x, w2)
        y3 = relay.nn.batch_matmul(x, w3)
        y1 = relay.add(y1, b1)
        y2 = relay.add(y2, b2)
        y3 = relay.add(y3, b3)
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3, b1, b2, b3):
        # use a fixed order of args so alpha equal check can pass
        s1 = w1.type_annotation.shape[1]
        s2 = w2.type_annotation.shape[1]
        s3 = w3.type_annotation.shape[1]
        args = [x, w1, w2, w3, b1, b2, b3]
        w = relay.concatenate((w1, w2, w3), axis=1)
        b = relay.concatenate((b1, b2, b3), axis=-1)
        y = relay.nn.batch_matmul(x, w)
        y = relay.add(y, b)
        y1 = relay.strided_slice(
            y, begin=[0, 0, 0], end=[-1, -1, s1], strides=[1, 1, 1], slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, 0, s1], end=[-1, -1, s2], strides=[1, 1, 1], slice_mode="size"
        )
        y3 = relay.strided_slice(
            y, begin=[0, 0, s1 + s2], end=[-1, -1, s3], strides=[1, 1, 1], slice_mode="size"
        )
        y = relay.Tuple((y1, y2, y3))
        return relay.Function(args, y)

    def check(b, i, j, k):
        x = relay.var("x", shape=(b, i, k))
        w1 = relay.var("w1", shape=(b, j, k))
        w2 = relay.var("w2", shape=(b, j, k))
        w3 = relay.var("w3", shape=(b, j, k))
        b1 = relay.var("b1", shape=(j,))
        b2 = relay.var("b2", shape=(j,))
        b3 = relay.var("b3", shape=(j,))

        y_before = before(x, w1, w2, w3, b1, b2, b3)
        y = run_opt_pass(y_before, transform.CombineParallelBatchMatmul(min_num_branches=2))
        y_expected = expected(x, w1, w2, w3, b1, b2, b3)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        tvm.ir.assert_structural_equal(y, y_expected, map_free_vars=True)

    check(2, 3, 5, 4)
    check(1, 100, 200, 300)


if __name__ == "__main__":
    test_combine_parallel_batch_matmul()
    test_combine_parallel_batch_matmul_biasadd()
