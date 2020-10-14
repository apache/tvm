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
from tvm import relay
from tvm.relay import transform


def run_combine_parallel(expr, min_num_branches=3):
    mod = tvm.IRModule.from_expr(expr)
    mod = transform.CombineParallelConv2D(min_num_branches)(mod)
    return mod["main"]


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    return mod["main"]


def test_combine_parallel_conv2d():
    """Simple testcase."""

    def before(x, w1, w2, w3, w4):
        args = [x, w1, w2, w3, w4]
        y1 = relay.nn.conv2d(x, w1)
        y2 = relay.nn.conv2d(x, w2)
        # y3 cannot be combined
        y3 = relay.nn.conv2d(x, w3)
        y4 = relay.nn.conv2d(x, w4)
        y5 = relay.nn.max_pool2d(x)
        y = relay.Tuple((y1, y2, y3, y4, y5))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1, w2, w3, w4]
        w = relay.concatenate((w1, w2, w4), axis=0)
        y = relay.nn.conv2d(x, w, channels=channels1 + channels2 + channels4)
        y1 = relay.strided_slice(
            y, begin=[0, 0], end=[-1, channels1], strides=[1, 1], slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, channels1], end=[-1, channels2], strides=[1, 1], slice_mode="size"
        )
        y3 = relay.nn.conv2d(x, w3)
        y4 = relay.strided_slice(
            y,
            begin=[0, channels1 + channels2],
            end=[-1, channels4],
            strides=[1, 1],
            slice_mode="size",
        )
        y5 = relay.nn.max_pool2d(x)
        y = relay.Tuple((y1, y2, y3, y4, y5))
        return relay.Function(args, y)

    def check(x_shape, channels1, channels2, channels3, channels4):
        x = relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
        w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
        w3 = relay.var("w3", shape=(channels3, in_c, 3, 3))
        w4 = relay.var("w4", shape=(channels4, in_c, 1, 1))

        y_before = before(x, w1, w2, w3, w4)
        y = run_opt_pass(y_before, transform.CombineParallelConv2D(min_num_branches=2))
        y_expected = expected(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

    check((1, 4, 16, 16), 4, 4, 4, 4)
    check((1, 4, 16, 16), 4, 8, 4, 7)


def test_combine_parallel_conv2d_scale_relu():
    """Testcase of combining conv2d + scale + relu"""

    def before(x, w1, w2, scale1, scale2, bias):
        args = [x, w1, w2, scale1, scale2, bias]
        y1 = relay.nn.conv2d(x, w1)
        y1 = relay.multiply(y1, scale1)
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(x, w2)
        y2 = relay.multiply(y2, scale2)
        y2 = relay.nn.relu(y2)
        y2 = relay.add(y2, bias)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, scale1, scale2, bias, channels1, channels2):
        args = [x, w1, w2, scale1, scale2, bias]
        w = relay.concatenate((w1, w2), axis=0)
        scale = relay.concatenate((scale1, scale2), axis=0)
        y = relay.nn.conv2d(x, w, channels=channels1 + channels2)
        y = relay.multiply(y, scale)
        y = relay.nn.relu(y)
        y1 = relay.strided_slice(
            y, begin=[0, 0], end=[-1, channels1], strides=[1, 1], slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, channels1], end=[-1, channels2], strides=[1, 1], slice_mode="size"
        )
        y2 = relay.add(y2, bias)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(x_shape, channels1, channels2):
        x = relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
        w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
        scale1 = relay.var("scale1", shape=(channels1, 1, 1))
        scale2 = relay.var("scale2", shape=(channels2, 1, 1))
        bias = relay.var("bias", shape=(channels2, 1, 1))
        y_before = before(x, w1, w2, scale1, scale2, bias)
        y = run_opt_pass(y_before, transform.CombineParallelConv2D(min_num_branches=2))
        y_expected = expected(x, w1, w2, scale1, scale2, bias, channels1, channels2)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

    check((1, 4, 16, 16), 4, 8)


def test_combine_parallel_conv2d_scale():
    """Testcase of un-combinable scale"""

    def before(x, w1, w2, scale1, scale2):
        args = [x, w1, w2, scale1, scale2]
        y1 = relay.nn.conv2d(x, w1)
        y1 = relay.multiply(y1, scale1)
        y2 = relay.nn.conv2d(x, w2)
        y2 = relay.multiply(y2, scale2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def expected(x, w1, w2, scale1, scale2, channels1, channels2):
        args = [x, w1, w2, scale1, scale2]
        w = relay.concatenate((w1, w2), axis=0)
        y = relay.nn.conv2d(x, w, channels=channels1 + channels2)
        y1 = relay.strided_slice(
            y, begin=[0, 0], end=[-1, channels1], strides=[1, 1], slice_mode="size"
        )
        y2 = relay.strided_slice(
            y, begin=[0, channels1], end=[-1, channels2], strides=[1, 1], slice_mode="size"
        )
        y1 = relay.multiply(y1, scale1)
        y2 = relay.multiply(y2, scale2)
        y = relay.Tuple((y1, y2))
        return relay.Function(args, y)

    def check(x_shape, channels1, channels2):
        x = relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
        w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
        scale1 = relay.var("scale1", shape=(1,))
        scale2 = relay.var("scale2", shape=(1,))
        y_before = before(x, w1, w2, scale1, scale2)
        y = run_opt_pass(y_before, transform.CombineParallelConv2D(min_num_branches=2))
        y_expected = expected(x, w1, w2, scale1, scale2, channels1, channels2)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

    check((1, 4, 16, 16), 4, 8)


def test_combine_parallel_conv2d_multiple_blocks():
    def before(x, w, repeat):
        args = [x, w]
        y = x
        for i in range(repeat):
            y1 = relay.nn.conv2d(y, w)
            y2 = relay.nn.conv2d(y, w)
            y = relay.concatenate((y1, y2), axis=1)
        return relay.Function(args, y)

    def expected(x, w, channels, repeat):
        args = [x, w]
        y = x
        for i in range(repeat):
            w_concat = relay.concatenate((w, w), axis=0)
            y = relay.nn.conv2d(y, w_concat, channels=channels * 2)
            y1 = relay.strided_slice(
                y, begin=[0, 0], end=[-1, channels], strides=[1, 1], slice_mode="size"
            )
            y2 = relay.strided_slice(
                y, begin=[0, channels], end=[-1, channels], strides=[1, 1], slice_mode="size"
            )
            y = relay.concatenate((y1, y2), axis=1)
        return relay.Function(args, y)

    def check(x_shape, repeat):
        x = relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        out_c = in_c // 2
        w = relay.var("w", shape=(out_c, in_c, 1, 1))
        y_before = before(x, w, repeat)
        y = run_opt_pass(y_before, transform.CombineParallelConv2D(min_num_branches=2))
        y_expected = expected(x, w, out_c, repeat)
        y_expected = run_opt_pass(y_expected, transform.InferType())
        assert tvm.ir.structural_equal(y, y_expected, map_free_vars=True)

    check((1, 4, 16, 16), 4)


if __name__ == "__main__":
    test_combine_parallel_conv2d()
    test_combine_parallel_conv2d_scale_relu()
    test_combine_parallel_conv2d_scale()
    test_combine_parallel_conv2d_multiple_blocks()
