from tvm import relay
import numpy as np


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
        y1 = relay.strided_slice(y, [0, 0], [None, channels1])
        y2 = relay.strided_slice(y, [0, channels1], [None, channels1 + channels2])
        y3 = relay.nn.conv2d(x, w3)
        y4 = relay.strided_slice(y, [0, channels1 + channels2],
                                 [None, channels1 + channels2 + channels4])
        y5 = relay.nn.max_pool2d(x)
        y = relay.Tuple((y1, y2, y3, y4, y5))
        return relay.Function(args, y)

    def check(x_shape, channels1, channels2, channels3, channels4):
        x =  relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
        w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
        w3 = relay.var("w3", shape=(channels3, in_c, 3, 3))
        w4 = relay.var("w4", shape=(channels4, in_c, 1, 1))

        y_before = before(x, w1, w2, w3, w4)
        y = relay.ir_pass.infer_type(y_before)
        y = relay.ir_pass.combine_parallel_conv2d(y)
        y = relay.ir_pass.infer_type(y)
        y_expected = expected(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4)
        y_expected = relay.ir_pass.infer_type(y_expected)
        assert relay.ir_pass.alpha_equal(y, y_expected)

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
        y1 = relay.strided_slice(y, [0, 0], [None, channels1])
        y2 = relay.strided_slice(y, [0, channels1], [None, channels1 + channels2])
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
        y = relay.ir_pass.infer_type(y_before)
        y = relay.ir_pass.combine_parallel_conv2d(y)
        y = relay.ir_pass.infer_type(y)
        y_expected = expected(x, w1, w2, scale1, scale2, bias, channels1, channels2)
        y_expected = relay.ir_pass.infer_type(y_expected)
        assert relay.ir_pass.alpha_equal(y, y_expected)

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
        y1 = relay.strided_slice(y, [0, 0], [None, channels1])
        y2 = relay.strided_slice(y, [0, channels1], [None, channels1 + channels2])
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
        y = relay.ir_pass.infer_type(y_before)
        y = relay.ir_pass.combine_parallel_conv2d(y)
        y = relay.ir_pass.infer_type(y)
        y_expected = expected(x, w1, w2, scale1, scale2, channels1, channels2)
        y_expected = relay.ir_pass.infer_type(y_expected)
        assert relay.ir_pass.alpha_equal(y, y_expected)

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
            y = relay.nn.conv2d(y, w_concat, channels=channels*2)
            y1 = relay.strided_slice(y, [0, 0], [None, channels])
            y2 = relay.strided_slice(y, [0, channels], [None, channels * 2])
            y = relay.concatenate((y1, y2), axis=1)
        return relay.Function(args, y)

    def check(x_shape, repeat):
        x = relay.var("x", shape=x_shape)
        in_c = x_shape[1]
        out_c = in_c // 2
        w = relay.var("w", shape=(out_c, in_c, 1, 1))
        y_before = before(x, w, repeat)
        y = relay.ir_pass.infer_type(y_before)
        y = relay.ir_pass.combine_parallel_conv2d(y)
        y = relay.ir_pass.infer_type(y)
        y_expected = expected(x, w, out_c, repeat)
        y_expected = relay.ir_pass.infer_type(y_expected)
        assert relay.ir_pass.alpha_equal(y, y_expected)

    check((1, 4, 16, 16), 4)


if __name__ == "__main__":
    test_combine_parallel_conv2d()
    test_combine_parallel_conv2d_scale_relu()
    test_combine_parallel_conv2d_scale()
    test_combine_parallel_conv2d_multiple_blocks()
