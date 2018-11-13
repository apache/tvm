from tvm import relay
import numpy as np


def test_combine_parallel_conv2d():
    """Simple testcase."""
    def before(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4):
        args = [x, w1, w2, w3, w4]
        y1 = relay.nn.conv2d(x, w1,
                            channels=channels1,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y2 = relay.nn.conv2d(x, w2,
                            channels=channels2,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        # y3 is not foldable
        y3 = relay.nn.conv2d(x, w3,
                             channels=channels3,
                             kernel_size=(1, 1),
                             padding=(1, 1))
        y4 = relay.nn.conv2d(x, w4,
                             channels=channels4,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y = relay.Tuple((y1, y2, y3, y4))
        return relay.Function(args, y)

    def expected(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4):
        # use a fixed order of args so alpha equal check can pass
        args = [x, w1, w2, w3, w4]
        w = relay.concatenate((w1, w2, w4), axis=0)
        y = relay.nn.conv2d(x, w,
                            channels=channels1 + channels2 + channels4,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.take(y, relay.const(np.arange(channels1, dtype='int64')), axis=1)
        y2 = relay.take(y, relay.const(np.arange(channels1, channels1 + channels2, dtype='int64')), axis=1)
        y3 = relay.nn.conv2d(x, w3,
                             channels=channels3,
                             kernel_size=(1, 1),
                             padding=(1, 1))
        y4 = relay.take(y, relay.const(np.arange(channels1 + channels2,
                                                 channels1 + channels2 + channels4, dtype='int64')), axis=1)
        y = relay.Tuple((y1, y2, y3, y4))
        return relay.Function(args, y)

    def check(channels1, channels2, channels3, channels4):
        x =  relay.var("x")
        w1 = relay.var("w1")
        w2 = relay.var("w2")
        w3 = relay.var("w3")
        w4 = relay.var("w4")

        y_before = before(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4)
        y = relay.ir_pass.combine_parallel_conv2d(y_before)
        y_expected = expected(x, w1, w2, w3, w4, channels1, channels2, channels3, channels4)
        assert relay.ir_pass.alpha_equal(y, y_expected)

    check(4, 4, 4, 4)
    check(4, 8, 4, 7)


if __name__ == "__main__":
    test_combine_parallel_conv2d()
