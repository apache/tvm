from tvm import relay
import numpy as np

def _get_positive_scale(size):
    return np.random.uniform(0.5, 1, size=size).astype('float32')


def test_fold_fwd_simple():
    """Simple testcase."""
    def before(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias]
        in_bias = relay.expand_dims(in_bias, axis=1, num_newaxis=2)
        x = relay.multiply(x, in_scale)
        x = relay.nn.relu(x)
        x = relay.add(x, in_bias)
        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))

        return relay.Function(args, y)

    def expected(x, conv_weight, in_bias, in_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, in_bias]
        in_bias = relay.expand_dims(in_bias, axis=1, num_newaxis=2)
        squeezed_scale = relay.squeeze(in_scale, axis=[1,2])
        x = relay.nn.relu(x)
        in_bias = relay.divide(in_bias, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2))
        x = relay.add(x, in_bias)
        conv_weight = relay.multiply(
            conv_weight , relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2))
        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(_get_positive_scale((in_channels, 1, 1)))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, in_bias, in_scale, channels)

        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 2)


def test_fold_fwd_dual_path():
    """scale axis being consumed by two consumers"""
    def before(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias]
        x = relay.multiply(in_scale, x)
        x = relay.nn.relu(x)
        x = relay.subtract(x, in_bias)
        y1 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             kernel_layout="HWIO",
                             groups=channels,
                             padding=(1, 1))
        y2 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             kernel_layout="HWIO",
                             groups=channels,
                             padding=(1, 1))
        z = relay.add(y1, y2)
        return relay.Function(args, z)

    def expected(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias]
        x = relay.nn.relu(x)
        in_bias = relay.divide(in_bias, in_scale)
        x = relay.subtract(x, in_bias)
        y1 = relay.nn.conv2d(x,
                             relay.multiply(conv_weight, in_scale),
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             kernel_layout="HWIO",
                             groups=channels,
                             padding=(1, 1))
        y2 = relay.nn.conv2d(x,
                             relay.multiply(conv_weight, in_scale),
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             kernel_layout="HWIO",
                             groups=channels,
                             padding=(1, 1))
        z = relay.add(y1, y2)
        return relay.Function(args, z)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[-1]
        # test depthwise
        assert in_channels == channels
        weight = relay.var("weight")
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(_get_positive_scale(in_channels,))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_expected = expected(x, weight, in_bias, in_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 3), 3)


def test_fold_fwd_fail():
    """testcase where we canont fold"""
    def before(x, conv_weight, in_bias, in_scale, channels):
        x = relay.multiply(x, in_scale)
        xx = relay.nn.leaky_relu(x, alpha=0.1)
        y1 = relay.nn.conv2d(xx, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             padding=(1, 1))
        z = relay.add(y1, x)
        return relay.Function(relay.ir_pass.free_vars(z), z)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[-1]
        # test depthwise
        assert in_channels == channels
        weight = relay.var("weight")
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(_get_positive_scale(size=(in_channels,)))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        assert relay.ir_pass.alpha_equal(y1, y1_folded)

    check((2, 11, 10, 4), 4)


def test_fold_fwd_relu_fail():
    """testcase where we canont fold because scale can not pass relu"""
    def before(x, conv_weight, in_bias, in_scale, channels):
        x = relay.multiply(x, in_scale)
        xx = relay.nn.relu(x)
        y1 = relay.nn.conv2d(xx, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             padding=(1, 1))
        z = relay.add(y1, x)
        return relay.Function(relay.ir_pass.free_vars(z), z)

    def check(shape, channels, in_scale):
        x =  relay.var("x", shape=shape)
        in_channels = shape[-1]
        # test depthwise
        assert in_channels == channels
        weight = relay.var("weight")
        in_bias = relay.var("in_bias", shape=(in_channels,))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        assert relay.ir_pass.alpha_equal(y1, y1_folded)

    in_scale = relay.var("in_scale", shape=(4,))
    check((2, 11, 10, 4), 4, in_scale)
    in_scale = relay.const(-_get_positive_scale((4,)))
    check((2, 11, 10, 4), 4, in_scale)


def test_fold_fwd_negative_scale():
    """Testcase of folding negative scale"""
    def before(x, conv_weight, in_scale, channels):
        args = [x, conv_weight]
        x = relay.multiply(x, in_scale)
        y = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        return relay.Function(args, y)

    def expected(x, conv_weight, in_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight]
        squeezed_scale = relay.squeeze(in_scale, axis=[1,2])
        conv_weight = relay.multiply(
            conv_weight , relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2))
        y = relay.nn.conv2d(x,
                             conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        in_scale = relay.const(-_get_positive_scale((in_channels, 1, 1)))
        weight = relay.var("weight")
        y1 = before(x, weight, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, in_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4)


def test_fold_bwd_simple():
    """Simple testcase."""
    def before(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        squeezed_scale = relay.squeeze(out_scale, axis=[1,2])
        conv_weight = relay.multiply(
            conv_weight , relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3))

        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        out_bias = relay.multiply(out_bias,
                                  relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2))
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

        y1 = before(x, weight, out_bias, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.backward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, out_bias, out_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 8)


def test_fold_bwd_dual_path():
    """Dual path testcase."""
    def before(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        y1 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        squeezed_scale = relay.squeeze(out_scale, axis=[1,2])
        def fold_conv_weight():
            return  relay.multiply(
                conv_weight ,
                relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3))
        y1 = relay.nn.conv2d(x, fold_conv_weight(),
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(x, fold_conv_weight(),
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

        y1 = before(x, weight, out_bias, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.backward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, out_bias, out_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 8)


def test_fold_bwd_dual_consumer():
    def before(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        y0 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y0 = relay.multiply(y0, out_scale)
        y0 = relay.nn.relu(y0)

        y1 = relay.nn.conv2d(y0, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.multiply(y1, out_scale)
        y1 = relay.nn.relu(y1)

        y2 = relay.nn.conv2d(y0, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y2 = relay.multiply(y2, out_scale)
        y2 = relay.nn.relu(y2)

        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        def fold_conv_weight():
            squeezed_scale = relay.squeeze(out_scale, axis=[1,2])
            return  relay.multiply(
                conv_weight ,
                relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3))
        y0 = relay.nn.conv2d(x, fold_conv_weight(),
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y0 = relay.nn.relu(y0)
        y1 = relay.nn.conv2d(y0, fold_conv_weight(),
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(y0, fold_conv_weight(),
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels,1, 1)))

        y1 = before(x, weight, out_bias, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.backward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, out_bias, out_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4)


def test_fold_bwd_fail():
    """Dual path testcase."""
    def fail1(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        y1 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1),
                             out_layout="CNHW")
        # fold will fail because the axis from two path
        # differs from each other.
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def fail2(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        y1 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             padding=(1, 1))
        y2 = relay.nn.relu(y1)
        # fold will fail because y1 is referred also by y2
        y1 = relay.multiply(y1, out_scale)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, channels, fbefore):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))
        y1 = fbefore(x, weight, out_bias, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.backward_fold_scale_axis(y1)
        assert relay.ir_pass.alpha_equal(y1_folded, y1)

    check((4, 4, 10, 10), 4, fail1)
    check((4, 4, 10, 10), 4, fail2)


def test_fold_bwd_relu_fail():
    """testcase where we canont fold because scale can not pass relu"""
    def before(x, conv_weight, out_scale, channels):
        y = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NCHW",
                             padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.multiply(x, out_scale)
        return relay.Function(relay.ir_pass.free_vars(y), y)

    def check(shape, channels, out_scale):
        x =  relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        y1 = before(x, weight, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        assert relay.ir_pass.alpha_equal(y1, y1_folded)

    out_scale = relay.var("in_scale", shape=(4, 1, 1))
    check((4, 4, 10, 10), 4, out_scale)
    out_scale = relay.const(np.random.uniform(size=(4, 1, 1), low=-1.0, high=0.0)).astype("float32")
    check((4, 4, 10, 10), 4, out_scale)


def test_fold_bwd_negative_scale():
    """Testcase of folding negative scale"""
    def before(x, conv_weight, out_scale, channels):
        args = [x, conv_weight]
        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight]
        squeezed_scale = relay.squeeze(out_scale, axis=[1,2])
        conv_weight = relay.multiply(
            conv_weight , relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3))
        y = relay.nn.conv2d(x, conv_weight,
                            channels=channels,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        return relay.Function(args, y)

    def check(shape, channels):
        x =  relay.var("x", shape=shape)
        weight = relay.var("weight")
        out_scale = relay.const(-_get_positive_scale((channels, 1, 1)))
        y1 = before(x, weight, out_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.backward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, out_scale, channels)
        y1_folded = relay.ir_pass.infer_type(y1_folded)
        y1_expected = relay.ir_pass.infer_type(y1_expected)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 8)


if __name__ == "__main__":
    test_fold_fwd_simple()
    test_fold_fwd_dual_path()
    test_fold_fwd_fail()
    test_fold_fwd_relu_fail()
    test_fold_fwd_negative_scale()
    test_fold_bwd_simple()
    test_fold_bwd_dual_path()
    test_fold_bwd_dual_consumer()
    test_fold_bwd_fail()
    test_fold_bwd_relu_fail()
    test_fold_bwd_negative_scale()
