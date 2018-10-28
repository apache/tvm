from tvm import relay


def test_fold_fwd_simple():
    """Simple testcase."""
    def before(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias, in_scale]
        in_scale = relay.expand_dims(in_scale, axis=1, num_newaxis=2)
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
        args = [x, conv_weight, in_bias, in_scale]
        in_scale = relay.expand_dims(in_scale, axis=1, num_newaxis=2)
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
        in_scale = relay.var("in_scale", shape=(in_channels,))

        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        y1_expected = expected(x, weight, in_bias, in_scale, channels)
        assert relay.ir_pass.alpha_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 2)


def test_fold_fwd_dual_path():
    """scale axis being consumed by two consumers"""
    def before(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias, in_scale]
        x = relay.multiply(in_scale, x)
        x = relay.nn.relu(x)
        x = relay.subtract(x, in_bias)
        y1 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             weight_layout="HWOI",
                             groups=channels,
                             padding=(1, 1))
        y2 = relay.nn.conv2d(x, conv_weight,
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             weight_layout="HWOI",
                             groups=channels,
                             padding=(1, 1))
        z = relay.add(y1, y2)
        return relay.Function(args, z)

    def expected(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias, in_scale]
        x = relay.nn.relu(x)
        in_bias = relay.divide(in_bias, in_scale)
        x = relay.subtract(x, in_bias)
        y1 = relay.nn.conv2d(x,
                             relay.multiply(conv_weight, in_scale),
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             weight_layout="HWOI",
                             groups=channels,
                             padding=(1, 1))
        y2 = relay.nn.conv2d(x,
                             relay.multiply(conv_weight, in_scale),
                             channels=channels,
                             kernel_size=(3, 3),
                             data_layout="NHWC",
                             weight_layout="HWOI",
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
        in_scale = relay.var("in_scale", shape=(in_channels,))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        type_dict = {x.name_hint:x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_expected = expected(x, weight, in_bias, in_scale, channels)
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
        in_scale = relay.var("in_scale", shape=(in_channels,))
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = relay.ir_pass.infer_type(y1)
        y1_folded = relay.ir_pass.forward_fold_scale_axis(y1)
        assert relay.ir_pass.alpha_equal(y1, y1_folded)

    check((2, 11, 10, 4), 4)


if __name__ == "__main__":
    test_fold_fwd_simple()
    test_fold_fwd_dual_path()
    test_fold_fwd_fail()
