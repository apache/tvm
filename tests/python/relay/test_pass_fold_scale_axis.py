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
import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import create_workload
from tvm.relay.build_module import bind_params_by_name


def initializer(_, param):
    param = np.zeros(param.shape)


def _get_positive_scale(size):
    return np.random.uniform(0.5, 1, size=size).astype("float32")


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_fold_fwd_simple():
    """Simple testcase."""

    def before(x, conv_weight, in_bias, in_scale, channels, blocking):
        args = [x, conv_weight, in_bias]
        x = relay.multiply(x, in_scale)
        x = relay.nn.relu(x)
        x = relay.add(x, in_bias)
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW2i{}o".format(blocking[1]) if blocking else "OIHW",
        )

        return relay.Function(args, y)

    def expected(x, conv_weight, in_bias, in_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, in_bias]
        if blocking:
            squeezed_scale = relay.squeeze(in_scale, axis=[0, 2, 3])
            x = relay.nn.relu(x)
            in_bias = relay.divide(
                in_bias,
                relay.reshape(squeezed_scale, (1, in_channels // blocking[0], 1, 1, blocking[0])),
            )  # NCHWc
            x = relay.add(x, in_bias)
            conv_weight = relay.multiply(
                conv_weight, relay.reshape(squeezed_scale, (1, in_channels // 2, 1, 1, 2, 1))
            )  # OIHWio
        else:
            squeezed_scale = relay.squeeze(in_scale, axis=[1, 2])
            x = relay.nn.relu(x)
            in_bias = relay.divide(
                in_bias, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2)
            )
            x = relay.add(x, in_bias)
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2)
            )

        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW2i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        return relay.Function(args, y)

    def check(shape, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            in_channels = shape[1] * shape[4]
            in_bias = relay.var("in_bias", shape=(1, in_channels // blocking[0], 1, 1, blocking[0]))
            in_scale = relay.const(
                _get_positive_scale((1, in_channels // blocking[0], 1, 1, blocking[0]))
            )
        else:
            in_channels = shape[1]
            in_bias = relay.var("in_bias", shape=(in_channels, 1, 1))
            in_scale = relay.const(_get_positive_scale((in_channels, 1, 1)))
        y1 = before(x, weight, in_bias, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        y1_expected = expected(x, weight, in_bias, in_scale, in_channels, channels, blocking)

        y1_folded = run_opt_pass(y1_folded, transform.InferType())
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 2, None)
    check((2, 2, 10, 10, 2), 8, (2, 4))


def test_fold_fwd_dual_path():
    """scale axis being consumed by two consumers"""

    def before(x, conv_weight, in_bias, in_scale, channels, blocking):
        args = [x, conv_weight, in_bias]
        x = relay.multiply(in_scale, x)
        x = relay.nn.relu(x)
        x = relay.subtract(x, in_bias)
        y1 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            groups=channels,
            padding=(1, 1),
        )
        y2 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            groups=channels,
            padding=(1, 1),
        )
        z = relay.add(y1, y2)
        return relay.Function(args, z)

    def expected(x, conv_weight, in_bias, in_scale, channels, blocking):
        args = [x, conv_weight, in_bias]
        x = relay.nn.relu(x)
        if blocking:
            _in_scale = relay.reshape(
                in_scale, (1, 1, 1, channels // blocking[0], blocking[0])
            )  # NHWCc
        else:
            _in_scale = in_scale
        in_bias = relay.divide(in_bias, _in_scale)
        x = relay.subtract(x, in_bias)
        if blocking:
            _in_scale = relay.reshape(
                in_scale, (1, 1, 1, channels // blocking[0], 1, blocking[0])
            )  # HWIOio
        y1 = relay.nn.conv2d(
            x,
            relay.multiply(conv_weight, _in_scale),
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            groups=channels,
            padding=(1, 1),
        )
        if blocking:
            _in_scale = relay.reshape(
                in_scale, (1, 1, 1, channels // blocking[0], 1, blocking[0])
            )  # HWIOio
        y2 = relay.nn.conv2d(
            x,
            relay.multiply(conv_weight, _in_scale),
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            groups=channels,
            padding=(1, 1),
        )
        z = relay.add(y1, y2)
        return relay.Function(args, z)

    def check(dshape, channels, blocking):
        x = relay.var("x", shape=dshape)
        if blocking:
            in_channels = dshape[3] * dshape[4]
            wshape = (3, 3, 1, channels // blocking[1], 1, blocking[1])  # HWIOio
            weight = relay.var("weight", shape=wshape)
            in_bias = relay.var("in_bias", shape=(in_channels // blocking[0], blocking[0]))
            in_scale = relay.const(_get_positive_scale((in_channels // blocking[0], blocking[0])))
        else:
            in_channels = dshape[-1]
            wshape = (3, 3, 1, channels)  # HWIO
            weight = relay.var("weight", shape=wshape)
            in_bias = relay.var("in_bias", shape=(in_channels,))
            in_scale = relay.const(
                _get_positive_scale(
                    in_channels,
                )
            )

        # test depthwise
        assert in_channels == channels

        y1 = before(x, weight, in_bias, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_expected = expected(x, weight, in_bias, in_scale, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 3), 3, None)
    check((2, 4, 10, 2, 2), 4, (2, 2))


def test_fold_fwd_fail():
    """testcase where we canont fold"""

    def before(x, conv_weight, in_bias, in_scale, channels, blocking):
        x = relay.multiply(x, in_scale)
        xx = relay.nn.leaky_relu(x, alpha=0.1)
        y1 = relay.nn.conv2d(
            xx,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            padding=(1, 1),
        )
        z = relay.add(y1, x)
        return relay.Function(relay.analysis.free_vars(z), z)

    def check(shape, channels, blocking):
        x = relay.var("x", shape=shape)
        if blocking:
            in_channels = shape[3] * shape[4]
            in_bias = relay.var("in_bias", shape=(in_channels // blocking[0], blocking[0]))
            in_scale = relay.const(_get_positive_scale((in_channels // blocking[0], blocking[0])))
        else:
            in_channels = shape[-1]
            in_bias = relay.var("in_bias", shape=(in_channels,))
            in_scale = relay.const(_get_positive_scale(size=(in_channels,)))
        # test depthwise
        assert in_channels == channels
        weight = relay.var("weight")
        y1 = before(x, weight, in_bias, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        assert tvm.ir.structural_equal(y1, y1_folded)

    check((2, 11, 10, 4), 4, None)
    check((2, 11, 10, 2, 2), 4, (2, 2))


def test_fold_fwd_relu_fail():
    """testcase where we canont fold because scale can not pass relu"""

    def before(x, conv_weight, in_bias, in_scale, channels, blocking):
        x = relay.multiply(x, in_scale)
        xx = relay.nn.relu(x)
        y1 = relay.nn.conv2d(
            xx,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC{}c".format(blocking[0]) if blocking else "NHWC",
            kernel_layout="HWIO1i{}o".format(blocking[1]) if blocking else "HWIO",
            padding=(1, 1),
        )
        z = relay.add(y1, x)
        return relay.Function(relay.analysis.free_vars(z), z)

    def check(shape, channels, blocking, in_scale):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            in_channels = shape[3] * shape[4]
            in_bias = relay.var("in_bias", shape=(1, in_channels // blocking[0], 1, 1, blocking[0]))
        else:
            in_channels = shape[-1]
            in_bias = relay.var("in_bias", shape=(in_channels,))

        assert in_channels == channels
        y1 = before(x, weight, in_bias, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        assert tvm.ir.structural_equal(y1, y1_folded)

    in_scale = relay.var("in_scale", shape=(4,))
    check((2, 11, 10, 4), 4, None, in_scale)
    in_scale = relay.const(-_get_positive_scale((4,)))
    check((2, 11, 10, 4), 4, None, in_scale)

    in_scale = relay.var("in_scale", shape=(1, 1, 1, 2, 2))
    check((2, 11, 10, 2, 2), 4, (2, 2), in_scale)
    in_scale = relay.const(-_get_positive_scale((1, 1, 1, 2, 2)))
    check((2, 11, 10, 2, 2), 4, (2, 2), in_scale)


def test_fold_fwd_let_fail():
    """testcase where we canont fold"""

    def before(x, conv_weight, in_bias, in_scale, channels):
        args = [x, conv_weight, in_bias]
        x = relay.multiply(x, in_scale)
        x = relay.nn.relu(x)
        x = relay.add(x, in_bias)
        x_var = relay.Var("x_var")
        y1 = relay.nn.conv2d(
            x_var,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="HWIO",
            padding=(1, 1),
        )
        z = relay.add(y1, x)
        let = relay.Let(x_var, x, z)
        return relay.Function(args, let)

    def check(shape, channels):
        x = relay.var("x", shape=shape)
        in_channels = shape[-1]
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(_get_positive_scale(size=(in_channels,)))
        # test depthwise
        assert in_channels == channels
        weight = relay.var("weight")
        y1 = before(x, weight, in_bias, in_scale, channels)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        assert tvm.ir.structural_equal(y1, y1_folded)

    check((2, 11, 10, 4), 4)


def test_fold_fwd_negative_scale():
    """Testcase of folding negative scale"""

    def before(x, conv_weight, in_scale, channels, blocking):
        args = [x, conv_weight]
        x = relay.multiply(x, in_scale)
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW4i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        return relay.Function(args, y)

    def expected(x, conv_weight, in_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight]
        if blocking:
            squeezed_scale = relay.squeeze(in_scale, axis=[0, 2, 3])
            conv_weight = relay.multiply(
                conv_weight, relay.reshape(squeezed_scale, (1, in_channels // 4, 1, 1, 4, 1))
            )
            # blocking by "i" in OIHWio
        else:
            squeezed_scale = relay.squeeze(in_scale, axis=[1, 2])
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2)
            )
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW4i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        return relay.Function(args, y)

    def check(shape, channels, blocking):
        x = relay.var("x", shape=shape)
        if blocking:
            in_channels = shape[1] * shape[4]
            in_scale = relay.const(-_get_positive_scale((1, shape[1], 1, 1, shape[4])))
        else:
            in_channels = shape[1]
            in_scale = relay.const(-_get_positive_scale((in_channels, 1, 1)))
        weight = relay.var("weight")
        y1 = before(x, weight, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        y1_expected = expected(x, weight, in_scale, in_channels, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4, None)
    check((2, 2, 10, 10, 2), 8, (2, 2))


def test_fold_fwd_dense():
    """dense testcase."""

    def before(x, weight, in_bias, in_scale):
        args = [x, weight, in_bias]
        x = relay.multiply(x, in_scale)
        x = relay.nn.relu(x)
        x = relay.add(x, in_bias)
        y = relay.nn.dense(x, weight)
        return relay.Function(args, y)

    def expected(x, weight, in_bias, in_scale):
        # use a fixed order of args so alpha equal check can pass
        args = [x, weight, in_bias]
        x = relay.nn.relu(x)
        in_bias = relay.divide(in_bias, in_scale)
        x = relay.add(x, in_bias)
        weight = relay.multiply(weight, in_scale)
        y = relay.nn.dense(x, weight)
        return relay.Function(args, y)

    def check(data_shape, weight_shape):
        x = relay.var("x", shape=data_shape)
        weight = relay.var("weight", shape=weight_shape)
        in_channels = data_shape[1]
        in_bias = relay.var("in_bias", shape=(in_channels,))
        in_scale = relay.const(_get_positive_scale((in_channels,)))
        y1 = before(x, weight, in_bias, in_scale)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        y1_expected = expected(x, weight, in_bias, in_scale)

        y1_folded = run_opt_pass(y1_folded, transform.InferType())
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4), (3, 4))
    check((3, 5), (4, 5))


def test_fold_bwd_simple():
    """Simple testcase."""

    def before(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        if blocking:
            out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, blocking[1]))
        else:
            out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        if blocking:
            out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, blocking[1]))
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        if blocking:
            out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, blocking[1]))
            out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, blocking[1]))
            squeezed_scale = relay.squeeze(out_scale, axis=[0, 2, 3])
            conv_weight = relay.multiply(
                conv_weight,
                relay.reshape(squeezed_scale, (channels // blocking[1], 1, 1, 1, 1, blocking[1])),
            )
        else:
            out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
            squeezed_scale = relay.squeeze(out_scale, axis=[1, 2])
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
            )

        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        if blocking:
            out_bias = relay.multiply(
                out_bias,
                relay.reshape(squeezed_scale, (1, channels // blocking[1], 1, 1, blocking[1])),
            )
        else:
            out_bias = relay.multiply(
                out_bias, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=2)
            )
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        return relay.Function(args, y)

    def check(shape, in_channels, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        if blocking:
            out_scale = relay.const(_get_positive_scale((channels,)))
        else:
            out_scale = relay.const(_get_positive_scale((channels, 1, 1)))
        y1 = before(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4, 8, None)
    check((2, 2, 10, 10, 16), 32, 64, (16, 16))


def test_fold_bwd_dual_path():
    """Dual path testcase."""

    def before(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        y1 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        if not blocking:
            out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=2)
        squeezed_scale = relay.squeeze(out_scale, axis=[1, 2])

        def fold_conv_weight():
            if blocking:
                return relay.multiply(
                    conv_weight,
                    relay.reshape(
                        squeezed_scale, (channels // blocking[1], 1, 1, 1, 1, blocking[1])
                    ),
                )
            else:
                return relay.multiply(
                    conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
                )

        y1 = relay.nn.conv2d(
            x,
            fold_conv_weight(),
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(
            x,
            fold_conv_weight(),
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, in_channels, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            out_bias = relay.var("out_bias", shape=(channels // blocking[1], 1, 1, blocking[1]))
            out_scale = relay.const(
                _get_positive_scale((channels // blocking[1], 1, 1, blocking[1]))
            )
        else:
            out_bias = relay.var("out_bias", shape=(channels,))
            out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

        y1 = before(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4, 8, None)
    check((2, 2, 10, 10, 2), 4, 8, (2, 2))


def test_fold_bwd_simple_constant():
    def before(data, weight, out_bias, channels):
        y = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )

        y = relay.add(y, out_bias)
        c2 = relay.const(2.0)
        y = relay.nn.relu(y)
        y = relay.multiply(y, c2)
        mod, params = create_workload(y, initializer)
        mod["main"] = bind_params_by_name(mod["main"], params)
        return mod

    def expected(data, weight, out_bias, channels):
        y0 = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        y0 = relay.add(y0, out_bias)
        y0 = relay.nn.relu(y0)
        mod, params = create_workload(y0, initializer)
        mod["main"] = bind_params_by_name(mod["main"], params)
        return mod

    def check(shape, channels):
        x = relay.var("data", relay.TensorType(shape, "float32"))
        weight = relay.var("weight")
        out_bias = relay.var("in_bias", shape=(channels, 1, 1))

        y0 = before(x, weight, out_bias, channels)
        remove_last_multiply = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.FoldScaleAxis(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            y0 = remove_last_multiply(y0)
        _expect = expected(x, weight, out_bias, channels)
        tvm.ir.assert_structural_equal(y0, _expect)

    check((1, 3, 200, 200), 16)


def test_fold_bwd_dual_consumer():
    def before(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        y0 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y0 = relay.multiply(y0, out_scale)
        y0 = relay.nn.relu(y0)

        y1 = relay.nn.conv2d(
            y0,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y1 = relay.multiply(y1, out_scale)
        y1 = relay.nn.relu(y1)

        y2 = relay.nn.conv2d(
            y0,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y2 = relay.multiply(y2, out_scale)
        y2 = relay.nn.relu(y2)

        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]

        def fold_conv_weight():
            squeezed_scale = relay.squeeze(out_scale, axis=[1, 2])
            if blocking:
                return relay.multiply(
                    conv_weight,
                    relay.reshape(
                        squeezed_scale, (channels // blocking[1], 1, 1, 1, 1, blocking[1])
                    ),
                )
            else:
                return relay.multiply(
                    conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
                )

        y0 = relay.nn.conv2d(
            x,
            fold_conv_weight(),
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y0 = relay.nn.relu(y0)
        y1 = relay.nn.conv2d(
            y0,
            fold_conv_weight(),
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(
            y0,
            fold_conv_weight(),
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, in_channels, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            out_bias = relay.var("out_bias", shape=(channels // blocking[1], 1, 1, blocking[1]))
            out_scale = relay.const(
                _get_positive_scale((channels // blocking[1], 1, 1, blocking[1]))
            )
        else:
            out_bias = relay.var("out_bias", shape=(channels,))
            out_scale = relay.const(_get_positive_scale((channels, 1, 1)))

        y1 = before(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4, 4, None)
    check((2, 2, 10, 10, 2), 4, 4, (2, 2))


def test_fold_bwd_fail():
    """Dual path testcase."""

    def fail1(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        y1 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
            out_layout="CNHW{}c".format(blocking[1]) if blocking else "CNHW",
        )
        # fold will fail because the axis from two path
        # differs from each other.
        y2 = relay.nn.relu(y2)
        y = relay.add(y1, y2)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def fail2(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        y1 = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y2 = relay.nn.relu(y1)
        # fold will fail because y1 is referred also by y2
        y1 = relay.multiply(y1, out_scale)
        y = relay.add(y1, y2)
        return relay.Function(args, y)

    def check(shape, in_channels, channels, blocking, fbefore):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            out_bias = relay.var("out_bias", shape=(channels // blocking[1], 1, 1, blocking[1]))
            out_scale = relay.const(
                _get_positive_scale((channels // blocking[1], 1, 1, blocking[1]))
            )
        else:
            out_bias = relay.var("out_bias", shape=(channels, 1, 1))
            out_scale = relay.const(_get_positive_scale((channels, 1, 1)))
        y1 = fbefore(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        assert tvm.ir.structural_equal(y1_folded, y1)

    check((4, 4, 10, 10), 4, 4, None, fail1)
    check((2, 2, 10, 10, 2), 4, 4, (2, 2), fail1)
    check((4, 4, 10, 10), 4, 4, None, fail2)
    check((4, 2, 10, 10, 2), 4, 4, (2, 2), fail2)


def test_fold_bwd_relu_fail():
    """testcase where we canont fold because scale can not pass relu"""

    def before(x, conv_weight, out_scale, channels, blocking):
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.multiply(x, out_scale)
        return relay.Function(relay.analysis.free_vars(y), y)

    def check(shape, channels, blocking, out_scale):
        x = relay.var("x", shape=shape)
        in_channels = shape[1]
        weight = relay.var("weight")
        y1 = before(x, weight, out_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        assert tvm.ir.structural_equal(y1, y1_folded)

    out_scale = relay.var("in_scale", shape=(4, 1, 1))
    check((4, 4, 10, 10), 4, None, out_scale)
    out_scale = relay.const(np.random.uniform(size=(4, 1, 1), low=-1.0, high=0.0)).astype("float32")
    check((4, 4, 10, 10), 4, None, out_scale)

    out_scale = relay.var("in_scale", shape=(1, 2, 1, 1, 2))
    check((4, 2, 10, 10, 2), 4, (2, 2), out_scale)
    out_scale = relay.const(np.random.uniform(size=(1, 2, 1, 1, 2), low=-1.0, high=0.0)).astype(
        "float32"
    )
    check((4, 2, 10, 10, 2), 4, (2, 2), out_scale)


def test_fold_bwd_negative_scale():
    """Testcase of folding negative scale"""

    def before(x, conv_weight, out_scale, channels, blocking):
        args = [x, conv_weight]
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_scale, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight]
        if blocking:
            squeezed_scale = relay.squeeze(out_scale, axis=[0, 2, 3])
            conv_weight = relay.multiply(
                conv_weight,
                relay.reshape(squeezed_scale, (channels // blocking[1], 1, 1, 1, 1, blocking[1])),
            )
        else:
            squeezed_scale = relay.squeeze(out_scale, axis=[1, 2])
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
            )
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW{}c".format(blocking[0]) if blocking else "NCHW",
            kernel_layout="OIHW1i{}o".format(blocking[1]) if blocking else "OIHW",
        )
        return relay.Function(args, y)

    def check(shape, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            out_scale = relay.const(
                -_get_positive_scale((1, channels // blocking[1], 1, 1, blocking[1]))
            )
        else:
            out_scale = relay.const(-_get_positive_scale((channels, 1, 1)))
        y1 = before(x, weight, out_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_scale, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 8, None)
    check((2, 2, 10, 10, 2), 8, (2, 2))


def test_fold_bwd_dense():
    """dense testcase."""

    def before(x, weight, in_bias, in_scale):
        args = [x, weight, in_bias]
        x = relay.nn.dense(x, weight)
        x = relay.add(x, in_bias)
        x = relay.nn.relu(x)
        y = relay.multiply(x, in_scale)
        return relay.Function(args, y)

    def expected(x, weight, in_bias, in_scale):
        # use a fixed order of args so alpha equal check can pass
        args = [x, weight, in_bias]
        scale = relay.expand_dims(in_scale, axis=1)
        weight = relay.multiply(weight, scale)
        x = relay.nn.dense(x, weight)
        bias = relay.multiply(in_bias, in_scale)
        x = relay.add(x, bias)
        y = relay.nn.relu(x)
        return relay.Function(args, y)

    def check(data_shape, weight_shape):
        x = relay.var("x", shape=data_shape)
        weight = relay.var("weight", shape=weight_shape)
        out_channels = weight_shape[0]
        in_bias = relay.var("in_bias", shape=(out_channels,))
        in_scale = relay.const(_get_positive_scale((out_channels,)))
        y1 = before(x, weight, in_bias, in_scale)
        y1 = run_opt_pass(y1, transform.InferType())
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, in_bias, in_scale)

        y1_folded = run_opt_pass(y1_folded, transform.InferType())
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4), (3, 4))
    check((3, 5), (4, 5))


def test_fold_bwd_bias_add():
    """bias add testcase."""

    def before(x, conv_weight, out_bias, out_scale, channels):
        args = [x, conv_weight, out_bias]
        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.bias_add(y, out_bias)
        y = relay.nn.relu(y)
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, channels):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        squeezed_scale = relay.squeeze(out_scale, axis=[1, 2])
        conv_weight = relay.multiply(
            conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
        )

        y = relay.nn.conv2d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )

        out_bias = relay.multiply(out_bias, squeezed_scale)
        y = relay.nn.bias_add(y, out_bias)
        y = relay.nn.relu(y)
        return relay.Function(args, y)

    def check(shape, channels):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        out_scale = relay.const(_get_positive_scale((channels, 1, 1)))
        y1 = before(x, weight, out_bias, out_scale, channels)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_bias, out_scale, channels)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10), 4)


def test_fold_fwd_conv3d():
    """Conv3d testcase."""

    def before(x, conv_weight, in_bias, in_scale, channels, blocking):
        args = [x, conv_weight, in_bias]
        x = relay.multiply(x, in_scale)
        x = relay.nn.relu(x)
        x = relay.add(x, in_bias)
        y = relay.nn.conv3d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            data_layout="NCDHW{}c".format(blocking[0]) if blocking else "NCDHW",
            kernel_layout="OIDHW2i{}o".format(blocking[1]) if blocking else "OIDHW",
        )

        return relay.Function(args, y)

    def expected(x, conv_weight, in_bias, in_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, in_bias]
        if blocking:
            squeezed_scale = relay.squeeze(in_scale, axis=[0, 2, 3, 4])
            x = relay.nn.relu(x)
            in_bias = relay.divide(
                in_bias,
                relay.reshape(
                    squeezed_scale, (1, in_channels // blocking[0], 1, 1, 1, blocking[0])
                ),
            )  # NCHWc
            x = relay.add(x, in_bias)
            conv_weight = relay.multiply(
                conv_weight, relay.reshape(squeezed_scale, (1, in_channels // 2, 1, 1, 1, 2, 1))
            )  # OIHWio
        else:
            squeezed_scale = relay.squeeze(in_scale, axis=[1, 2, 3])
            x = relay.nn.relu(x)
            in_bias = relay.divide(
                in_bias, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
            )
            x = relay.add(x, in_bias)
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
            )

        y = relay.nn.conv3d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            data_layout="NCDHW{}c".format(blocking[0]) if blocking else "NCDHW",
            kernel_layout="OIDHW2i{}o".format(blocking[1]) if blocking else "OIDHW",
        )
        return relay.Function(args, y)

    def check(shape, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        if blocking:
            in_channels = shape[1] * shape[-1]
            in_bias = relay.var(
                "in_bias", shape=(1, in_channels // blocking[0], 1, 1, 1, blocking[0])
            )
            in_scale = relay.const(
                _get_positive_scale((1, in_channels // blocking[0], 1, 1, 1, blocking[0]))
            )
        else:
            in_channels = shape[1]
            in_bias = relay.var("in_bias", shape=(in_channels, 1, 1, 1))
            in_scale = relay.const(_get_positive_scale((in_channels, 1, 1, 1)))
        y1 = before(x, weight, in_bias, in_scale, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.ForwardFoldScaleAxis())
        y1_expected = expected(x, weight, in_bias, in_scale, in_channels, channels, blocking)

        y1_folded = run_opt_pass(y1_folded, transform.InferType())
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10, 10), 2, None)
    check((2, 2, 10, 10, 10, 2), 8, (2, 4))


def test_fold_bwd_conv3d():
    """Conv3d testcase."""

    def before(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        args = [x, conv_weight, out_bias]
        if blocking:
            out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, 1, blocking[1]))
        else:
            out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=3)
        y = relay.nn.conv3d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            data_layout="NCDHW{}c".format(blocking[0]) if blocking else "NCDHW",
            kernel_layout="OIDHW1i{}o".format(blocking[1]) if blocking else "OIDHW",
        )
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        if blocking:
            out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, 1, blocking[1]))
        y = relay.multiply(y, out_scale)
        return relay.Function(args, y)

    def expected(x, conv_weight, out_bias, out_scale, in_channels, channels, blocking):
        # use a fixed order of args so alpha equal check can pass
        args = [x, conv_weight, out_bias]
        if blocking:
            out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, 1, blocking[1]))
            out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, 1, blocking[1]))
            squeezed_scale = relay.squeeze(out_scale, axis=[0, 2, 3, 4])
            conv_weight = relay.multiply(
                conv_weight,
                relay.reshape(
                    squeezed_scale, (channels // blocking[1], 1, 1, 1, 1, 1, blocking[1])
                ),
            )
        else:
            out_bias = relay.expand_dims(out_bias, axis=1, num_newaxis=3)
            squeezed_scale = relay.squeeze(out_scale, axis=[1, 2, 3])
            conv_weight = relay.multiply(
                conv_weight, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=4)
            )

        y = relay.nn.conv3d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            data_layout="NCDHW{}c".format(blocking[0]) if blocking else "NCDHW",
            kernel_layout="OIDHW1i{}o".format(blocking[1]) if blocking else "OIDHW",
        )
        if blocking:
            out_bias = relay.multiply(
                out_bias,
                relay.reshape(squeezed_scale, (1, channels // blocking[1], 1, 1, 1, blocking[1])),
            )
        else:
            out_bias = relay.multiply(
                out_bias, relay.expand_dims(squeezed_scale, axis=1, num_newaxis=3)
            )
        y = relay.add(y, out_bias)
        y = relay.nn.relu(y)
        return relay.Function(args, y)

    def check(shape, in_channels, channels, blocking):
        x = relay.var("x", shape=shape)
        weight = relay.var("weight")
        out_bias = relay.var("out_bias", shape=(channels,))
        if blocking:
            out_scale = relay.const(_get_positive_scale((channels,)))
        else:
            out_scale = relay.const(_get_positive_scale((channels, 1, 1, 1)))
        y1 = before(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1 = run_opt_pass(y1, transform.InferType())
        type_dict = {x.name_hint: x.checked_type for x in y1.params}
        weight = relay.var("weight", type_dict["weight"])
        y1_folded = run_opt_pass(y1, transform.BackwardFoldScaleAxis())
        y1_expected = expected(x, weight, out_bias, out_scale, in_channels, channels, blocking)
        y1_expected = run_opt_pass(y1_expected, transform.InferType())
        assert tvm.ir.structural_equal(y1_folded, y1_expected)

    check((2, 4, 10, 10, 10), 4, 8, None)
    check((2, 2, 10, 10, 10, 16), 32, 64, (16, 16))


if __name__ == "__main__":
    test_fold_fwd_simple()
    test_fold_fwd_dual_path()
    test_fold_fwd_fail()
    test_fold_fwd_relu_fail()
    test_fold_fwd_negative_scale()
    test_fold_fwd_dense()
    test_fold_bwd_simple_constant()
    test_fold_bwd_simple()
    test_fold_bwd_dual_path()
    test_fold_bwd_dual_consumer()
    test_fold_bwd_fail()
    test_fold_bwd_relu_fail()
    test_fold_bwd_negative_scale()
    test_fold_bwd_dense()
    test_fold_bwd_bias_add()
    test_fold_fwd_conv3d()
    test_fold_bwd_conv3d()
