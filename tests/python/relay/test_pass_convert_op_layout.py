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
"""Test alter op layout pass"""
import tvm
from tvm import te

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_no_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        return before()

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_nhwc_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_transpose_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d_transpose(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d_transpose(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d_transpose": ["NCHW", "OIHW"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_bias_pool_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.bias_add(y, bias, axis=3)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NHWC")
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))

        bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        bias = relay.layout_transform(bias, "NHWC", "NCHW")
        y = relay.add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, "int32")
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64))
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        ret = relay.concatenate([y, y1], axis=3)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64))
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2, channels=64, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_dual_path_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(3, 3, 32, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(
            y,
            weight2,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(3, 3, 32, 32))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW", "NHWC")
        y2 = relay.layout_transform(y, "NCHW", "NHWC")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_bn_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        gamma = relay.var("gamma")
        beta = relay.var("beta")
        mean = relay.var("mean")
        variance = relay.var("variance")
        y, _, _ = relay.nn.batch_norm(y, gamma, beta, mean, variance, axis=3)
        return relay.Function(analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))

    # Check that there is only 1 NHWC to NCHW transform.
    has_lt = list()
    find_op = lambda x: has_lt.append(
        isinstance(x, tvm.relay.expr.Call)
        and x.op.name == "layout_transform"
        and x.attrs.src_layout == "NCHW"
        and x.attrs.dst_layout == "NHWC"
    )
    relay.analysis.post_order_visit(a, find_op)
    has_lt = list(filter(lambda x: x, has_lt))
    assert len(has_lt) == 1


def test_resnet_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(
            x, weight2, channels=32, kernel_size=(1, 1), data_layout="NHWC", kernel_layout="HWIO"
        )
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NHWC")
        return relay.Function(analysis.free_vars(y), y)

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 32))
        weight2 = relay.var("weight2", shape=(1, 1, 64, 32))
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2, channels=32, kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        return relay.Function(analysis.free_vars(y), y)

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_scalar_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_bn_convert_layout():
    """ Check that layout transforms are propagated through bn. """

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=3)
        y = relay.nn.relu(y[0])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        x = relay.layout_transform(x, "NHWC", "NCHW")
        w = relay.layout_transform(w, "HWIO", "OIHW")
        y = relay.nn.conv2d(x, w, channels=64, kernel_size=(3, 3), padding=(1, 1))

        dtype = "float32"
        beta = relay.var("beta", relay.TensorType((64,), dtype))
        gamma = relay.var("gamma", relay.TensorType((64,), dtype))
        moving_mean = relay.var("moving_mean", relay.TensorType((64,), dtype))
        moving_var = relay.var("moving_var", relay.TensorType((64,), dtype))

        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var, axis=1)
        y = relay.nn.relu(y[0])
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_requantize_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        x = relay.layout_transform(x, "NHWC", "NCHW")
        weight = relay.layout_transform(weight, "HWIO", "OIHW")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.qnn.op.requantize(
            y,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=1,
            out_dtype="int32",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NHWC")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_concat_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y, "int8")
        ret = relay.qnn.op.concatenate(
            [y, y1],
            [relay.const(1, "float32"), relay.const(1, "float32")],
            [relay.const(1, "int32"), relay.const(1, "int32")],
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=3,
        )
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(
            y,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y, "int8")
        ret = relay.qnn.op.concatenate(
            [y, y1],
            [relay.const(1, "float32"), relay.const(1, "float32")],
            [relay.const(1, "int32"), relay.const(1, "int32")],
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            axis=1,
        )
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_add_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y, "int8")
        ret = relay.qnn.op.add(
            y,
            y1,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
        )
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64), dtype="int8")
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64), dtype="int8")
        weight1 = relay.layout_transform(weight1, "HWIO", "OIHW")
        weight2 = relay.layout_transform(weight2, "HWIO", "OIHW")
        y = relay.layout_transform(x, "NHWC", "NCHW")
        y = relay.qnn.op.conv2d(
            y,
            weight1,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y1 = relay.qnn.op.conv2d(
            y,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.cast(y, "int8")
        y1 = relay.cast(y, "int8")
        ret = relay.qnn.op.add(
            y,
            y1,
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "int32"),
        )
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"qnn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_conv_nhwc_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        weight = relay.var("weight", shape=(64, 64, 3, 3), dtype="int8")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        weight = relay.var("weight", shape=(64, 64, 3, 3), dtype="int8")
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight = relay.layout_transform(weight, "OIHW", "HWIO")
        y = relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"qnn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_convert_kernel_layout():
    """ Check that convolution kernel layout is correctly transformed. """

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        w = relay.var("weight", shape=(3, 3, 64, 64))
        w = relay.layout_transform(w, "HWIO", "OHWI")
        y = relay.nn.conv2d(
            x,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="OHWI",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_roi_align_convert_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NHWC"
        )
        ret = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "vision.roi_align": ["NHWC", "default"],
    }
    a = run_opt_pass(a, transform.ConvertLayout(desired_layouts))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_default_keyword():
    """ Check that the default keyword selects correct TVM default layout. """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 3, 3, 64))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight", shape=(64, 3, 3, 64))
        w = relay.layout_transform(w, "OHWI", "OIHW")
        y = relay.nn.conv2d(
            x,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_different_ops_convert_layout():
    """Check convert layout correctly supports converting the layout of
    different ops in the same graph.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 3, 3, 64))
        weight2 = relay.var("weight2", shape=(64, 3, 3, 64), dtype="int8")
        weight3 = relay.var("weight3", shape=(64, 3, 3, 64))
        out = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.cast(out, "int8")
        out = relay.qnn.op.conv2d(
            out,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.cast(out, "float32")
        out = relay.nn.conv2d_transpose(
            out,
            weight3,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OHWI",
        )
        out = relay.Function(analysis.free_vars(out), out)
        return out

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 3, 3, 64))
        weight2 = relay.var("weight2", shape=(64, 3, 3, 64), dtype="int8")
        weight3 = relay.var("weight3", shape=(64, 3, 3, 64))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OHWI", "HWIO")
        out = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        out = relay.cast(out, "int8")
        out = relay.layout_transform(out, "NHWC", "NCHW")
        weight2 = relay.layout_transform(weight2, "OHWI", "OIHW")
        out = relay.qnn.op.conv2d(
            out,
            weight2,
            relay.const(1, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        out = relay.cast(out, "float32")
        out = relay.layout_transform(out, "NCHW", "NHWC")
        weight3 = relay.layout_transform(weight3, "OHWI", "HWIO")
        out = relay.nn.conv2d_transpose(
            out,
            weight3,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        out = relay.layout_transform(out, "NHWC", "NCHW")
        out = relay.Function(analysis.free_vars(out), out)
        return out

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "qnn.conv2d": ["NCHW", "OIHW"],
        "nn.conv2d_transpose": ["NHWC", "HWIO"],
    }
    a = run_opt_pass(a, transform.ConvertLayout(desired_layouts))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_no_desired_layout():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(64, 64, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NHWC")
        weight1 = relay.layout_transform(weight1, "OIHW", "HWIO")
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.layout_transform(y, "NHWC", "NCHW")
        rois = relay.var("rois", shape=(32, 5))
        y = relay.vision.roi_align(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, sample_ratio=2, layout="NCHW"
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NHWC", "HWIO"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


if __name__ == "__main__":
    test_no_convert_layout()
    test_conv_convert_layout()
    test_conv_nhwc_convert_layout()
    test_conv_bias_pool_convert_layout()
    test_conv_concat_convert_layout()
    test_dual_path_convert_layout()
    test_bn_convert_layout()
    test_resnet_convert_layout()
    test_scalar_convert_layout()
    test_conv_bn_convert_layout()
    test_qnn_conv_requantize_convert_layout()
    test_qnn_conv_concat_convert_layout()
    test_qnn_conv_add_convert_layout()
    test_qnn_conv_nhwc_convert_layout()
    test_conv_convert_kernel_layout()
    test_conv_transpose_convert_layout()
    test_conv_roi_align_convert_layout()
    test_default_keyword()
    test_different_ops_convert_layout()
    test_no_desired_layout()
