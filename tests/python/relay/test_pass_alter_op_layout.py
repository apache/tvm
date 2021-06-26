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
import pytest

import tvm
from tvm import relay, topi
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import run_infer_type
import numpy as np
import tvm.testing


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_alter_op():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            relay.multiply(weight, relay.const(2.0, "float32")),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_return_none():
    """Test doing nothing by returning 'None'"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        called[0] = True
        return None

    with TempOpAttr("nn.global_max_pool2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(before(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)
    assert called[0]


def test_alter_layout():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias")
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias)
        # a useless tuple, which will be eliminated
        y = relay.Tuple([y])[0]
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "OIHW16i"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(64, 64, 3, 3))

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        w = relay.layout_transform(weight, "OIHW", "OIHW16i")
        y = relay.nn.conv2d(
            y,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            kernel_layout="OIHW16i",
            data_layout="NCHW16c",
        )
        b = relay.expand_dims(bias, axis=1, num_newaxis=2)
        b = relay.expand_dims(b, axis=0, num_newaxis=1)
        b = relay.layout_transform(b, "NCHW", "NCHW16c")
        y = relay.add(y, b)

        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NCHW16c")
        y = relay.cast(y, "int32")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_lrn():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias")
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.max_pool2d(y, pool_size=(2, 2))
        y = relay.nn.lrn(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "OIHW16i"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        weight = relay.var("weight", shape=(64, 64, 3, 3))

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        w = relay.layout_transform(weight, "OIHW", "OIHW16i")
        y = relay.nn.conv2d(
            y,
            w,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            kernel_layout="OIHW16i",
            data_layout="NCHW16c",
        )
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.lrn(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_dual_path():
    """
    Test alternating the layout with two outputs.
    One path continues to use the new layout while one path fall backs to old layout.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.relu(y1)
        y2 = relay.nn.batch_flatten(y)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.nn.relu(y)
        y1 = relay.nn.conv2d(
            y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y1 = relay.nn.relu(y1)
        y1 = relay.layout_transform(y1, "NCHW16c", "NCHW")
        y2 = relay.layout_transform(y, "NCHW16c", "NCHW")
        y2 = relay.nn.batch_flatten(y2)
        ret = relay.Tuple([y1, y2])
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_resnet():
    """Test alternating the layout of a residual block
    This also tests the elimination of duplicated transformation.
    If a same transformation applies to a same node twice, only one transformation will be created.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2, channels=32, kernel_size=(1, 1))
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y)
        return relay.Function(analysis.free_vars(y), y)

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.nn.relu(y)
        y2 = relay.nn.conv2d(x, weight2, channels=32, kernel_size=(1, 1), data_layout="NCHW16c")
        y2 = relay.nn.relu(y2)
        y = y + y2
        y = relay.nn.global_max_pool2d(y, layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        return relay.Function(analysis.free_vars(y), y)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_broadcast_op():
    """Test boradcast operators"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.bias_add(y, bias)  # test broadcasting to lhs
        y = relay.multiply(scale, y)  # test broadcasting to rhs
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        bias = relay.var("bias", shape=(64,))
        scale = relay.var("scale", shape=(64, 1, 1))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        bias = relay.expand_dims(bias, 1, 2)
        bias = relay.expand_dims(bias, 0, 1)
        bias = relay.layout_transform(bias, "NCHW", "NCHW16c")
        scale = relay.expand_dims(scale, 0, 1)
        scale = relay.layout_transform(scale, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.add(y, bias)  # test broadcasting to lhs
        y = relay.multiply(scale, y)  # test broadcasting to rhs
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_broadcast_scalar_op():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """

    def before():
        x = relay.var("x", shape=(1, 500, 500, 64))
        kernel = relay.var("kernel", shape=(3, 3, 64, 64), dtype="float32")
        bias = relay.var("bias", shape=(64,))
        multiplier1 = relay.var("multiplier1", shape=(1,), dtype="float32")
        multiplier2 = relay.var("multiplier2", shape=(1, 1), dtype="float32")

        y = relay.nn.conv2d(x, kernel, data_layout="NHWC", kernel_layout="HWIO", kernel_size=(3, 3))
        y = relay.add(bias, y)
        y = relay.nn.relu(y)

        y = relay.multiply(multiplier1, y)
        y = relay.multiply(y, multiplier2)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 500, 500, 64))
        kernel = relay.var("kernel", shape=(3, 3, 64, 64), dtype="float32")
        bias = relay.var("bias", shape=(64,))
        multiplier1 = relay.var("multiplier1", shape=(1,), dtype="float32")
        multiplier2 = relay.var("multiplier2", shape=(1, 1), dtype="float32")

        b = relay.expand_dims(bias, axis=0, num_newaxis=3)
        b = relay.layout_transform(b, "NHWC", "NCHW16c")

        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(
            y, kernel, data_layout="NCHW16c", kernel_layout="HWIO", kernel_size=(3, 3)
        )

        y = relay.add(b, y)
        y = relay.nn.relu(y)

        y = relay.multiply(multiplier1, y)
        y = relay.multiply(y, multiplier2)
        y = relay.layout_transform(y, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_scalar():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.add(y, relay.const(1, "float32"))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight")

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, w, channels=64, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.add(y, relay.const(1.0, "float32"))

        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_scalar_regression():
    """regression test where scalar fails"""

    def before():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 16))
        bias = relay.var("bias", shape=(1, 1, 1, 16))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.add(y, bias)
        mean = relay.mean(y, axis=3, exclude=True)
        var = relay.variance(y, axis=3, exclude=True)
        gamma = relay.var("gamma")
        beta = relay.var("beta")
        y = relay.nn.batch_norm(y, gamma, beta, mean, var, axis=3)
        y = y[0]
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 16))
        bias = relay.var("bias", shape=(1, 1, 1, 16))
        x = relay.layout_transform(x, src_layout="NHWC", dst_layout="NCHW")
        x = relay.layout_transform(x, src_layout="NCHW", dst_layout="NCHW16c")
        weight = relay.layout_transform(weight, src_layout="HWIO", dst_layout="OIHW")
        y = relay.nn.conv2d(
            x, weight, channels=16, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        bias = relay.layout_transform(bias, src_layout="NHWC", dst_layout="NCHW")
        bias = relay.layout_transform(bias, src_layout="NCHW", dst_layout="NCHW16c")
        add = relay.add(y, bias)
        y = relay.layout_transform(add, src_layout="NCHW16c", dst_layout="NCHW")
        y = relay.layout_transform(y, src_layout="NCHW", dst_layout="NHWC")
        mean = relay.mean(y, axis=3, exclude=True)
        var = relay.variance(y, axis=3, exclude=True)
        denom = relay.const(1.0) / relay.sqrt(var + relay.const(1e-05))
        gamma = relay.var("gamma", shape=(16,))
        denom = denom * gamma
        denom_expand1 = relay.expand_dims(denom, axis=1, num_newaxis=2)
        denom_expand2 = relay.expand_dims(denom_expand1, axis=0)
        denom_nchwc16 = relay.layout_transform(
            denom_expand2, src_layout="NCHW", dst_layout="NCHW16c"
        )
        out = add * denom_nchwc16
        beta = relay.var("beta", shape=(16,))
        numerator = (-mean) * denom + beta
        numerator_expand1 = relay.expand_dims(numerator, axis=1, num_newaxis=2)
        numerator_expand2 = relay.expand_dims(numerator_expand1, axis=0)
        numerator_nchwc16 = relay.layout_transform(
            numerator_expand2, src_layout="NCHW", dst_layout="NCHW16c"
        )
        out = out + numerator_nchwc16
        out = relay.layout_transform(out, src_layout="NCHW16c", dst_layout="NCHW")
        y = relay.layout_transform(out, src_layout="NCHW", dst_layout="NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        desired_layouts = {"nn.conv2d": ["NCHW", "default"], "nn.batch_norm": ["NHWC", "default"]}
        a = run_opt_pass(
            a,
            [
                transform.InferType(),
                relay.transform.ConvertLayout(desired_layouts),
                transform.SimplifyInference(),
                transform.CanonicalizeOps(),
                transform.AlterOpLayout(),
            ],
        )
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_concatenate():
    """NCHW, NHWC and corner case concatenate layout transform."""

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    # NCHW layout transformation.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y1 = relay.nn.conv2d(y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.concatenate([y, y1], axis=1)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y1 = relay.nn.conv2d(
            y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nchw()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nchw(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # NHWC layout transformation.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.nn.conv2d(
            x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NHWC"
        )
        y1 = relay.nn.conv2d(
            y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NHWC"
        )
        ret = relay.concatenate([y, y1], axis=3)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        weight2 = relay.var("weight2")
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y1 = relay.nn.conv2d(
            y, weight2, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.concatenate([y, y1], axis=1)
        ret = relay.layout_transform(ret, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_nchw_upsamping_op():
    """Test upsamping operators"""

    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight", shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.upsampling(y, scale_h=2, scale_w=2)
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.nn.upsampling(y, scale_h=2, scale_w=2, layout="NCHW16c")
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2), layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_nchw_dyn_upsamping_op():
    """Test upsamping operators"""

    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight", shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.upsampling(y, scale_h=relay.const(2), scale_w=relay.const(2))
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight")
        x = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.nn.upsampling(y, scale_h=relay.const(2), scale_w=relay.const(2), layout="NCHW16c")
        y = relay.nn.avg_pool2d(y, pool_size=(2, 2), strides=(2, 2), layout="NCHW16c")
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


@tvm.testing.uses_gpu
def test_alter_layout_strided_slice():
    """Test rewriting strided_slice during alter_iop_layout"""

    def before():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight", shape=(32, 32, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        y = relay.strided_slice(y, begin=[0, 16], end=[1, 33], strides=[1, 1])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW4c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 32, 28, 28))
        weight = relay.var("weight", shape=(32, 32, 3, 3))
        weight = relay.layout_transform(weight, "OIHW", "OIHW4i4o")
        x = relay.layout_transform(x, "NCHW", "NCHW4c")
        y = relay.op.nn.contrib_conv2d_nchwc(
            x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW4c"
        )

        y = relay.strided_slice(y, begin=[0, 4], end=[1, 21], strides=[1, 1])

        y = relay.layout_transform(y, "NCHW4c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        b = run_opt_pass(expected(), transform.InferType())

    # Verify inference result
    mod_before = tvm.IRModule()
    mod_new = tvm.IRModule()
    mod_before["main"] = a
    mod_new["main"] = b
    mod_before = transform.InferType()(mod_before)
    mod_new = transform.InferType()(mod_new)
    with relay.build_config(opt_level=3):
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug", "vm"]:
                ex_before = relay.create_executor(kind, mod=mod_before, device=dev, target=target)
                ex_new = relay.create_executor(kind, mod=mod_new, device=dev, target=target)
                np_data = np.random.uniform(size=(1, 32, 28, 28)).astype("float32")
                np_weight = np.random.uniform(size=(32, 32, 3, 3)).astype("float32")
                result_before = ex_before.evaluate()(np_data, np_weight)
                result_new = ex_new.evaluate()(np_data, np_weight)
                tvm.testing.assert_allclose(
                    result_before.numpy(), result_new.numpy(), rtol=1e-5, atol=1e-5
                )


@tvm.testing.uses_gpu
def test_alter_layout_strided_slice_axes_nhwc():
    """Test rewriting strided_slice with axes during alter_iop_layout"""

    def before():
        x = relay.var("x", shape=(1, 28, 28, 32))
        weight = relay.var("weight", shape=(3, 3, 32, 32))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y = relay.strided_slice(y, begin=[0, 16], end=[1, 32], strides=[1, 1], axes=[0, 3])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NHWC4c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 28, 28, 32))
        weight = relay.var("weight", shape=(3, 3, 32, 32))
        x = relay.layout_transform(x, "NHWC", "NHWC4c")
        y = relay.op.nn.conv2d(
            x,
            weight,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC4c",
            kernel_layout="HWIO",
        )
        y = relay.strided_slice(y, begin=[0, 4], end=[1, 8], strides=[1, 1], axes=[0, 3])
        y = relay.layout_transform(y, "NHWC4c", "NHWC")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    mod_before = tvm.IRModule()
    mod_new = tvm.IRModule()
    mod_before["main"] = a
    mod_new["main"] = b
    assert tvm.ir.structural_equal(mod_before, mod_new)


def test_alter_layout_depthwise_conv2d():
    """Test depthwise_conv2d operator"""

    def before():
        x = relay.var("x", shape=(1, 32, 56, 56))
        w = relay.var("w", shape=(32, 1, 3, 3))
        y = relay.nn.conv2d(x, w, padding=(1, 1), channels=32, kernel_size=(3, 3), groups=32)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    from tvm import topi

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        with tvm.target.Target("llvm"):
            return topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, out_type)

    def expected():
        x = relay.var("x", shape=(1, 32, 56, 56))
        w = relay.var("w", shape=(32, 1, 3, 3))
        x = relay.layout_transform(x, "NCHW", "NCHW8c")
        w = relay.layout_transform(w, "OIHW", "OIHW1i8o")
        y = relay.nn.contrib_depthwise_conv2d_nchwc(
            x,
            w,
            padding=(1, 1, 1, 1),
            channels=32,
            kernel_size=(3, 3),
            groups=32,
            data_layout="NCHW8c",
            kernel_layout="OIHW1i8o",
            out_layout="NCHW8c",
        )
        y = relay.layout_transform(y, "NCHW8c", "NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b)


def test_alter_layout_prelu():
    """Test PRelu operator"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        alpha = relay.var("alpha", relay.IncompleteType())
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.prelu(y, alpha)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        w = relay.var("weight")
        alpha = relay.var("alpha", relay.IncompleteType())

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, w, channels=64, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
        y = relay.nn.prelu(y, alpha)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, [transform.CanonicalizeOps(), transform.AlterOpLayout()])
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b)


def test_alter_layout_pad():
    """Check NCHW, NHWC and corner case for pad layout conversion"""

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Check NCHW conversion.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nchw()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nchw(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(
            x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NHWC"
        )
        ret = relay.nn.pad(y, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.nn.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)))
        ret = relay.layout_transform(ret, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # Check that conversion does not happen when padding along split axis.
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.nn.pad(y, pad_width=((0, 0), (1, 1), (1, 1), (1, 1)))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.layout_transform(y, "NCHW16c", "NCHW")
        ret = relay.nn.pad(ret, pad_width=((0, 0), (1, 1), (1, 1), (1, 1)))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_pool():
    """Check NCHW, NHWC pool layout conversion"""

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Check NCHW conversion.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1))
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout="NCHW16c")
        ret = relay.layout_transform(ret, "NCHW16c", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nchw()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nchw(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(
            x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NHWC"
        )
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout="NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout="NCHW16c")
        ret = relay.layout_transform(ret, "NCHW16c", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_sum():
    """Check NCHW, NHWC sum layout conversion"""

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    # Check NCHW conversion.
    def before_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1))
        ret = relay.sum(y, axis=1, keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.layout_transform(y, "NCHW16c", "NCHW")
        ret = relay.sum(ret, axis=[1], keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nchw()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nchw(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.nn.conv2d(
            x, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NHWC"
        )
        ret = relay.sum(y, axis=3, keepdims=True)
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    def expected_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1")
        y = relay.layout_transform(x, "NHWC", "NCHW16c")
        y = relay.nn.conv2d(
            y, weight1, channels=32, kernel_size=(3, 3), padding=(1, 1), data_layout="NCHW16c"
        )
        ret = relay.layout_transform(y, "NCHW16c", "NCHW")
        ret = relay.sum(ret, axis=[1], keepdims=True)
        ret = relay.layout_transform(ret, "NCHW", "NHWC")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_nhwc_arm():
    """Check that AlterOplayout does not alter NHWC data layout."""

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        from tvm import topi

        with tvm.target.Target("llvm -device=arm_cpu"):
            return topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, out_type)

    # Check NHWC conversion.
    def before_nhwc():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight1 = relay.var("weight1", shape=(3, 3, 64, 64))
        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        y = relay.nn.conv2d(
            x, weight1, channels=64, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWIO"
        )
        y = relay.nn.relu(y)
        y = relay.nn.avg_pool2d(y, pool_size=(1, 1), layout="NHWC")
        y = relay.nn.conv2d(
            y, weight2, channels=64, kernel_size=(3, 3), data_layout="NHWC", kernel_layout="HWIO"
        )
        y = relay.nn.relu(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected_nhwc():
        return before_nhwc()

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_layout_nhwc_int8_aarch64():
    """Check that AlterOplayout does not alter NHWC data layout."""
    from tvm import autotvm

    expected_workload_shape = (20, 42, 4, 16)

    # We use Int8Fallback  to disable the fallback flag
    # and to test the new workload produced during the pass
    class Int8Fallback(autotvm.FallbackContext):
        def _query_inside(self, target, workload):
            key = (target, workload)
            if key in self.memory:
                return self.memory[key]
            cfg = autotvm.task.space.FallbackConfigEntity()
            cfg.is_fallback = False
            cfg.cost = 0
            self.memory[key] = cfg
            return cfg

        def update(self, target, workload, cfg):
            key = (str(target), workload)
            assert workload[2][1] == expected_workload_shape
            assert workload[0] == "conv2d_NHWC_quantized_interleaved_without_transform.arm_cpu"
            self.memory[key] = cfg

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        from tvm import topi

        with tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"):
            with Int8Fallback():
                tmp = topi.nn.conv2d_alter_layout(attrs, inputs, tinfos, out_type)
                return tmp

    # Check NHWC conversion.
    def before_nhwc_int8():
        x = relay.var("x", shape=(1, 56, 56, 73), dtype="int8")
        weight = relay.var("weight1", shape=(3, 3, 73, 79), dtype="int8")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=79,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected_nhwc_int8():
        x = relay.var("x", shape=(1, 56, 56, 73), dtype="int8")
        weight = relay.var("weight1", shape=(3, 3, 73, 79), dtype="int8")
        tile_rows = 4
        tile_cols = 16
        weight_transformed = relay.nn.contrib_conv2d_gemm_weight_transform(
            weight, tile_rows, tile_cols
        )
        y = relay.nn.contrib_conv2d_gemm_without_weight_transform(
            x,
            weight_transformed,
            channels=79,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="HWIO",
            out_dtype="int32",
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before_nhwc_int8()
        a = run_opt_pass(a, transform.AlterOpLayout())
        b = run_opt_pass(expected_nhwc_int8(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_alter_op_with_global_var():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        mod = tvm.IRModule()
        foo = relay.GlobalVar("foo")
        mod[foo] = relay.Function([x, weight], y)
        mod = transform.InferType()(mod)
        mod["main"] = relay.Function([x, weight], foo(x, weight))
        mod = transform.InferType()(mod)
        return mod

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            relay.multiply(weight, relay.const(2.0, "float32")),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
        )
        y = relay.nn.relu(y)
        mod = tvm.IRModule()
        foo = relay.GlobalVar("foo")
        mod[foo] = relay.Function([x, weight], y)
        mod = transform.InferType()(mod)
        mod["main"] = relay.Function([x, weight], foo(x, weight))
        return mod

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = before()
        a = transform.AlterOpLayout()(a)
        b = transform.InferType()(expected())

    assert tvm.ir.structural_equal(a, b, map_free_vars=True), "Actual = \n" + str(a)


def test_alter_op_dense():
    def before():
        x = relay.var("x", shape=(32, 64))
        weight = relay.var("weight", shape=(48, 64))
        y = relay.nn.dense(x, weight)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(32, 64))
        weight = relay.var("weight", shape=(48, 64))
        target_layout = "NK16n"
        weight_transform = relay.layout_transform(weight, "NK", target_layout)
        y = relay.nn.contrib_dense_pack(x, weight_transform, units=None, out_dtype="float32")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    target = "llvm"
    with tvm.target.Target(target):
        with TempOpAttr(
            "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
        ):
            a = before()
            a = run_opt_pass(a, transform.AlterOpLayout())
            b = run_opt_pass(expected(), transform.InferType())
            assert tvm.ir.structural_equal(a, b)


def test_not_inplace_modify():
    def func():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0])
        y = relay.Function([x, weight], y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "OIHW16i"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        before = func()
        run_opt_pass(before, [transform.AlterOpLayout()])
        assert before.body.attrs.layout == "NCHW"


if __name__ == "__main__":
    test_alter_op()
    test_alter_return_none()
    test_alter_layout()
    test_alter_layout_dual_path()
    test_alter_layout_lrn()
    test_alter_layout_resnet()
    test_alter_layout_broadcast_op()
    test_alter_layout_broadcast_scalar_op()
    test_alter_layout_scalar()
    test_alter_layout_concatenate()
    test_alter_layout_nchw_upsamping_op()
    test_alter_layout_strided_slice()
    test_alter_layout_depthwise_conv2d()
    test_alter_layout_prelu()
    test_alter_layout_pad()
    test_alter_layout_pool()
    test_alter_layout_sum()
    test_alter_layout_nhwc_arm()
    test_alter_layout_nhwc_int8_aarch64()
    test_alter_op_with_global_var()
    test_alter_op_dense()
    test_alter_layout_strided_slice_axes_nhwc()
    test_not_inplace_modify()
