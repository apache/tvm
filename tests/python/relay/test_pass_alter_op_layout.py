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
from tvm.relay import testing


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


def test_alter_layout_multi():
    """Test alternating the layout of a conv2d.
    The layout of broadcast operators and the weight should be changed accordingly.
    """

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight")
        y = relay.nn.conv2d(x, weight, channels=128, kernel_size=(3, 3), padding=(1, 1))
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW16c"
        new_attrs["kernel_layout"] = "OHWI16i64o2i"
        return relay.nn.conv2d(data, weight, **new_attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(128, 64, 3, 3))

        y = relay.layout_transform(x, "NCHW", "NCHW16c")
        w = relay.layout_transform(weight, "OIHW", "OHWI16i64o2i")
        y = relay.nn.conv2d(
            y,
            w,
            channels=128,
            kernel_size=(3, 3),
            padding=(1, 1),
            kernel_layout="OHWI16i64o2i",
            data_layout="NCHW16c",
        )
        y = relay.layout_transform(y, "NCHW16c", "NCHW")
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
        return relay.Function(analysis.free_vars(y), y)

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
        mean = relay.mean(add, axis=[1, 4], exclude=True)
        var = relay.variance(add, axis=[1, 4], exclude=True)
        denom = relay.const(1.0) / relay.sqrt(var + relay.const(1e-05))
        gamma = relay.var("gamma", shape=(16,))
        denom_c16c = denom * relay.layout_transform(gamma, src_layout="C", dst_layout="C16c")
        denom = relay.layout_transform(denom_c16c, src_layout="C16c", dst_layout="C")
        denom_expand1 = relay.expand_dims(denom, axis=1, num_newaxis=2)
        denom_expand2 = relay.expand_dims(denom_expand1, axis=0)
        denom_nchwc16 = relay.layout_transform(
            denom_expand2, src_layout="NCHW", dst_layout="NCHW16c"
        )
        out = add * denom_nchwc16
        beta = relay.var("beta", shape=(16,))
        numerator_c16c = (-mean) * denom_c16c + relay.layout_transform(
            beta, src_layout="C", dst_layout="C16c"
        )
        numerator = relay.layout_transform(numerator_c16c, src_layout="C16c", dst_layout="C")
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


@tvm.testing.parametrize_targets("llvm")
def test_alter_layout_strided_slice(target, dev):
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
        for kind in ["graph", "debug", "vm"]:
            np_data = np.random.uniform(size=(1, 32, 28, 28)).astype("float32")
            np_weight = np.random.uniform(size=(32, 32, 3, 3)).astype("float32")
            f_before = relay.create_executor(
                kind, mod=mod_before, device=dev, target=target
            ).evaluate()
            result_before = f_before(np_data, np_weight)
            f_new = relay.create_executor(kind, mod=mod_new, device=dev, target=target).evaluate()
            result_new = f_new(np_data, np_weight)
            tvm.testing.assert_allclose(
                result_before.numpy(), result_new.numpy(), rtol=1e-5, atol=1e-5
            )


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
        with tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mcpu=core-avx2"):
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
        ret = relay.sum(y, axis=[1, 4], keepdims=True)
        ret = relay.layout_transform(ret, "NCHW1c", "NCHW")
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
        ret = relay.sum(y, axis=[1, 4], keepdims=True)
        ret = relay.layout_transform(ret, "NCHW1c", "NHWC")
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

    expected_workload_shape = (20, 44, 4, 16)

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
        x = relay.var("x", shape=(32, 1, 128))
        weight = relay.var("weight", shape=(48, 64))
        avg1d = relay.nn.adaptive_avg_pool1d(x, [64])
        squeeze = relay.squeeze(avg1d, axis=[1])
        y = relay.nn.dense(squeeze, weight)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(32, 1, 128))
        weight = relay.var("weight", shape=(48, 64))
        target_layout = "NC16n"
        weight_transform = relay.layout_transform(weight, "NC", target_layout)
        avg1d = relay.nn.adaptive_avg_pool1d(x, [64])
        squeeze = relay.squeeze(avg1d, axis=[1])
        y = relay.nn.contrib_dense_pack(
            squeeze, weight_transform, target_layout, units=None, out_dtype="float32"
        )
        y = relay.Function(analysis.free_vars(y), y)
        return y

    target = "llvm -mtriple=x86_64-linux-gnu -mcpu=core-avx2"
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


def test_alter_op_dense_packed_data():
    def before():
        x = relay.var("x", shape=(1, 32, 8, 8))
        weight = relay.var("conv2d_weight", shape=(32, 32, 3, 3))
        conv = relay.nn.conv2d(x, weight, channels=32, kernel_size=(3, 3), padding=(1, 1))
        pool = relay.nn.avg_pool2d(conv, pool_size=[8, 8], padding=[0, 0, 0, 0])
        squeeze = relay.squeeze(pool, axis=[2, 3])
        dense = relay.nn.dense(squeeze, relay.var("dense_weight", shape=(16, 32)))
        return relay.Function(analysis.free_vars(dense), dense)

    def expected():
        x = relay.var("x", shape=(1, 32, 8, 8))
        conv_weight = relay.var("conv2d_weight", shape=(32, 32, 3, 3))
        dense_weight = relay.var("dense_weight", shape=(16, 32))
        conv = relay.nn.contrib_conv2d_nchwc(
            relay.layout_transform(x, "NCHW", "NCHW8c"),
            relay.layout_transform(conv_weight, "OIHW", "OIHW8i8o"),
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW8c",
            kernel_layout="OIHW8i8o",
            out_layout="NCHW8c",
        )
        pool = relay.nn.avg_pool2d(conv, pool_size=[8, 8], padding=[0, 0, 0, 0], layout="NCHW8c")
        squeeze = relay.squeeze(pool, axis=[2, 3])
        dense = relay.nn.contrib_dense_pack(
            relay.layout_transform(squeeze, "NC8c", "NC"),
            relay.layout_transform(dense_weight, "NC", "NC16n"),
            "NC16n",
            out_dtype="float32",
        )
        return relay.Function(analysis.free_vars(dense), dense)

    with tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mcpu=core-avx2"):
        with TempOpAttr(
            "nn.dense", "FTVMAlterOpLayout", topi.x86.dense_alter_op._alter_dense_layout
        ):
            a = run_opt_pass(before(), transform.AlterOpLayout())
            b = run_opt_pass(expected(), transform.InferType())
            assert tvm.ir.structural_equal(a, b)


def test_conv2d_strided_slice_packed_to_unpacked():
    """We do not support propagating through packed to unpacked layout"""
    x_shape = (1, 1, 1, 1, 4)
    w_shape = (9, 1, 3, 3, 4, 4)

    def before():
        x = relay.var("x", shape=x_shape)
        weight = relay.var("weight", shape=w_shape)
        y = relay.nn.conv2d(
            x,
            weight,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
        )
        y = relay.strided_slice(y, begin=[0, 0], end=[1, -1], strides=[1, 8])
        return relay.Function([x, weight], y)

    def expected():
        x = relay.var("x", shape=x_shape)
        weight = relay.var("weight", shape=w_shape)
        x_nchw = relay.layout_transform(x, src_layout="NCHW4c", dst_layout="NCHW")
        weight_oihw = relay.layout_transform(weight, src_layout="OIHW4i4o", dst_layout="OIHW")
        y = relay.nn.conv2d(
            x_nchw,
            weight_oihw,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        y = relay.layout_transform(y, src_layout="NCHW", dst_layout="NCHW4c")
        y = relay.strided_slice(y, begin=[0, 0], end=[1, -1], strides=[1, 8])
        return relay.Function([x, weight], y)

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW"
        new_attrs["kernel_layout"] = "OIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b)


def test_conv2d_strided_slice_arbitrary_stride():
    """Test rewriting strided_slice with arbitrary stride"""

    def before():
        x = relay.var("x", shape=(4, 12, 1, 1))
        weight = relay.var("weight", shape=(9, 12, 1, 1))
        y = relay.nn.conv2d(x, weight, channels=9, kernel_size=(1, 1), padding=(0, 0))
        y = relay.strided_slice(y, begin=[3], end=[6], strides=[3], axes=[1])
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW3c"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        run_opt_pass(before(), transform.AlterOpLayout())


def test_conv2d_reduce_channels():
    x = relay.var("data", shape=(1, 8, 48, 48))
    y = relay.nn.conv2d(
        data=x,
        weight=relay.var("weight"),
        kernel_size=(1, 1),
        channels=8,
        dilation=1,
        strides=(47, 47),
    )
    z = relay.argmin(y, axis=1)

    mod, params = testing.create_workload(z)

    with tvm.transform.PassContext(opt_level=3):
        relay.build(mod, params=params, target="llvm")


def test_alter_layout_nonscalar_broadcast():
    """Test boradcast operators"""

    def before():
        x = relay.var("x", shape=(1, 16, 3, 3))
        weight = relay.var("weight", shape=(16, 16, 1, 1))
        y = relay.nn.conv2d(
            x, weight, channels=16, kernel_size=(1, 1), padding=(0, 0), data_layout="NCHW"
        )
        z = relay.var("z", shape=(1, 3, 3))
        y = y + z
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 16, 3, 3))
        weight = relay.var("weight", shape=(16, 16, 1, 1))
        x = relay.layout_transform(x, src_layout="NCHW", dst_layout="NCHW4c")
        weight = relay.layout_transform(weight, src_layout="OIHW", dst_layout="OIHW4i4o")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=16,
            kernel_size=(1, 1),
            padding=(0, 0),
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
        )
        z = relay.var("z", shape=(1, 3, 3))
        z = relay.expand_dims(z, 0)
        z = relay.layout_transform(z, src_layout="NCHW", dst_layout="NCHW1c")
        y = y + z
        y = relay.layout_transform(y, src_layout="NCHW4c", dst_layout="NCHW")
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW4c"
        new_attrs["kernel_layout"] = "OIHW4i4o"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpected = \n" + str(b)

    inp = np.random.uniform(size=(1, 16, 3, 3)).astype(np.float32)
    weight = np.random.uniform(size=(16, 16, 1, 1)).astype(np.float32)
    z = np.random.uniform(size=(1, 3, 3)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight, z)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight, z)
    np.testing.assert_allclose(res.numpy(), res1.numpy())


def test_alter_layout_blocked_no_broadcast():
    """Test boradcast operators working on already blocked layout"""

    def before():
        dtype = "float32"
        input_shape = (1, 8, 16, 16, 4)
        filter_shape = (1, 8, 4, 4, 4, 4)
        bias_shape = (1, 1, 1, 1, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        conv = relay.nn.conv2d(
            A,
            B,
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
            padding=[3, 3, 0, 0],
            strides=[2, 2],
            out_dtype=dtype,
            channels=4,
            kernel_size=(4, 4),
        )
        bias = relay.op.add(conv, C)
        bias = relay.Function(analysis.free_vars(bias), bias)
        return bias

    def expected():
        return before()

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW4c"
        new_attrs["kernel_layout"] = "OIHW4i4o"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpected = \n" + str(b)

    inp = np.random.uniform(size=(1, 8, 16, 16, 4)).astype(np.float32)
    weight = np.random.uniform(size=(1, 8, 4, 4, 4, 4)).astype(np.float32)
    z = np.random.uniform(size=(1, 1, 1, 1, 4)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight, z)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight, z)
    np.testing.assert_allclose(res.numpy(), res1.numpy())


def test_alter_layout_blocked_broadcast():
    """Test boradcast operators working on already blocked layout"""

    def before():
        dtype = "float32"
        input_shape = (1, 8, 16, 16, 4)
        filter_shape = (1, 8, 4, 4, 4, 4)
        bias_shape = (1, 1, 1, 1, 1)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        conv = relay.nn.conv2d(
            A,
            B,
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
            padding=[3, 3, 0, 0],
            strides=[2, 2],
            out_dtype=dtype,
            channels=4,
            kernel_size=(4, 4),
        )
        bias = relay.op.add(conv, C)
        bias = relay.Function(analysis.free_vars(bias), bias)
        return bias

    def expected():
        return before()

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW4c"
        new_attrs["kernel_layout"] = "OIHW4i4o"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpected = \n" + str(b)

    inp = np.random.uniform(size=(1, 8, 16, 16, 4)).astype(np.float32)
    weight = np.random.uniform(size=(1, 8, 4, 4, 4, 4)).astype(np.float32)
    z = np.random.uniform(size=(1, 1, 1, 1, 1)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight, z)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight, z)
    np.testing.assert_allclose(res.numpy(), res1.numpy())


def test_alter_layout_re_blocking_broadcast():
    """Test of re-blocking shapes with boradcast operators"""

    def before():
        dtype = "float32"
        input_shape = (1, 8, 16, 16, 4)
        filter_shape = (1, 8, 4, 4, 4, 4)
        bias_shape = (1, 1, 1, 1, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        conv = relay.nn.conv2d(
            A,
            B,
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
            padding=[3, 3, 0, 0],
            strides=[2, 2],
            out_dtype=dtype,
            channels=4,
            kernel_size=(4, 4),
        )
        bias = relay.op.add(conv, C)
        bias = relay.Function(analysis.free_vars(bias), bias)
        return bias

    def expected():
        dtype = "float32"
        input_shape = (1, 8, 16, 16, 4)
        filter_shape = (1, 8, 4, 4, 4, 4)
        bias_shape = (1, 1, 1, 1, 4)
        A = relay.var("data", shape=input_shape, dtype=dtype)
        B = relay.var("weight", shape=filter_shape, dtype=dtype)
        C = relay.var("bias", shape=bias_shape, dtype=dtype)

        A = relay.layout_transform(A, src_layout="NCHW4c", dst_layout="NCHW2c")
        B = relay.layout_transform(B, src_layout="OIHW4i4o", dst_layout="OIHW2i2o")

        conv = relay.nn.conv2d(
            A,
            B,
            data_layout="NCHW2c",
            kernel_layout="OIHW2i2o",
            padding=[3, 3, 0, 0],
            strides=[2, 2],
            out_dtype=dtype,
            channels=4,
            kernel_size=(4, 4),
        )
        C = relay.layout_transform(C, src_layout="NCHW4c", dst_layout="NCHW2c")
        bias = relay.op.add(conv, C)
        bias = relay.layout_transform(bias, src_layout="NCHW2c", dst_layout="NCHW4c")
        bias = relay.Function(analysis.free_vars(bias), bias)
        return bias

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW2c"
        new_attrs["kernel_layout"] = "OIHW2i2o"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpected = \n" + str(b)

    inp = np.random.uniform(size=(1, 8, 16, 16, 4)).astype(np.float32)
    weight = np.random.uniform(size=(1, 8, 4, 4, 4, 4)).astype(np.float32)
    z = np.random.uniform(size=(1, 1, 1, 1, 4)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight, z)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight, z)
    np.testing.assert_allclose(res.numpy(), res1.numpy(), rtol=1e-5, atol=1e-5)


def test_broadcast_non_adaptable():
    """NCHW4c + [x, x, 4] and NCHW4c is being altered to NCHW"""

    def before():
        x = relay.var("x", shape=(1, 4, 3, 3, 4))
        weight = relay.var("weight", shape=(4, 4, 1, 1, 4, 4))
        y = relay.nn.conv2d(
            x,
            weight,
            channels=16,
            kernel_size=(1, 1),
            padding=(0, 0),
            data_layout="NCHW4c",
            kernel_layout="OIHW4i4o",
        )
        z = relay.var("z", shape=(3, 3, 4))
        y = y + z
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected():
        x = relay.var("x", shape=(1, 4, 3, 3, 4))
        weight = relay.var("weight", shape=(4, 4, 1, 1, 4, 4))
        x = relay.layout_transform(x, src_layout="NCHW4c", dst_layout="NCHW")
        weight = relay.layout_transform(weight, src_layout="OIHW4i4o", dst_layout="OIHW")
        y = relay.nn.conv2d(
            x,
            weight,
            channels=16,
            kernel_size=(1, 1),
            padding=(0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        z = relay.var("z", shape=(3, 3, 4))
        y = relay.layout_transform(y, src_layout="NCHW", dst_layout="NCHW4c")
        y = y + z
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW"
        new_attrs["kernel_layout"] = "OIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)

    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        a = run_opt_pass(before(), transform.AlterOpLayout())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "\nExpected = \n" + str(b)

    inp = np.random.uniform(size=(1, 4, 3, 3, 4)).astype(np.float32)
    weight = np.random.uniform(size=(4, 4, 1, 1, 4, 4)).astype(np.float32)
    z = np.random.uniform(size=(3, 3, 4)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight, z)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight, z)
    np.testing.assert_allclose(res.numpy(), res1.numpy())


def test_broadcast_respect_input_layouts():
    def before():
        x = relay.var("x", shape=(1, 16, 1, 1))
        w = relay.var("w", shape=(16, 16, 1, 1))
        x = relay.nn.conv2d(
            x,
            w,
            kernel_size=(1, 1),
            padding=(0, 0),
            channels=16,
        )
        y1 = relay.min(x, axis=[2])
        y2 = relay.min(x, axis=[3])
        z = y1 + y2
        z = relay.Function(analysis.free_vars(z), z)
        return z

    def alter_conv2d(attrs, inputs, tinfos, out_type):
        data, weight = inputs
        new_attrs = dict(attrs)
        new_attrs["data_layout"] = "NCHW4c"
        new_attrs["kernel_layout"] = "OIHW4i4o"
        return relay.nn.conv2d(data, weight, **new_attrs)

    inp = np.random.uniform(size=(1, 16, 1, 1)).astype(np.float32)
    weight = np.random.uniform(size=(16, 16, 1, 1)).astype(np.float32)
    mod = tvm.IRModule.from_expr(before())
    with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv2d):
        with tvm.transform.PassContext(opt_level=4):
            res = relay.build_module.create_executor(
                "graph", mod, target="llvm", device=tvm.cpu()
            ).evaluate()(inp, weight)
    with tvm.transform.PassContext(opt_level=0):
        res1 = relay.build_module.create_executor(
            "debug", mod, target="llvm", device=tvm.cpu()
        ).evaluate()(inp, weight)
    np.testing.assert_allclose(res.numpy(), res1.numpy())


def test_axis_semantic_change():
    x = relay.var("x", shape=(1, 1, 24, 48))
    w1 = relay.const(np.random.uniform(size=(1, 1, 1, 1)))
    w2 = relay.const(np.random.uniform(size=(1, 1, 1, 1)))
    y = relay.nn.conv2d(x, w1, kernel_size=(1, 1), padding=(0, 0), channels=1)
    y = relay.transpose(y, (0, 1, 3, 2))
    z = relay.nn.conv2d(y, w2, kernel_size=(1, 1), padding=(0, 0), channels=1)
    func = relay.Function([x], z)
    mod = tvm.IRModule.from_expr(func)
    with tvm.transform.PassContext(opt_level=3):
        relay.build(mod, target="llvm")


def test_alter_with_subfunc():
    v1 = relay.var("v", shape=[1, 256, 10, 10], dtype="float32")
    v2 = relay.image.resize2d(v1, size=[16, 16], roi=[0.0, 0.0, 0.0, 0.0], rounding_method="")
    sub_func = relay.Function([v1], v2)
    x1 = relay.var("x", shape=[1, 256, 10, 10], dtype="float32")
    x2 = sub_func(x1)
    x3 = relay.image.resize2d(x2, size=[8, 8], roi=[0.0, 0.0, 0.0, 0.0], rounding_method="")
    func = relay.Function([x1], x3)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(relay.transform.AlterOpLayout()(mod), mod)


def test_alter_with_reduce():
    x = relay.var("x", shape=(1, 1, 1, 1))
    y = relay.image.resize2d(x, (2, 4))
    z = relay.mean(y, axis=0)
    a = relay.image.resize1d(z, (1,))
    func = relay.Function((x,), a)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    with tvm.transform.PassContext(opt_level=4):
        relay.build(mod, target="llvm")


if __name__ == "__main__":
    tvm.testing.main()
