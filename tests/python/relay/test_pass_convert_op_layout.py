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


def test_qnn_binary_no_convert_layout():
    def before():
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(1, 2))
        return relay.Function(
            [x, y],
            relay.qnn.op.add(
                x,
                y,
                lhs_scale=relay.const(0.0156863, "float32"),
                lhs_zero_point=relay.const(127, "int32"),
                rhs_scale=relay.const(0.0117647, "float32"),
                rhs_zero_point=relay.const(85, "int32"),
                output_scale=relay.const(0.0235294, "float32"),
                output_zero_point=relay.const(128, "int32"),
            ),
        )

    def expected():
        return before()

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({}))
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


def test_deformable_conv_bias_pool_convert_layout():
    def before(N, CI, H, W, CO, KH, KW, layout):
        if layout == "NCHW":
            data_shape = (N, CI, H, W)
            weight_shape = (CO, CI, KH, KW)
            kernel_layout = "OIHW"
        else:
            data_shape = (N, H, W, CI)
            weight_shape = (KH, KW, CI, CO)
            kernel_layout = "HWIO"
        bias_shape = (CO,)

        data = relay.var("data", shape=data_shape, dtype="float32")
        offset = relay.var("offset")
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout,
            kernel_layout=kernel_layout,
        )
        y = relay.nn.bias_add(y, bias, axis=-1 if layout == "NHWC" else 1)
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout=layout)
        y = relay.cast(y, "int32")
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    def expected(N, CI, H, W, CO, KH, KW, OH, OW, src_layout, dst_layout):
        layout_map = {"src": {}, "dst": {}}
        if src_layout == "NCHW":
            nchw = layout_map["src"]
            nhwc = layout_map["dst"]
        else:
            nchw = layout_map["dst"]
            nhwc = layout_map["src"]

        nchw["data_layout"] = "NCHW"
        nchw["data_shape"] = (N, CI, H, W)
        nchw["offset_shape"] = (N, KH * KW * 2, OH, OW)
        nchw["weight_shape"] = (CO, CI, KH, KW)
        nchw["kernel_layout"] = "OIHW"

        nhwc["data_layout"] = "NHWC"
        nhwc["data_shape"] = (N, H, W, CI)
        nhwc["offset_shape"] = (N, OH, OW, KH * KW * 2)
        nhwc["weight_shape"] = (KH, KW, CI, CO)
        nhwc["kernel_layout"] = "HWIO"

        bias_shape = (CO,)

        data = relay.var("data", shape=layout_map["src"]["data_shape"], dtype="float32")
        offset = relay.var("offset", shape=layout_map["src"]["offset_shape"], dtype="float32")
        weight = relay.var("weight", shape=layout_map["src"]["weight_shape"], dtype="float32")
        bias = relay.var("bias", shape=bias_shape, dtype="float32")

        data = relay.layout_transform(
            data, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        offset = relay.layout_transform(
            offset, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        weight = relay.layout_transform(
            weight, layout_map["src"]["kernel_layout"], layout_map["dst"]["kernel_layout"]
        )
        y = relay.nn.deformable_conv2d(
            data,
            offset,
            weight,
            kernel_size=(KH, KW),
            channels=CO,
            data_layout=layout_map["dst"]["data_layout"],
            kernel_layout=layout_map["dst"]["kernel_layout"],
        )
        if layout_map["src"]["data_layout"] == "NHWC":
            bias = relay.expand_dims(bias, axis=0, num_newaxis=3)
        else:
            bias = relay.expand_dims(bias, axis=1, num_newaxis=2)
            bias = relay.expand_dims(bias, axis=0)
        bias = relay.layout_transform(
            bias, layout_map["src"]["data_layout"], layout_map["dst"]["data_layout"]
        )
        y = relay.add(y, bias)
        y = relay.nn.relu(y)
        y = relay.nn.max_pool2d(y, pool_size=(2, 2), layout=layout_map["dst"]["data_layout"])
        y = relay.cast(y, "int32")
        y = relay.layout_transform(
            y, layout_map["dst"]["data_layout"], layout_map["src"]["data_layout"]
        )
        y = relay.nn.batch_flatten(y)
        y = relay.Function(analysis.free_vars(y), y)
        return y

    # NHWC -> NCHW
    a = before(1, 3, 224, 224, 32, 3, 3, "NHWC")
    a = run_opt_pass(a, transform.ConvertLayout({"nn.deformable_conv2d": ["NCHW", "default"]}))
    b = run_opt_pass(
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NHWC", "NCHW"), transform.InferType()
    )
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    # NCHW -> NHWC
    a = before(1, 3, 224, 224, 32, 3, 3, "NCHW")
    a = run_opt_pass(a, transform.ConvertLayout({"nn.deformable_conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(
        expected(1, 3, 224, 224, 32, 3, 3, 222, 222, "NCHW", "NHWC"), transform.InferType()
    )
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


def test_slice_like_convert_layout():
    def verify_slice_like(after, expected_axes):
        # Verify if the slice_like after the convert layout has the expected axes.
        has_expected = list()
        checker = lambda x: has_expected.append(
            isinstance(x, tvm.relay.expr.Call)
            and x.op.name == "slice_like"
            and str(x.attrs.axes) == str(expected_axes)
        )
        relay.analysis.post_order_visit(after, checker)
        assert any(has_expected)

    def func_nhwc():
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
        out = relay.slice_like(y, y, axes=[1, 2])
        return relay.Function(analysis.free_vars(out), out)

    after = run_opt_pass(func_nhwc(), transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    verify_slice_like(after, [2, 3])

    def func_nchw():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        out = relay.slice_like(y, y, axes=[2, 3])
        return relay.Function(analysis.free_vars(out), out)

    after = run_opt_pass(func_nchw(), transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    verify_slice_like(after, [1, 2])

    def func_vars():
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
        # z has no layout information so convert layout won't happen.
        z = relay.var("y", shape=(1, 56, 56, 32))
        out = relay.slice_like(y, z, axes=[1, 2])
        return relay.Function(analysis.free_vars(out), out)

    after = run_opt_pass(func_vars(), transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
    verify_slice_like(after, [1, 2])


def test_transpose_convert_layout():
    def verify_transpose(after, expected_axes, expected_transform_cnt):
        # Verify if the transpose after the convert layout has the expected axes.
        has_expected = list()
        checker = lambda x: has_expected.append(
            isinstance(x, tvm.relay.expr.Call)
            and x.op.name == "transpose"
            and str(x.attrs.axes) == str(expected_axes)
        )
        relay.analysis.post_order_visit(after, checker)
        assert any(has_expected), after

        is_transform = list()
        checker = lambda x: is_transform.append(
            1 if isinstance(x, tvm.relay.expr.Call) and x.op.name == "layout_transform" else 0
        )
        relay.analysis.post_order_visit(after, checker)
        assert (
            sum(is_transform) == expected_transform_cnt
        ), "Expected %s layout_transform, but get\n%s" % (expected_transform_cnt, after)

    def nhwc_to_nchw():
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
        z = relay.var("z", shape=(56, 56, 32))
        out = relay.add(y, z)
        out = relay.transpose(out, axes=[0, 3, 1, 2])
        out = relay.nn.batch_flatten(out)
        func = relay.Function(analysis.free_vars(out), out)
        return run_opt_pass(func, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))

    verify_transpose(nhwc_to_nchw(), [0, 1, 2, 3], 3)

    def nchw_to_nhwc():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        z = relay.var("z", shape=(32, 56, 56))
        out = relay.add(y, z)
        out = relay.transpose(out, axes=[0, 2, -1, 1])  # Also test a negative axis.
        out = relay.nn.batch_flatten(out)
        func = relay.Function(analysis.free_vars(out), out)
        return run_opt_pass(func, transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))

    verify_transpose(nchw_to_nhwc(), [0, 1, 2, 3], 3)

    def default_axes():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight1 = relay.var("weight1", shape=(32, 64, 3, 3))
        y = relay.nn.conv2d(
            x,
            weight1,
            channels=32,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        z = relay.var("z", shape=(32, 56, 56))
        out = relay.add(y, z)
        out = relay.transpose(out)  # No axes provided, will use the reversed axes.
        func = relay.Function(analysis.free_vars(out), out)
        return run_opt_pass(func, transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))

    verify_transpose(default_axes(), [2, 1, 3, 0], 3)


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


def test_conv_strided_slice_convert_layout():
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
        y = relay.strided_slice(y, begin=[0, 1], end=[1, -1, 10], strides=[1, 1, 2, 1])
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
        y = relay.strided_slice(y, begin=[0, 0, 0, 1], end=[1, 10, 56, -1], strides=[1, 2, 1, 1])
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(relay.analysis.free_vars(y), y)
        return y

    a = before()
    a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_roi_pool_convert_layout():
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
        y = relay.vision.roi_pool(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, layout="NCHW"
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
        y = relay.vision.roi_pool(
            y, rois, pooled_size=(14, 14), spatial_scale=0.0625, layout="NHWC"
        )
        ret = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.Function(analysis.free_vars(ret), ret)
        return y

    a = before()
    desired_layouts = {
        "nn.conv2d": ["NHWC", "HWIO"],
        "vision.roi_pool": ["NHWC", "default"],
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


def test_convert_with_config():
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

        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        y2 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )
        y2 = relay.nn.relu(y2)

        out = relay.Function([x, weight, weight2], y2)
        return out

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64))
        weight = relay.var("weight", shape=(3, 3, 64, 64))

        weight2 = relay.var("weight2", shape=(3, 3, 64, 64))
        weight2 = relay.layout_transform(weight2, "HWIO", "HWOI")

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
        y = relay.layout_transform(y, "NHWC", "HWNC")

        y2 = relay.nn.conv2d(
            y,
            weight2,
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="HWNC",
            kernel_layout="HWOI",
        )
        y2 = relay.nn.relu(y2)

        y2 = relay.layout_transform(y2, "HWNC", "NHWC")
        output = relay.Function(relay.analysis.free_vars(y2), y2)
        return output

    a = before()
    layout_config = relay.transform.LayoutConfig(skip_layers=[0])
    with layout_config:
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["HWNC", "default"]}))
    b = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_conv_squeeze_convert_layout():
    def _test_conv_squeeze_convert_layout1():
        # specified axis is squeezed
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[-3])
            return relay.Function(analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[2])
            y = relay.layout_transform(y, "NCW", "NWC")
            return relay.Function(analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    def _test_conv_squeeze_convert_layout2():
        # all axes of dimension 1 are squeezed
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y)
            return relay.Function(analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, [0, 2, 3])
            return relay.Function(analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    def _test_conv_squeeze_convert_layout3():
        # squeeze axis is empty
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[])
            return relay.Function(analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.squeeze(y, axis=[])
            y = relay.layout_transform(y, "NCHW", "NHWC")
            return relay.Function(analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    _test_conv_squeeze_convert_layout1()
    _test_conv_squeeze_convert_layout2()
    _test_conv_squeeze_convert_layout3()


def test_conv_reduce_convert_layout():
    def _test_conv_reduce_convert_layout1():
        def before():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=1000,
                kernel_size=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.nn.relu(y)
            y = relay.sum(y, axis=(1, 2))
            y = relay.sum(y, axis=(1,))
            y = relay.sum(y)
            y = relay.sum(y)
            return relay.Function(analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 1, 1, 2048))
            weight = relay.var("weight", shape=(1, 1, 2048, 1000))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=1000, kernel_size=(1, 1))
            y = relay.nn.relu(y)
            y = relay.sum(y, axis=(2, 3))
            y = relay.sum(y, axis=(1,))
            y = relay.sum(y)
            y = relay.sum(y)
            return relay.Function(analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    def _test_conv_reduce_convert_layout2():
        def _set_span(y, text):
            return relay.Call(
                y.op, y.args, y.attrs, y.type_args, relay.Span(relay.SourceName(text), 0, 0, 0, 0)
            )

        def before():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            y = relay.nn.conv2d(
                x,
                weight,
                channels=512,
                kernel_size=(3, 3),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = _set_span(y, "SpanConv2D")
            y = relay.nn.relu(y)
            y = _set_span(y, "SpanRelu")
            y = relay.multiply(y, y)
            y = _set_span(y, "SpanMultiply")
            y = relay.sum(y, axis=(3,), keepdims=True)
            y = _set_span(y, "SpanSum")
            return relay.Function(analysis.free_vars(y), y)

        def expected():
            x = relay.var("x", shape=(1, 38, 38, 512))
            weight = relay.var("weight", shape=(3, 3, 512, 512))
            weight = relay.layout_transform(weight, "HWIO", "OIHW")
            x = relay.layout_transform(x, "NHWC", "NCHW")
            y = relay.nn.conv2d(x, weight, channels=512, kernel_size=(3, 3))
            y = relay.nn.relu(y)
            y = relay.multiply(y, y)
            y = relay.sum(y, axis=(1,), keepdims=True)
            y = relay.layout_transform(y, "NCHW", "NHWC")
            return relay.Function(analysis.free_vars(y), y)

        a = before()
        a = run_opt_pass(a, transform.ConvertLayout({"nn.conv2d": ["NCHW", "default"]}))
        assert "SpanConv2D" in a.astext()
        assert "SpanRelu" in a.astext()
        assert "SpanMultiply" in a.astext()
        assert "SpanSum" in a.astext()
        b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

    _test_conv_reduce_convert_layout1()
    _test_conv_reduce_convert_layout2()


if __name__ == "__main__":
    test_qnn_binary_no_convert_layout()
    test_no_convert_layout()
    test_conv_convert_layout()
    test_conv_nhwc_convert_layout()
    test_conv_bias_pool_convert_layout()
    test_conv_concat_convert_layout()
    test_dual_path_convert_layout()
    test_bn_convert_layout()
    test_slice_like_convert_layout()
    test_transpose_convert_layout()
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
    test_conv_roi_pool_convert_layout()
    test_conv_strided_slice_convert_layout()
    test_deformable_conv_bias_pool_convert_layout()
    test_default_keyword()
    test_different_ops_convert_layout()
    test_no_desired_layout()
    test_convert_with_config()
    test_conv_squeeze_convert_layout()
    test_conv_reduce_convert_layout()
