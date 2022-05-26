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
from tvm.relay.testing import run_opt_pass
from tvm import utils

import numpy as np


def test_simplify_conv_pad():
    convs = [relay.nn.conv1d, relay.nn.conv2d, relay.nn.conv3d]

    def validate(ndim, pad_width, pad_value, pad_mode, orig_padding, layout):
        np.random.seed(0)
        if layout[1] == "C":
            shape = [1, 3] + [10] * ndim
            wshape = [8, 3] + [3] * ndim
        elif layout[-1] == "C":
            shape = [1] + [10] * ndim + [3]
            wshape = [8] + [3] * ndim + [3]
        else:
            raise ValueError("This test only supports NC* and N*C")

        x = relay.var("x", shape=shape, dtype="float32")
        w = relay.var("w", shape=wshape, dtype="float32")
        pad = relay.nn.pad(x, pad_width, pad_value, pad_mode)
        if layout[1] == "C":
            conv = convs[ndim - 1](pad, w, padding=orig_padding)
        else:
            conv = convs[ndim - 1](
                pad, w, padding=orig_padding, data_layout=layout, kernel_layout="DHWIO"[3 - ndim :]
            )

        if pad_mode == "constant" and pad_value == 0:
            new_padding = []
            for j in range(2):
                for i in range(len(pad_width)):
                    if layout[i] in ["D", "H", "W"]:
                        new_padding.append(pad_width[i][j])
            for i in range(len(new_padding)):
                new_padding[i] += orig_padding[i]
            if layout[1] == "C":
                after = convs[ndim - 1](x, w, padding=new_padding)
            else:
                after = convs[ndim - 1](
                    x, w, padding=new_padding, data_layout=layout, kernel_layout="DHWIO"[3 - ndim :]
                )
        else:
            after = conv

        zz = run_opt_pass(conv, transform.FoldExplicitPadding())
        expected = run_opt_pass(after, transform.InferType())
        assert tvm.ir.structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(conv)
        mod2 = tvm.IRModule.from_expr(zz)

        op_freqs = relay.analysis.list_op_freqs(mod2)
        assert "nn.pad" not in op_freqs

        with tvm.transform.PassContext():
            func1 = relay.create_executor(
                "vm", mod=mod1, device=tvm.cpu(), target="llvm"
            ).evaluate()
        func2 = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm").evaluate()
        x_np = np.random.rand(*shape).astype("float32")
        w_np = np.random.rand(*wshape).astype("float32")

        result1 = func1(x_np, w_np)
        result2 = func2(x_np, w_np)

        tvm.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5, atol=1e-5)

    for orig_pad in [[0, 0], [2, 0], [0, 2]]:
        for i_pad in [[0, 0], [1, 1], [1, 0]]:
            for ndim in [1, 2, 3]:
                for channels_last in [0, 1]:
                    if channels_last:
                        layout = "NDHWC"
                        layout = layout[0:1] + layout[4 - ndim : 4] + layout[-1:]
                        padding = [[0, 0]] + [i_pad] * ndim + [[0, 0]]
                    else:
                        layout = "NCDHW"
                        layout = layout[0:2] + layout[5 - ndim :]
                        padding = [[0, 0]] * 2 + [i_pad] * ndim

                    validate(ndim, padding, 0, "constant", orig_pad * ndim, layout)
                    # validate(ndim, padding, 1, "edge", orig_pad * ndim, layout)
                    # validate(ndim, padding, 1, "reflect", orig_pad * ndim, layout)
                    # breakpoint()
                    # raise AssertionError('done')


def get_min_value(dtype):
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).min
    elif np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    else:
        raise ValueError("Cannot get min value for dtypes that are not integer or not floating")


def test_simplify_pool_pad():
    max_pools = [relay.nn.max_pool1d, relay.nn.max_pool2d, relay.nn.max_pool3d]
    avg_pools = [relay.nn.avg_pool1d, relay.nn.avg_pool2d, relay.nn.avg_pool3d]
    def validate(pools, ndim, pad_width, pad_value, pad_mode, orig_padding, layout, pool_size, dtype):
        pad_value_const = relay.const(pad_value, dtype=dtype)

        if layout[1] == "C":
            shape = [1, 3] + [10] * ndim
        elif layout[-1] == "C":
            shape = [1] + [10] * ndim + [3]
        else:
            raise ValueError("This test only supports NC* and N*C")

        x = relay.var("x", shape=shape, dtype=dtype)
        pad = relay.nn.pad(x, pad_width, pad_value_const, pad_mode)
        if layout[1] == "C":
            pool = pools[ndim - 1](pad, padding=orig_padding, pool_size=pool_size)
        else:
            pool = pools[ndim - 1](
                pad, padding=orig_padding, layout=layout, pool_size=pool_size
            )

        if pools == max_pools:
            foldable_pad_value = get_min_value(dtype)
        else:
            foldable_pad_value = 0

        if pad_mode == "constant" and pad_value == foldable_pad_value:
            new_padding = []
            for j in range(2):
                for i in range(len(pad_width)):
                    if layout[i] in ["D", "H", "W"]:
                        new_padding.append(pad_width[i][j])
            for i in range(len(new_padding)):
                new_padding[i] += orig_padding[i]
            if layout[1] == "C":
                after = pools[ndim - 1](x, padding=new_padding, pool_size=pool_size)
            else:
                after = pools[ndim - 1](
                    x, padding=new_padding, layout=layout, pool_size=pool_size
                )
        else:
            after = pool

        zz = run_opt_pass(pool, transform.FoldExplicitPadding())
        expected = run_opt_pass(after, transform.InferType())

        assert tvm.ir.structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(pool)
        mod2 = tvm.IRModule.from_expr(zz)

        op_freqs = relay.analysis.list_op_freqs(mod2)
        assert "nn.pad" not in op_freqs

        saver = utils.roofline.SaveLoweredTIR()
        with tvm.transform.PassContext(instruments=[saver]):
            func1 = relay.create_executor(
                "vm", mod=mod1, device=tvm.cpu(), target="llvm"
            ).evaluate()
            # lib = tvm.build(mod1, target="llvm")
            # print("lowered1", saver.functions)
            # print("lowered", tvm.lower(mod1))
        
        saver = utils.roofline.SaveLoweredTIR()
        with tvm.transform.PassContext(instruments=[saver]):
            func2 = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm").evaluate()
        x_np = np.random.rand(*shape).astype(dtype)

        result1 = func1(x_np)
        result2 = func2(x_np)

        tvm.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5, atol=1e-5)

    # validate(avg_pools, 1, [[0, 0], [0, 0], [1, 1]], 0, "constant", [0, 0], "NCW", 2, "float32")

    float_min_val = get_min_value("float32")
    # Test fold cases
    for orig_pad in [[0, 0], [2, 0], [0, 2]]:
        for i_pad in [[0, 0], [1, 1], [1, 0]]:
            for ndim in [1, 2, 3]:
                for channels_last in [0, 1]:
                    if channels_last:
                        layout = "NDHWC"
                        layout = layout[0:1] + layout[4 - ndim : 4] + layout[-1:]
                        padding = [[0, 0]] + [i_pad] * ndim + [[0, 0]]
                    else:
                        layout = "NCDHW"
                        layout = layout[0:2] + layout[5 - ndim :]
                        padding = [[0, 0]] * 2 + [i_pad] * ndim

                    validate(max_pools, ndim, padding, float_min_val, "constant", orig_pad * ndim, layout, 2, "float32")

    int_min_val = get_min_value("int32")
    validate(avg_pools, ndim, padding, 0, "constant", orig_pad * ndim, layout, 2, "float32")
    # Check max pool pad folding with int dtype
    # validate(max_pools, 2, [[0, 0], [1, 1], [0, 0]], int_min_val, "constant", [2, 0], "NCHW", 2, "int32")

    # Test no fold cases


def fold_pad_qconv2d():
    def before():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        input_zero_point = 10
        pad = relay.nn.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], pad_value=input_zero_point)
        return relay.qnn.op.conv2d(
            pad,
            weight,
            relay.const(input_zero_point, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(0, 0),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

    def expected():
        x = relay.var("x", shape=(1, 56, 56, 64), dtype="int8")
        weight = relay.var("weight", shape=(3, 3, 64, 64), dtype="int8")
        input_zero_point = 10
        return relay.qnn.op.conv2d(
            x,
            weight,
            relay.const(input_zero_point, "int32"),
            relay.const(1, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            data_layout="NHWC",
            kernel_layout="HWIO",
        )

    a = run_opt_pass(before(), relay.transform.FoldExplicitPadding())
    b = run_opt_pass(expected(), transform.InferType())

    assert tvm.ir.structural_equal(a, b, map_free_vars=True), "Actual = \n" + str(a)


def test_pad_qconv2d_no_fold():
    def get_expr():
        x = relay.var("x", shape=(1, 1, 2, 2), dtype="int8")
        weight = relay.var("weight", shape=(1, 1, 2, 2), dtype="int8")
        # Pad value and input zp are not equal
        pad_value = 1
        input_zero_point = 0
        pad = relay.nn.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], pad_value=pad_value)
        return relay.qnn.op.conv2d(
            pad,
            weight,
            relay.const(input_zero_point, "int32"),
            relay.const(0, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=1,
            kernel_size=(2, 2),
            padding=(0, 0),
        )

    a = run_opt_pass(get_expr(), relay.transform.FoldExplicitPadding())
    b = run_opt_pass(get_expr(), transform.InferType())

    assert tvm.ir.structural_equal(a, b, map_free_vars=True), (
        "\nActual = \n" + str(a) + "\nExpected = \n" + str(b)
    )


if __name__ == "__main__":
    test_simplify_conv_pad()
    test_simplify_pool_pad()
    fold_pad_qconv2d()
    test_pad_qconv2d_no_fold()
