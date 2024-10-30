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

import numpy as np


def test_simplify_conv_pad():
    convs = [relay.nn.conv1d, relay.nn.conv2d, relay.nn.conv3d]

    def validate(ndim, pad_width, pad_value, pad_mode, orig_padding, layout, no_fold=False):
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
        tvm.ir.assert_structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(conv)
        mod2 = tvm.IRModule.from_expr(zz)

        if not no_fold:
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

                    validate(ndim, padding, 0, "constant", orig_pad * ndim, layout)

    # Test no fold cases
    ndim = 2
    # Conv only folds when pad_value=0
    validate(
        ndim, [[0, 0]] * 2 + [i_pad] * ndim, 1, "constant", orig_pad * ndim, "NCHW", no_fold=True
    )
    # Conv only folds when pad's pad_mode="constant"
    validate(ndim, [[0, 0]] * 2 + [i_pad] * ndim, 0, "edge", orig_pad * ndim, "NCHW", no_fold=True)


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

    def validate(
        pools,
        ndim,
        pad_width,
        pad_value,
        orig_padding,
        layout,
        pool_size,
        pad_mode="constant",
        dtype="float32",
        no_fold=False,
        **kwargs,
    ):
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
            pool = pools[ndim - 1](pad, padding=orig_padding, pool_size=pool_size, **kwargs)
        else:
            pool = pools[ndim - 1](
                pad, padding=orig_padding, layout=layout, pool_size=pool_size, **kwargs
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

            if pools == avg_pools and all(v == 0 for v in orig_padding):
                # If the orig padding for AvgPool is all zero and the pad op to fold
                # has non-zero pad width, the resultant folded AvgPool will have
                # count_include_pad=True so AvgPool's divisor is agnostic of pad boundaries
                kwargs["count_include_pad"] = True
            if layout[1] == "C":
                after = pools[ndim - 1](x, padding=new_padding, pool_size=pool_size, **kwargs)
            else:
                after = pools[ndim - 1](
                    x, padding=new_padding, layout=layout, pool_size=pool_size, **kwargs
                )
        else:
            after = pool

        zz = run_opt_pass(pool, transform.FoldExplicitPadding())
        expected = run_opt_pass(after, transform.InferType())

        tvm.ir.assert_structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(pool)
        mod2 = tvm.IRModule.from_expr(zz)

        if not no_fold:
            op_freqs = relay.analysis.list_op_freqs(mod2)
            assert "nn.pad" not in op_freqs

        with tvm.transform.PassContext():
            func1 = relay.create_executor(
                "vm", mod=mod1, device=tvm.cpu(), target="llvm"
            ).evaluate()

        func2 = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm").evaluate()
        x_np = np.random.rand(*shape).astype(dtype)

        result1 = func1(x_np)
        result2 = func2(x_np)

        tvm.testing.assert_allclose(result1.numpy(), result2.numpy(), rtol=1e-5, atol=1e-5)

    # Test fold cases
    float_min_val = get_min_value("float32")
    for orig_pad in [[0, 0], [2, 0]]:
        for i_pad in [[1, 1], [1, 0]]:
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

                    validate(max_pools, ndim, padding, float_min_val, orig_pad * ndim, layout, 2)

    # Check Pool pad folding when pad width on pad op is all zero.
    validate(max_pools, 1, [[0, 0], [0, 0], [0, 0]], float_min_val, [2, 0], "NCW", 2)
    # Check MaxPool pad folding with uint dtype
    int_min_val = get_min_value("uint8")
    validate(
        max_pools,
        2,
        [[0, 0], [0, 0], [0, 2], [2, 0]],
        int_min_val,
        [2, 0, 0, 0],
        "NCHW",
        2,
        dtype="uint8",
    )
    # Fold when original AvgPool has its own padding but count_include_pad=True
    validate(
        avg_pools,
        2,
        [[0, 0], [0, 0], [0, 2], [2, 0]],
        0,
        [0, 0, 1, 0],
        "NCHW",
        2,
        count_include_pad=True,
    )
    # Fold when count_include_pad=False but original AvgPool has no orig padding
    validate(avg_pools, 2, [[0, 0], [0, 0], [0, 2], [2, 0]], 0, [0, 0, 0, 0], "NCHW", 2)

    # Test no fold cases
    # AvgPool only folds pad when count_include_pad (False by default) is True
    validate(
        avg_pools, 2, [[0, 0], [0, 0], [0, 2], [2, 0]], 0, [0, 0, 0, 0], "NCHW", 2, no_fold=True
    )
    # MaxPool only folds pad when pad_value is the min for its dtype
    validate(max_pools, 1, [[0, 0], [0, 0], [0, 2]], 0, [0, 0], "NCHW", 2, no_fold=True)
    # AvgPool only folds pad when pad_value=0
    validate(avg_pools, 1, [[0, 0], [0, 0], [0, 2]], 1, [0, 0], "NCHW", 2, no_fold=True)
    # Pools only fold when pad_mode="constant"
    validate(
        avg_pools, 1, [[0, 0], [0, 0], [0, 2]], 0, [0, 0], "NCHW", 2, pad_mode="edge", no_fold=True
    )


def test_fold_pad_qconv2d():
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

    tvm.ir.assert_structural_equal(a, b, map_free_vars=True)


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

    tvm.ir.assert_structural_equal(a, b, map_free_vars=True)


if __name__ == "__main__":
    test_simplify_conv_pad()
    test_simplify_pool_pad()
    test_fold_pad_qconv2d()
    test_pad_qconv2d_no_fold()
