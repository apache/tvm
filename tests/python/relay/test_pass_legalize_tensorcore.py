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
"""Test legalize pass"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


@tvm.testing.uses_gpu
def test_legalize_conv2d_NHWC():
    """test legalize NHWC conv2d to enable tensorcore"""

    def _test_legalize_conv2d(data_shape, kernel_shape, pad_shape, dtype, do_pad=True):
        out_channel = kernel_shape[3]
        out_shape = list(data_shape)
        out_shape[3] = out_channel
        db, di, do = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype=dtype)
            weight = relay.var("weight", shape=kernel_shape, dtype=dtype)
            y = relay.nn.conv2d(
                x,
                weight,
                channels=out_channel,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            y = relay.Function([x, weight], y)
            return y

        def legalize_conv2d(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.conv2d_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()
            x = relay.var("x", shape=data_shape, dtype=dtype)
            if db or di:
                x_pad = relay.nn.pad(x, pad_width=((0, db), (0, 0), (0, 0), (0, di)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype=dtype)
            if di or do:
                weight_pad = relay.nn.pad(weight, pad_width=((0, 0), (0, 0), (0, di), (0, do)))
            else:
                weight_pad = weight
            y_pad = relay.nn.conv2d(
                x_pad,
                weight=weight_pad,
                channels=out_channel + do,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
            if db or do:
                y = relay.strided_slice(y_pad, begin=[0, 0, 0, 0], end=out_shape)
            else:
                y = y_pad
            y = relay.Function([x, weight], y)
            return y

        with TempOpAttr("nn.conv2d", "FTVMLegalize", legalize_conv2d):
            a = before()
            a = run_opt_pass(a, transform.Legalize())
            b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "Expected = \n" + str(b)

    for dtype in ["float16", "int8", "int4"]:
        # conv2d pad batch
        _test_legalize_conv2d((7, 16, 16, 64), (3, 3, 64, 64), (1, 0, 0), dtype)
        _test_legalize_conv2d((3, 16, 16, 64), (3, 3, 64, 64), (5, 0, 0), dtype)
        _test_legalize_conv2d((2, 16, 16, 64), (3, 3, 64, 64), (0, 0, 0), dtype, False)
        # conv2d pad in_channel
        _test_legalize_conv2d((8, 16, 16, 63), (3, 3, 63, 64), (0, 1, 0), dtype)
        _test_legalize_conv2d((8, 16, 16, 33), (3, 3, 33, 64), (0, 15, 0), dtype)
        _test_legalize_conv2d((8, 16, 16, 13), (3, 3, 13, 64), (0, 3, 0), dtype)
        _test_legalize_conv2d((8, 16, 16, 1), (3, 3, 1, 64), (0, 0, 0), dtype, False)
        # conv2d pad out_channel
        _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 63), (0, 0, 1), dtype)
        _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 33), (0, 0, 31), dtype)
        _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 1), (0, 0, 0), dtype, False)


@tvm.testing.uses_gpu
def test_legalize_conv2d_HWNC():
    """test legalize HWNC conv2d to enable tensorcore"""

    def _test_legalize_conv2d(data_shape, kernel_shape, pad_shape, dtype, do_pad=True):
        out_channel = kernel_shape[2]
        out_shape = list(data_shape)
        out_shape[3] = out_channel
        db, di, do = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype=dtype)
            weight = relay.var("weight", shape=kernel_shape, dtype=dtype)
            y = relay.nn.conv2d(
                x,
                weight,
                channels=out_channel,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="HWNC",
                kernel_layout="HWOI",
            )
            y = relay.Function([x, weight], y)
            return y

        def legalize_conv2d(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.conv2d_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()
            x = relay.var("x", shape=data_shape, dtype=dtype)
            if db or di:
                x_pad = relay.nn.pad(x, pad_width=((0, 0), (0, 0), (0, db), (0, di)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype=dtype)
            if di or do:
                weight_pad = relay.nn.pad(weight, pad_width=((0, 0), (0, 0), (0, do), (0, di)))
            else:
                weight_pad = weight
            y_pad = relay.nn.conv2d(
                x_pad,
                weight=weight_pad,
                channels=out_channel + do,
                kernel_size=(3, 3),
                padding=(1, 1),
                data_layout="HWNC",
                kernel_layout="HWOI",
            )
            if db or do:
                y = relay.strided_slice(y_pad, begin=[0, 0, 0, 0], end=out_shape)
            else:
                y = y_pad
            y = relay.Function([x, weight], y)
            return y

        with TempOpAttr("nn.conv2d", "FTVMLegalize", legalize_conv2d):
            a = before()
            a = run_opt_pass(a, transform.Legalize())
            b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "Expected = \n" + str(b)

    # conv2d pad batch
    _test_legalize_conv2d((16, 16, 7, 64), (3, 3, 64, 64), (1, 0, 0), "int8")
    _test_legalize_conv2d((16, 16, 3, 64), (3, 3, 64, 64), (5, 0, 0), "int8")
    _test_legalize_conv2d((2, 16, 16, 64), (3, 3, 64, 64), (0, 0, 0), "int8", False)
    _test_legalize_conv2d((16, 16, 7, 64), (3, 3, 64, 64), (1, 0, 0), "int4")
    _test_legalize_conv2d((16, 16, 3, 64), (3, 3, 64, 64), (5, 0, 0), "int4")
    _test_legalize_conv2d((2, 16, 16, 64), (3, 3, 64, 64), (0, 0, 0), "int4", False)
    # conv2d pad in_channel
    _test_legalize_conv2d((16, 16, 8, 63), (3, 3, 64, 63), (0, 1, 0), "int8")
    _test_legalize_conv2d((16, 16, 8, 33), (3, 3, 64, 33), (0, 15, 0), "int8")
    _test_legalize_conv2d((16, 16, 8, 13), (3, 3, 64, 13), (0, 3, 0), "int8")
    _test_legalize_conv2d((16, 16, 8, 1), (3, 3, 64, 1), (0, 0, 0), "int8", False)
    _test_legalize_conv2d((16, 16, 8, 63), (3, 3, 64, 63), (0, 1, 0), "int4")
    _test_legalize_conv2d((16, 16, 8, 33), (3, 3, 64, 33), (0, 31, 0), "int4")
    _test_legalize_conv2d((16, 16, 8, 13), (3, 3, 64, 13), (0, 19, 0), "int4")
    _test_legalize_conv2d((16, 16, 8, 1), (3, 3, 64, 1), (0, 0, 0), "int4", False)
    # conv2d pad out_channel
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 63, 64), (0, 0, 1), "int8")
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 33, 64), (0, 0, 31), "int8")
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 1, 64), (0, 0, 0), "int8", False)
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 63, 64), (0, 0, 1), "int4")
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 33, 64), (0, 0, 7), "int4")
    _test_legalize_conv2d((16, 16, 8, 64), (3, 3, 1, 64), (0, 0, 0), "int4", False)


@tvm.testing.uses_gpu
def test_legalize_dense():
    def _test_legalize_dense(data_shape, kernel_shape, pad_shape, dtype, do_pad=True, units=None):
        """test legalize dense to enable tensorcore"""
        M, K = data_shape
        N, _ = kernel_shape
        out_shape = (M, N)
        dm, dk, dn = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype=dtype)
            weight = relay.var("weight", shape=kernel_shape, dtype=dtype)
            y = relay.nn.dense(x, weight, units)
            y = relay.Function([x, weight], y)
            return y

        def legalize_dense(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.dense_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()
            x = relay.var("x", shape=data_shape, dtype=dtype)
            if dm or dk:
                x_pad = relay.nn.pad(x, pad_width=((0, dm), (0, dk)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype=dtype)
            if dn or dk:
                weight_pad = relay.nn.pad(weight, pad_width=((0, dn), (0, dk)))
            else:
                weight_pad = weight
            y_pad = relay.nn.dense(x_pad, weight_pad, units=N + dn if units else None)
            if dm or dn:
                y = relay.strided_slice(y_pad, begin=[0, 0], end=out_shape)
            else:
                y = y_pad
            y = relay.Function([x, weight], y)
            return y

        with TempOpAttr("nn.dense", "FTVMLegalize", legalize_dense):
            a = before()
            a = run_opt_pass(a, transform.Legalize())
            b = run_opt_pass(expected(), transform.InferType())

        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "Expected = \n" + str(b)

    # dense
    for dtype in ["float16", "int8"]:
        _test_legalize_dense((8, 16), (32, 16), (0, 0, 0), dtype, False)
        _test_legalize_dense((7, 16), (32, 16), (1, 0, 0), dtype)
        _test_legalize_dense((8, 15), (32, 15), (0, 1, 0), dtype)
        _test_legalize_dense((8, 16), (31, 16), (0, 0, 1), dtype)
        _test_legalize_dense((7, 15), (31, 15), (1, 1, 1), dtype)
        _test_legalize_dense((3, 16), (32, 16), (5, 0, 0), dtype)
        _test_legalize_dense((1, 16), (32, 16), (0, 0, 0), dtype, False)

    # Test if units parameter is correctly updated
    _test_legalize_dense((8, 16), (30, 16), (0, 0, 2), "float16", units=30)

    _test_legalize_dense((8, 32), (32, 32), (0, 0, 0), "int4", False)
    _test_legalize_dense((7, 32), (32, 32), (1, 0, 0), "int4")
    _test_legalize_dense((8, 31), (32, 31), (0, 1, 0), "int4")
    _test_legalize_dense((8, 32), (31, 32), (0, 0, 1), "int4")
    _test_legalize_dense((7, 31), (31, 31), (1, 1, 1), "int4")
    _test_legalize_dense((3, 32), (32, 32), (5, 0, 0), "int4")
    _test_legalize_dense((8, 16), (32, 16), (0, 16, 0), "int4")
    _test_legalize_dense((2, 16), (32, 16), (0, 0, 0), "int4", False)


@tvm.testing.uses_gpu
def test_legalize_batch_matmul():
    def _test_legalize_batch_matmul(
        data_shape, kernel_shape, pad_shape, dtype, do_pad=True, transpose_a=False, transpose_b=True
    ):
        """test legalize dense to enable tensorcore"""
        if transpose_a:
            B, _, M = data_shape
        else:
            B, M, _ = data_shape

        if transpose_b:
            _, N, _ = kernel_shape
        else:
            _, _, N = kernel_shape

        out_shape = (B, M, N)
        dm, dk, dn = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype=dtype)
            weight = relay.var("weight", shape=kernel_shape, dtype=dtype)
            y = relay.nn.batch_matmul(x, weight, transpose_a=transpose_a, transpose_b=transpose_b)
            y = relay.Function([x, weight], y)
            return y

        def legalize_batch_matmul(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.batch_matmul_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()

            x = relay.var("x", shape=data_shape, dtype=dtype)
            weight = relay.var("weight", shape=(kernel_shape), dtype=dtype)

            if dm or dk:
                if transpose_a:
                    x_pad = relay.nn.pad(x, pad_width=((0, 0), (0, dk), (0, dm)))
                else:
                    x_pad = relay.nn.pad(x, pad_width=((0, 0), (0, dm), (0, dk)))
            else:
                x_pad = x

            if dn or dk:
                if transpose_b:
                    weight_pad = relay.nn.pad(weight, pad_width=((0, 0), (0, dn), (0, dk)))
                else:
                    weight_pad = relay.nn.pad(weight, pad_width=((0, 0), (0, dk), (0, dn)))
            else:
                weight_pad = weight

            y_pad = relay.nn.batch_matmul(
                x_pad,
                weight_pad,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
            if dm or dn:
                y = relay.strided_slice(y_pad, begin=[0, 0, 0], end=out_shape)
            else:
                y = y_pad
            y = relay.Function([x, weight], y)
            return y

        with TempOpAttr("nn.batch_matmul", "FTVMLegalize", legalize_batch_matmul):
            a = before()
            a = run_opt_pass(a, transform.Legalize())
            b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a) + "Expected = \n" + str(b)

    for dtype in ["float16", "int8"]:
        _test_legalize_batch_matmul((16, 8, 16), (16, 32, 16), (0, 0, 0), dtype, False)
        _test_legalize_batch_matmul((16, 7, 16), (16, 32, 16), (1, 0, 0), dtype)
        _test_legalize_batch_matmul((16, 8, 15), (16, 32, 15), (0, 1, 0), dtype)
        _test_legalize_batch_matmul((16, 8, 16), (16, 31, 16), (0, 0, 1), dtype)
        _test_legalize_batch_matmul((16, 7, 15), (16, 31, 15), (1, 1, 1), dtype)
        _test_legalize_batch_matmul((16, 3, 16), (16, 32, 16), (5, 0, 0), dtype)
        _test_legalize_batch_matmul((16, 2, 16), (16, 32, 16), (0, 0, 0), dtype, False)

    _test_legalize_batch_matmul((16, 8, 32), (16, 32, 32), (0, 0, 0), "int4", False)
    _test_legalize_batch_matmul((16, 7, 32), (16, 32, 32), (1, 0, 0), "int4")
    _test_legalize_batch_matmul((16, 8, 31), (16, 32, 31), (0, 1, 0), "int4")
    _test_legalize_batch_matmul((16, 8, 32), (16, 31, 32), (0, 0, 1), "int4")
    _test_legalize_batch_matmul((16, 7, 31), (16, 31, 31), (1, 1, 1), "int4")
    _test_legalize_batch_matmul((16, 3, 32), (16, 32, 32), (5, 0, 0), "int4")
    _test_legalize_batch_matmul((16, 8, 16), (16, 32, 16), (0, 16, 0), "int4")
    _test_legalize_batch_matmul((16, 2, 16), (16, 32, 16), (0, 0, 0), "int4", False)

    _test_legalize_batch_matmul(
        (16, 8, 16), (16, 16, 32), (0, 0, 0), "float16", False, transpose_b=False
    )
    _test_legalize_batch_matmul(
        (16, 16, 8), (16, 32, 16), (0, 0, 0), "float16", False, transpose_a=True
    )


if __name__ == "__main__":
    test_legalize_conv2d_NHWC()
    test_legalize_conv2d_HWNC()
    test_legalize_dense()
    test_legalize_batch_matmul()
