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
from tvm.contrib import graph_runtime
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
def test_legalize_conv2d():
    """test legalize conv2d to enable tensorcore"""

    def _test_legalize_conv2d(data_shape, kernel_shape, pad_shape, do_pad=True):
        out_channel = kernel_shape[3]
        out_shape = list(data_shape)
        out_shape[3] = out_channel
        db, di, do = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype="float16")
            weight = relay.var("weight", shape=kernel_shape, dtype="float16")
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
            x = relay.var("x", shape=data_shape, dtype="float16")
            if db or di:
                x_pad = relay.nn.pad(x, pad_width=((0, db), (0, 0), (0, 0), (0, di)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype="float16")
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

    # conv2d pad batch
    _test_legalize_conv2d((7, 16, 16, 64), (3, 3, 64, 64), (1, 0, 0))
    _test_legalize_conv2d((3, 16, 16, 64), (3, 3, 64, 64), (5, 0, 0))
    _test_legalize_conv2d((2, 16, 16, 64), (3, 3, 64, 64), (0, 0, 0), False)
    # conv2d pad in_channel
    _test_legalize_conv2d((8, 16, 16, 63), (3, 3, 63, 64), (0, 1, 0))
    _test_legalize_conv2d((8, 16, 16, 33), (3, 3, 33, 64), (0, 15, 0))
    _test_legalize_conv2d((8, 16, 16, 13), (3, 3, 13, 64), (0, 3, 0))
    _test_legalize_conv2d((8, 16, 16, 1), (3, 3, 1, 64), (0, 0, 0), False)
    # conv2d pad out_channel
    _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 63), (0, 0, 1))
    _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 33), (0, 0, 31))
    _test_legalize_conv2d((8, 16, 16, 64), (3, 3, 64, 1), (0, 0, 0), False)


@tvm.testing.uses_gpu
def test_legalize_dense():
    def _test_legalize_dense(data_shape, kernel_shape, pad_shape, do_pad=True):
        """test legalize dense to enable tensorcore"""
        M, K = data_shape
        N, _ = kernel_shape
        out_shape = (M, N)
        dm, dk, dn = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype="float16")
            weight = relay.var("weight", shape=kernel_shape, dtype="float16")
            y = relay.nn.dense(x, weight)
            y = relay.Function([x, weight], y)
            return y

        def legalize_dense(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.dense_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()
            x = relay.var("x", shape=data_shape, dtype="float16")
            if dm or dk:
                x_pad = relay.nn.pad(x, pad_width=((0, dm), (0, dk)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype="float16")
            if dn or dk:
                weight_pad = relay.nn.pad(weight, pad_width=((0, dn), (0, dk)))
            else:
                weight_pad = weight
            y_pad = relay.nn.dense(
                x_pad,
                weight_pad,
            )
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
    _test_legalize_dense((8, 16), (32, 16), (0, 0, 0), False)
    _test_legalize_dense((7, 16), (32, 16), (1, 0, 0))
    _test_legalize_dense((8, 15), (32, 15), (0, 1, 0))
    _test_legalize_dense((8, 16), (31, 16), (0, 0, 1))
    _test_legalize_dense((7, 15), (31, 15), (1, 1, 1))
    _test_legalize_dense((3, 16), (32, 16), (5, 0, 0))
    _test_legalize_dense((2, 16), (32, 16), (0, 0, 0), False)


@tvm.testing.uses_gpu
def test_legalize_batch_matmul():
    def _test_legalize_batch_matmul(data_shape, kernel_shape, pad_shape, do_pad=True):
        """test legalize dense to enable tensorcore"""
        B, M, _ = data_shape
        _, N, _ = kernel_shape
        out_shape = (B, M, N)
        dm, dk, dn = pad_shape

        def before():
            x = relay.var("x", shape=data_shape, dtype="float16")
            weight = relay.var("weight", shape=kernel_shape, dtype="float16")
            y = relay.nn.batch_matmul(x, weight)
            y = relay.Function([x, weight], y)
            return y

        def legalize_batch_matmul(attrs, inputs, types):
            with tvm.target.Target("cuda"):
                return topi.nn.batch_matmul_legalize(attrs, inputs, types)

        def expected():
            if not do_pad:
                return before()
            x = relay.var("x", shape=data_shape, dtype="float16")
            if dm or dk:
                x_pad = relay.nn.pad(x, pad_width=((0, 0), (0, dm), (0, dk)))
            else:
                x_pad = x
            weight = relay.var("weight", shape=(kernel_shape), dtype="float16")
            if dn or dk:
                weight_pad = relay.nn.pad(weight, pad_width=((0, 0), (0, dn), (0, dk)))
            else:
                weight_pad = weight
            y_pad = relay.nn.batch_matmul(
                x_pad,
                weight_pad,
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

    _test_legalize_batch_matmul((16, 8, 16), (16, 32, 16), (0, 0, 0), False)
    _test_legalize_batch_matmul((16, 7, 16), (16, 32, 16), (1, 0, 0))
    _test_legalize_batch_matmul((16, 8, 15), (16, 32, 15), (0, 1, 0))
    _test_legalize_batch_matmul((16, 8, 16), (16, 31, 16), (0, 0, 1))
    _test_legalize_batch_matmul((16, 7, 15), (16, 31, 15), (1, 1, 1))
    _test_legalize_batch_matmul((16, 3, 16), (16, 32, 16), (5, 0, 0))
    _test_legalize_batch_matmul((16, 2, 16), (16, 32, 16), (0, 0, 0), False)


if __name__ == "__main__":
    test_legalize_conv2d()
    test_legalize_dense()
    test_legalize_batch_matmul()
