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


def test_simplify_reshape():
    def before():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    def expected():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(32, 16, 16))
        return relay.Function([x, w], y)

    def symbolic():
        b = tvm.te.size_var("b")
        x = relay.var("x", shape=(b, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    z = before()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)

    z = symbolic()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(symbolic(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_simplify_full_elementwise():
    def validate(shape, value, dtype):
        def before_left(x, elem_op, full):
            return elem_op(full, x)

        def after_left(x, elem_op, value):
            return elem_op(relay.const(value, dtype), x)

        def before_right(x, elem_op, full):
            return elem_op(x, full)

        def after_right(x, elem_op, value):
            return elem_op(x, relay.const(value, dtype))

        x = relay.var("x", shape=shape, dtype=dtype)
        elem_ops = [relay.add, relay.multiply, relay.subtract, relay.divide]
        full_ops = []
        if value == 0:
            full_ops.append(relay.zeros(shape, dtype))
            full_ops.append(relay.zeros_like(x))
        if value == 1:
            full_ops.append(relay.ones(shape, dtype))
            full_ops.append(relay.ones_like(x))
        else:
            full_ops.append(relay.full(relay.const(value, dtype), shape))
            full_ops.append(relay.full_like(x, relay.const(value, dtype)))
        for op in elem_ops:
            for full in full_ops:
                z = before_left(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(after_left(x, op, value), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

                z = before_right(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(after_right(x, op, value), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

        # Test the case in which x is broadcast to full's shape
        full_ops = []
        if value == 0:
            full_ops.append(relay.zeros(shape * 2, dtype))
        if value == 1:
            full_ops.append(relay.ones(shape * 2, dtype))
        else:
            full_ops.append(relay.full(relay.const(value, dtype), shape * 2))
        for op in elem_ops:
            for full in full_ops:
                z = before_left(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(before_left(x, op, full), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

                z = before_right(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(before_right(x, op, full), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

    for shape in [[10], [10, 10], [10, 10, 10]]:
        for dtype in ["float32", "int32", "bool"]:
            for value in [0, 1, 2]:
                validate(shape, value, dtype)


def test_simplify_conv_pad():
    convs = [relay.nn.conv1d, relay.nn.conv2d, relay.nn.conv3d]

    def validate(ndim, pad_width, pad_value, pad_mode, orig_padding, layout):
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

        zz = run_opt_pass(conv, transform.SimplifyExpr())
        expected = run_opt_pass(after, transform.InferType())
        assert tvm.ir.structural_equal(zz, expected)

        mod1 = tvm.IRModule.from_expr(conv)
        mod2 = tvm.IRModule.from_expr(zz)

        with tvm.transform.PassContext(disabled_pass="SimplifyExpr"):
            ex1 = relay.create_executor("vm", mod=mod1, ctx=tvm.cpu(), target="llvm")
        ex2 = relay.create_executor("vm", mod=mod2, ctx=tvm.cpu(), target="llvm")
        x_np = np.random.rand(*shape).astype("float32")
        w_np = np.random.rand(*wshape).astype("float32")
        result1 = ex1.evaluate()(x_np, w_np)
        result2 = ex2.evaluate()(x_np, w_np)

        tvm.testing.assert_allclose(result1.asnumpy(), result2.asnumpy())

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
    ndim = 2
    validate(ndim, [[0, 0]] * 2 + [i_pad] * ndim, 1, "constant", orig_pad * ndim, "NCHW")
    validate(ndim, [[0, 0]] * 2 + [i_pad] * ndim, 0, "edge", orig_pad * ndim, "NCHW")


if __name__ == "__main__":
    test_simplify_reshape()
    test_simplify_full_elementwise()
