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
import pytest
import tvm
from tvm import te

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


def test_legalize():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def legalize_conv2d(attrs, inputs, types):
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

    with TempOpAttr("nn.conv2d", "FTVMLegalize", legalize_conv2d):
        a = before()
        a = run_opt_pass(a, transform.Legalize())
        b = run_opt_pass(expected(), transform.InferType())

    tvm.ir.assert_structural_equal(a, b)


def test_legalize_none():
    """Test doing nothing by returning 'None'"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    def legalize_conv2d(attrs, inputs, types):
        called[0] = True
        return None

    with TempOpAttr("nn.global_max_pool2d", "FTVMLegalize", legalize_conv2d):
        a = before()
        a = run_opt_pass(a, transform.Legalize())
        b = run_opt_pass(before(), transform.InferType())

    tvm.ir.assert_structural_equal(a, b)
    assert called[0]


def test_legalize_multiple_ops():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var("weight", shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    def legalize_conv2d(attrs, inputs, types):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0, "float32"))
        return relay.nn.conv2d(data, weight, **attrs)

    def legalize_relu(attrs, inputs, types):
        data = inputs[0]
        add = relay.add(tvm.relay.const(0, "float32"), data)
        return relay.nn.relu(add)

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
        y = relay.add(tvm.relay.const(0, "float32"), y)
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    with TempOpAttr("nn.conv2d", "FTVMLegalize", legalize_conv2d):
        with TempOpAttr("nn.relu", "FTVMLegalize", legalize_relu):
            a = before()
            a = run_opt_pass(a, transform.Legalize())
            b = run_opt_pass(expected(), transform.InferType())

    tvm.ir.assert_structural_equal(a, b)


def test_legalize_multi_input():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

    def legalize_concatenate(attrs, inputs, types):
        # Check that the correct multi-input case is handled.
        assert len(inputs) == 1
        assert isinstance(inputs[0], tvm.relay.expr.Tuple)
        assert len(types) == 2
        assert isinstance(types[0], tvm.relay.ty.TupleType)
        assert isinstance(types[1], tvm.relay.ty.TensorType)
        return None

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

    with TempOpAttr("concatenate", "FTVMLegalize", legalize_concatenate):
        a = before()
        a = run_opt_pass(a, transform.Legalize())
        b = run_opt_pass(expected(), transform.InferType())

    tvm.ir.assert_structural_equal(a, b)


@pytest.mark.parametrize(
    "target,exp_in_channels",
    [
        (
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu",
            8,
        ),
        (
            "llvm --device=arm_cpu --mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod",
            3,
        ),
        (
            "llvm --device=arm_cpu --mtriple=aarch64-linux-gnu -mattr=+v8.2a,+i8mm",
            8,
        ),
        (
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
            8,
        ),
        (
            "llvm -device=arm_cpu -mtriple=armv8l-linux-gnu -mattr=+neon",
            8,
        ),
    ],
)
def test_conv2d_NHWC_legalize(target, exp_in_channels):
    target = tvm.target.Target(target)

    dtype = "int8"
    data_layout = "NHWC"
    kernel_layout = "HWIO"
    in_channels = 3
    out_channels = 4
    kernel_size = (1, 1)

    x = relay.var("x", shape=(1, 1, 1, in_channels), dtype=dtype)
    weight = relay.var("weight", shape=(1, 1, in_channels, out_channels), dtype=dtype)
    out = relay.nn.conv2d(
        x,
        weight,
        kernel_size=kernel_size,
        channels=out_channels,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_dtype=dtype,
    )

    with target:
        out = run_opt_pass(out, transform.Legalize())

    act_in_channels = out.args[0].type_args[0].shape[3]

    assert act_in_channels == exp_in_channels, "Actual input channels = " + str(act_in_channels)


if __name__ == "__main__":
    tvm.testing.main()
