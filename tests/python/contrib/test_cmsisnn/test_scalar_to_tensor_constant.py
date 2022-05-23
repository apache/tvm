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

"""CMSIS-NN integration tests: scalar_to_tensor_constant pass"""
import sys

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


def generate_variable(name, shape, dtype="int8"):
    return relay.var(name, shape=shape, dtype=dtype)


def make_binary_op(
    op,
    input_0,
    input_1,
    input_0_scale,
    input_0_zero_point,
    input_1_scale,
    input_1_zero_point,
    out_scale=1.0 / 256,
    out_zero_point=-128,
):
    """Create a Relay Function / network model"""
    return op(
        input_0,
        input_1,
        relay.const(input_0_scale, "float32"),
        relay.const(input_0_zero_point, "int32"),
        relay.const(input_1_scale, "float32"),
        relay.const(input_1_zero_point, "int32"),
        relay.const(out_scale, "float32"),
        relay.const(out_zero_point, "int32"),
    )


class CheckFunctionsForConstants(tvm.relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.num_constants_ = 0

    def visit_call(self, call):
        super().visit_call(call)
        for arg in call.args:
            if isinstance(arg, relay.Constant) and arg.data.numpy().ndim > 0:
                self.num_constants_ += 1

    def check_num_constants(self, func):
        assert self.num_constants_ == 0, "Functions should not have constant arguments in Calls"


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func


def set_composite_func_attr(func, name):
    func = func.with_attr("Composite", name)
    return func


@tvm.testing.requires_cmsisnn
def test_single_scalar_position_0():
    dtype = "int8"
    shape = (8, 8)
    x0 = generate_variable("x0", None, dtype)
    x1 = generate_variable("x1", shape, dtype)
    z1 = make_binary_op(
        relay.qnn.op.add,
        x0,
        x1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    lf = relay.Function([x0, x1], z1, relay.TensorType(shape, dtype))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.expr.const(3, dtype)
    y1 = relay.var("y1", shape=shape, dtype=dtype)
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([y1], c0, relay.TensorType(shape, dtype))

    x = relay.var("x", shape=shape, dtype=dtype)
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType(shape, dtype))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    mod = relay.transform.InferType()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_single_scalar_position_1():
    dtype = "int8"
    shape = (8, 8)
    x0 = generate_variable("x0", shape, dtype)
    x1 = generate_variable("x1", None, dtype)
    z1 = make_binary_op(
        relay.qnn.op.add,
        x0,
        x1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    lf = relay.Function([x0, x1], z1, relay.TensorType(shape, dtype))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.var("y0", shape=shape, dtype=dtype)
    y1 = relay.expr.const(3, dtype)
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([y0], c0, relay.TensorType(shape, dtype))

    x = relay.var("x", shape=shape, dtype=dtype)
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType(shape, dtype))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    mod = relay.transform.InferType()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_primary_operands_all_scalars():
    dtype = "int8"
    shape = None
    x0 = generate_variable("x0", None, dtype)
    x1 = generate_variable("x1", None, dtype)
    z1 = make_binary_op(
        relay.qnn.op.add,
        x0,
        x1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    lf = relay.Function([x0, x1], z1, relay.TensorType(shape, dtype))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.expr.const(7, dtype)
    y1 = relay.expr.const(3, dtype)
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([], c0, relay.TensorType(shape, dtype))

    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [])
    mf = relay.Function([], c, relay.TensorType(shape, dtype))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    new_mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(mod[ev].body, new_mod[ev].body)


@tvm.testing.requires_cmsisnn
def test_all_primary_operands_tensor_constants():
    dtype = "int8"
    shape = (1, 3, 3, 32)
    x0 = generate_variable("x0", shape, dtype)
    x1 = generate_variable("x1", shape, dtype)
    z1 = make_binary_op(
        relay.qnn.op.add,
        x0,
        x1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    lf = relay.Function([x0, x1], z1, relay.TensorType(shape, dtype))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    rng = np.random.default_rng(12345)
    y0 = relay.const(rng.integers(-128, high=127, size=shape, dtype=dtype))
    y1 = relay.const(rng.integers(-128, high=127, size=shape, dtype=dtype))
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([], c0, relay.TensorType(shape, dtype))

    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [])
    mf = relay.Function([], c, relay.TensorType(shape, dtype))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    new_mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(mod[ev].body, new_mod[ev].body)


@tvm.testing.requires_cmsisnn
def test_non_cmsisnn_ext_func():
    """Non CMSISNN functions should not be altered."""

    def get_mod():
        x1 = relay.var("x1", shape=None)
        x2 = relay.var("x2", shape=None)
        z1 = x1 + x2
        lf = relay.Function([x1, x2], z1, relay.TensorType((), "float32"))
        lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

        y0 = relay.expr.const(5, "float32")
        y1 = relay.expr.const(3, "float32")
        c0 = relay.Call(lf, [y0, y1])
        ef = relay.Function([], c0, relay.TensorType((), "float32"))

        ev = relay.GlobalVar("external_function")
        ef = set_external_func_attr(ef, "foo", ev.name_hint)
        c = relay.Call(ev, [])
        mf = relay.Function([], c, relay.TensorType((), "float32"))
        mv = relay.GlobalVar("main")

        mod = tvm.IRModule()
        mod[ev] = ef
        mod[mv] = mf
        mod = relay.transform.InferType()(mod)
        return mod

    expected = get_mod()["external_function"].body
    actual = ScalarToTensorConstants()(get_mod())["external_function"].body
    assert tvm.ir.structural_equal(expected, actual)


if __name__ == "__main__":
    tvm.testing.main()
