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
import itertools
import math
import numpy as np
import pytest
import tvm
from tvm import relay

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


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
    x0 = relay.var("x0", shape=None)
    x1 = relay.var("x1", shape=(8, 8))
    z1 = x0 + x1
    lf = relay.Function([x0, x1], z1, relay.TensorType((8, 8), "float32"))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.expr.const(3, "float32")
    y1 = relay.var("y1", shape=(8, 8))
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([y1], c0, relay.TensorType((8, 8), "float32"))

    x = relay.var("x", shape=(8, 8))
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_single_scalar_position_1():
    x0 = relay.var("x0", shape=(8, 8))
    x1 = relay.var("x1", shape=None)
    z1 = x0 + x1
    lf = relay.Function([x0, x1], z1, relay.TensorType((8, 8), "float32"))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.var("y0", shape=(8, 8))
    y1 = relay.expr.const(3, "float32")
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([y0], c0, relay.TensorType((8, 8), "float32"))

    x = relay.var("x", shape=(8, 8))
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_two_scalars():
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
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [])
    mf = relay.Function([], c, relay.TensorType((), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 0
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_two_tensor_constants():
    x0 = relay.var("x0", shape=(8, 8))
    x1 = relay.var("x1", shape=(8, 8))
    z1 = x0 + x1
    lf = relay.Function([x0, x1], z1, relay.TensorType((8, 8), "float32"))
    lf = set_composite_func_attr(lf, "cmsis-nn.qnn_add")

    y0 = relay.const(np.random.uniform(0, 1, (8, 8)).astype("float32"), "float32")
    y1 = relay.const(np.random.uniform(0, 1, (8, 8)).astype("float32"), "float32")
    c0 = relay.Call(lf, [y0, y1])
    ef = relay.Function([], c0, relay.TensorType((8, 8), "float32"))

    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [])
    mf = relay.Function([], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[ev].body)
    assert (
        check_for_constants.num_constants_ == 2
    ), "Scalar constant wasn't converted into tensor constant"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
