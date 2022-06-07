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

"""CMSIS-NN integration tests: extract_constants pass"""
import itertools
import math
import numpy as np
import pytest
import tvm
import tvm.testing
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
def test_external_function():
    y0_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x0 = relay.var("x0", shape=(8, 8))
    y0_const = relay.const(y0_data, "float32")
    z0 = x0 + y0_const
    ef = relay.Function([x0], z0, relay.TensorType((8, 8), "float32"))
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)

    x = relay.var("x", shape=(8, 8))
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    CheckFunctionsForConstants().check_num_constants(mod[ev])
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_nested_function():
    y1_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x1 = relay.var("x1", shape=(8, 8))
    y1_const = relay.const(y1_data, "float32")
    z1 = x1 + y1_const
    w1 = z1 * relay.const(5.0, "float32")
    lf = relay.Function([x1], w1, relay.TensorType((8, 8), "float32"))
    lf = set_composite_func_attr(lf, "cmsis-nn")

    x0 = relay.var("x0", shape=(8, 8))
    c0 = relay.Call(lf, [x0])
    ef = relay.Function([x0], c0, relay.TensorType((8, 8), "float32"))

    x = relay.var("x", shape=(8, 8))
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x])
    mf = relay.Function([x], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    CheckFunctionsForConstants().check_num_constants(mod[ev])
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_multiple_functions():
    y20_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x20 = relay.var("x20", shape=(8, 8))
    y20_const = relay.const(y20_data, "float32")
    z20 = x20 + y20_const
    f20 = relay.Function([x20], z20, relay.TensorType((8, 8), "float32"))
    f20 = set_composite_func_attr(f20, "cmsis-nn")

    y21_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x21 = relay.var("x21", shape=(8, 8))
    y21_const = relay.const(y21_data, "float32")
    z21 = x21 + y21_const
    f21 = relay.Function([x21], z21, relay.TensorType((8, 8), "float32"))
    f21 = set_composite_func_attr(f21, "cmsis-nn")

    x10 = relay.var("x10", shape=(8, 8))
    c10 = relay.Call(f20, [x10])
    c11 = relay.Call(f21, [c10])
    ef = relay.Function([x10], c11, relay.TensorType((8, 8), "float32"))
    x0 = relay.var("x0", shape=(8, 8))
    ev = relay.GlobalVar("cmsis-nn")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)
    c = relay.Call(ev, [x0])
    mf = relay.Function([x0], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    CheckFunctionsForConstants().check_num_constants(mod[ev])
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_main_function():
    x0 = relay.var("x0", shape=(8, 8))
    y0 = relay.var("y0", shape=(8, 8))
    z0 = x0 + y0
    ef = relay.Function([x0, y0], z0, relay.TensorType((8, 8), "float32"))
    ev = relay.GlobalVar("external_function")
    ef = set_external_func_attr(ef, "cmsis-nn", ev.name_hint)

    x = relay.var("x", shape=(8, 8))
    y_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    y_const = relay.const(y_data, "float32")
    z = x + y_const
    c = relay.Call(ev, [x, z])
    mf = relay.Function([x], c, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev] = ef
    mod[mv] = mf

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[mv].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "main() should have same number of arguments as before"


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("external_compiler", ["cmsis-nn", "other_compiler"])
def test_multiple_functions_non_cmsisnn_compiler(external_compiler):
    y20_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x20 = relay.var("x20", shape=(8, 8))
    y20_const = relay.const(y20_data, "float32")
    z20 = x20 + y20_const
    f20 = relay.Function([x20], z20, relay.TensorType((8, 8), "float32"))
    f20 = set_composite_func_attr(f20, "cmsis-nn.qnn_op_1")
    x10 = relay.var("x10", shape=(8, 8))
    c10 = relay.Call(f20, [x10])
    ef0 = relay.Function([x10], c10, relay.TensorType((8, 8), "float32"))

    y21_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x21 = relay.var("x21", shape=(8, 8))
    y21_const = relay.const(y21_data, "float32")
    z21 = x21 + y21_const
    f21 = relay.Function([x21], z21, relay.TensorType((8, 8), "float32"))
    f21 = set_composite_func_attr(f21, "cmsis-nn.qnn_op_2")
    x11 = relay.var("x11", shape=(8, 8))
    c11 = relay.Call(f21, [x11])
    ef1 = relay.Function([x11], c11, relay.TensorType((8, 8), "float32"))

    x0 = relay.var("x0", shape=(8, 8))
    ev0 = relay.GlobalVar("external_function_0")
    ef0 = set_external_func_attr(ef0, external_compiler, ev0.name_hint)
    c0 = relay.Call(ev0, [x0])
    ev1 = relay.GlobalVar("external_function_1")
    ef1 = set_external_func_attr(ef1, external_compiler, ev1.name_hint)
    c1 = relay.Call(ev1, [c0])
    mf = relay.Function([x0], c1, relay.TensorType((8, 8), "float32"))
    mv = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[ev0] = ef0
    mod[ev1] = ef1
    mod[mv] = mf

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[mv].body)

    num_extracted_constants = 0
    if external_compiler == "cmsis-nn":
        num_extracted_constants = 2

    assert (
        check_for_constants.num_constants_ == num_extracted_constants
    ), "main() should have same number of arguments as before"


if __name__ == "__main__":
    tvm.testing.main()
