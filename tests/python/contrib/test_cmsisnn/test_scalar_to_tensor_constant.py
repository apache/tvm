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
import numpy as np
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
    """Provides method to test number of scalar constants present in a function"""

    def __init__(self):
        super().__init__()
        self.num_constants_ = 0

    def visit_call(self, call):
        super().visit_call(call)
        for arg in call.args:
            if isinstance(arg, relay.Constant) and arg.data.numpy().ndim > 0:
                self.num_constants_ += 1

    def check_num_constants(self):
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
    """Tests conversion to tensor constant when first operand is a scalar"""
    dtype = "int8"
    shape = (8, 8)
    operand0 = generate_variable("operand0", None, dtype)
    operand1 = generate_variable("operand1", shape, dtype)
    binary_op = make_binary_op(
        relay.qnn.op.add,
        operand0,
        operand1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    local_func = relay.Function([operand0, operand1], binary_op, relay.TensorType(shape, dtype))
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

    arg0 = relay.expr.const(3, dtype)
    arg1 = relay.var("arg1", shape=shape, dtype=dtype)
    call_local_func = relay.Call(local_func, [arg0, arg1])
    extern_func = relay.Function([arg1], call_local_func, relay.TensorType(shape, dtype))

    x = relay.var("x", shape=shape, dtype=dtype)
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [x])
    main_func = relay.Function([x], call_extern_func, relay.TensorType(shape, dtype))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    mod = relay.transform.InferType()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[global_var].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_single_scalar_position_1():
    """Tests conversion to tensor constant when second operand is a scalar"""
    dtype = "int8"
    shape = (8, 8)
    operand0 = generate_variable("operand0", shape, dtype)
    operand1 = generate_variable("operand1", None, dtype)
    binary_op = make_binary_op(
        relay.qnn.op.add,
        operand0,
        operand1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    local_func = relay.Function([operand0, operand1], binary_op, relay.TensorType(shape, dtype))
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

    arg0 = relay.var("arg0", shape=shape, dtype=dtype)
    arg1 = relay.expr.const(3, dtype)
    call_local_func = relay.Call(local_func, [arg0, arg1])
    extern_func = relay.Function([arg0], call_local_func, relay.TensorType(shape, dtype))

    x = relay.var("x", shape=shape, dtype=dtype)
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [x])
    main_func = relay.Function([x], call_extern_func, relay.TensorType(shape, dtype))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    mod = relay.transform.InferType()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[global_var].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "Scalar constant wasn't converted into tensor constant"


@tvm.testing.requires_cmsisnn
def test_primary_operands_all_scalars():
    """Tests conversion to tensor constants all operands are scalars"""
    dtype = "int8"
    shape = None
    operand0 = generate_variable("operand0", None, dtype)
    operand1 = generate_variable("operand1", None, dtype)
    binary_op = make_binary_op(
        relay.qnn.op.add,
        operand0,
        operand1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    local_func = relay.Function([operand0, operand1], binary_op, relay.TensorType(shape, dtype))
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

    arg0 = relay.expr.const(7, dtype)
    arg1 = relay.expr.const(3, dtype)
    call_local_func = relay.Call(local_func, [arg0, arg1])
    extern_func = relay.Function([], call_local_func, relay.TensorType(shape, dtype))

    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [])
    main_func = relay.Function([], call_extern_func, relay.TensorType(shape, dtype))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    new_mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(mod[global_var].body, new_mod[global_var].body)


@tvm.testing.requires_cmsisnn
def test_all_primary_operands_tensor_constants():
    """Tests conversion to tensor constants all operands are tensors"""
    dtype = "int8"
    shape = (1, 3, 3, 32)
    operand0 = generate_variable("operand0", shape, dtype)
    operand1 = generate_variable("operand1", shape, dtype)
    binary_op = make_binary_op(
        relay.qnn.op.add,
        operand0,
        operand1,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    local_func = relay.Function([operand0, operand1], binary_op, relay.TensorType(shape, dtype))
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

    rng = np.random.default_rng(12345)
    arg0 = relay.const(rng.integers(-128, high=127, size=shape, dtype=dtype))
    arg1 = relay.const(rng.integers(-128, high=127, size=shape, dtype=dtype))
    call_local_func = relay.Call(local_func, [arg0, arg1])
    extern_func = relay.Function([], call_local_func, relay.TensorType(shape, dtype))

    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [])
    main_func = relay.Function([], call_extern_func, relay.TensorType(shape, dtype))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    new_mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(mod[global_var].body, new_mod[global_var].body)


@tvm.testing.requires_cmsisnn
def test_duplicate_constant_arguments():
    """Tests the pass when repeating operands are arguments to the binary op"""
    dtype = "int8"
    shape = (1, 3, 3, 32)
    operand0 = generate_variable("operand0", shape, dtype)
    operand1 = generate_variable("operand1", shape, dtype)
    binary_op = make_binary_op(
        relay.qnn.op.add,
        operand0,
        operand0,
        input_0_scale=0.0128,
        input_0_zero_point=32,
        input_1_scale=0.256,
        input_1_zero_point=-64,
    )

    local_func = relay.Function([operand0, operand1], binary_op, relay.TensorType(shape, dtype))
    local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

    rng = np.random.default_rng(12345)
    arg0 = relay.const(rng.integers(-128, high=127, size=shape, dtype=dtype))
    call_local_func = relay.Call(local_func, [arg0, arg0])
    extern_func = relay.Function([], call_local_func, relay.TensorType(shape, dtype))

    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [])
    main_func = relay.Function([], call_extern_func, relay.TensorType(shape, dtype))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = relay.transform.InferType()(mod)
    mod = ScalarToTensorConstants()(mod)
    new_mod = relay.transform.InferType()(mod)
    assert tvm.ir.structural_equal(mod[global_var].body, new_mod[global_var].body)


@tvm.testing.requires_cmsisnn
def test_non_cmsisnn_ext_func():
    """Non CMSISNN functions should not be altered."""

    def get_mod():
        operand1 = relay.var("operand1", shape=None)
        operand2 = relay.var("operand2", shape=None)
        binary_op = operand1 + operand2
        local_func = relay.Function(
            [operand1, operand2], binary_op, relay.TensorType((), "float32")
        )
        local_func = set_composite_func_attr(local_func, "cmsis-nn.qnn_add")

        arg0 = relay.expr.const(5, "float32")
        arg1 = relay.expr.const(3, "float32")
        call_local_func = relay.Call(local_func, [arg0, arg1])
        extern_func = relay.Function([], call_local_func, relay.TensorType((), "float32"))

        global_var = relay.GlobalVar("external_function")
        extern_func = set_external_func_attr(extern_func, "foo", global_var.name_hint)
        call_extern_func = relay.Call(global_var, [])
        main_func = relay.Function([], call_extern_func, relay.TensorType((), "float32"))
        main_var = relay.GlobalVar("main")

        mod = tvm.IRModule()
        mod[global_var] = extern_func
        mod[main_var] = main_func
        mod = relay.transform.InferType()(mod)
        return mod

    expected = get_mod()["external_function"].body
    actual = ScalarToTensorConstants()(get_mod())["external_function"].body
    assert tvm.ir.structural_equal(expected, actual)


if __name__ == "__main__":
    tvm.testing.main()
