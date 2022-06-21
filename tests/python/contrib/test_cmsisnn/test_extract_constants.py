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
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


class CheckFunctionsForConstants(tvm.relay.ExprVisitor):
    """Provides methods to test number of constants present in a function"""

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
def test_external_function():
    """Tests the pass ExternConstants when the function is a global function"""
    input1_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    input0 = relay.var("input0", shape=(8, 8))
    input1_const = relay.const(input1_data, "float32")
    binary_op = input0 + input1_const
    extern_func = relay.Function([input0], binary_op, relay.TensorType((8, 8), "float32"))
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)

    arg = relay.var("arg", shape=(8, 8))
    call_extern_func = relay.Call(global_var, [arg])
    main_func = relay.Function([arg], call_extern_func, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    constant_verifier = CheckFunctionsForConstants()
    constant_verifier.visit_function(mod[global_var])
    constant_verifier.check_num_constants()
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_nested_function():
    """Tests the pass ExternConstants when a composite function
    is present within global function
    """
    input1_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    input0 = relay.var("input0", shape=(8, 8))
    input1_const = relay.const(input1_data, "float32")
    binary_op0 = input0 + input1_const
    binary_op1 = binary_op0 * relay.const(5.0, "float32")
    local_func = relay.Function([input0], binary_op1, relay.TensorType((8, 8), "float32"))
    local_func = set_composite_func_attr(local_func, "cmsis-nn")

    arg = relay.var("arg", shape=(8, 8))
    call_local_func = relay.Call(local_func, [arg])
    extern_func = relay.Function([arg], call_local_func, relay.TensorType((8, 8), "float32"))

    global_arg = relay.var("garg", shape=(8, 8))
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [global_arg])
    main_func = relay.Function([global_arg], call_extern_func, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    constant_verifier = CheckFunctionsForConstants()
    constant_verifier.visit_function(mod[global_var])
    constant_verifier.check_num_constants()
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_internal_function_with_duplicate_arguments():
    """Tests the pass ExternConstants when a composite function
    is present within global function with repeating arguments
    to one of the binary ops.
    """
    input0 = relay.var("input0", shape=(8, 8))
    binary_op0 = input0 + input0
    binary_op1 = binary_op0 * relay.const(5.0, "float32")
    local_func = relay.Function([input0], binary_op1, relay.TensorType((8, 8), "float32"))
    local_func = set_composite_func_attr(local_func, "cmsis-nn")

    arg = relay.var("arg", shape=(8, 8))
    call_local_func = relay.Call(local_func, [arg])
    extern_func = relay.Function([arg], call_local_func, relay.TensorType((8, 8), "float32"))

    global_arg = relay.var("global_var", shape=(8, 8))
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [global_arg])
    main_func = relay.Function([global_arg], call_extern_func, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    constant_verifier = CheckFunctionsForConstants()
    constant_verifier.visit_function(mod[global_var])
    constant_verifier.check_num_constants()
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_multiple_functions():
    """Tests the pass ExternConstants when global function
    contains multiple composite functions inside it
    """
    f0_input1_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    f0_input0 = relay.var("f0_in0", shape=(8, 8))
    f0_input1_const = relay.const(f0_input1_data, "float32")
    f0_binary_op = f0_input0 + f0_input1_const
    f0_func = relay.Function([f0_input0], f0_binary_op, relay.TensorType((8, 8), "float32"))
    f0_func = set_composite_func_attr(f0_func, "cmsis-nn")

    f1_input1_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    f1_input0 = relay.var("f1_in0", shape=(8, 8))
    f1_input1_const = relay.const(f1_input1_data, "float32")
    f1_binary_op = f1_input0 + f1_input1_const
    f1_func = relay.Function([f1_input0], f1_binary_op, relay.TensorType((8, 8), "float32"))
    f1_func = set_composite_func_attr(f1_func, "cmsis-nn")

    arg0 = relay.var("arg0", shape=(8, 8))
    call_local_func0 = relay.Call(f0_func, [arg0])
    call_local_func1 = relay.Call(f1_func, [call_local_func0])
    extern_func = relay.Function([arg0], call_local_func1, relay.TensorType((8, 8), "float32"))
    input0 = relay.var("input0", shape=(8, 8))
    global_var = relay.GlobalVar("cmsis-nn")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)
    call_extern_func = relay.Call(global_var, [input0])
    main_func = relay.Function([input0], call_extern_func, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    constant_verifier = CheckFunctionsForConstants()
    constant_verifier.visit_function(mod[global_var])
    constant_verifier.check_num_constants()
    relay.transform.InferType()(mod)


@tvm.testing.requires_cmsisnn
def test_main_function():
    """Tests the pass ExternConstants on main function"""
    input0 = relay.var("input0", shape=(8, 8))
    input1 = relay.var("input1", shape=(8, 8))
    binary_op = input0 + input1
    extern_func = relay.Function([input0, input1], binary_op, relay.TensorType((8, 8), "float32"))
    global_var = relay.GlobalVar("external_function")
    extern_func = set_external_func_attr(extern_func, "cmsis-nn", global_var.name_hint)

    arg = relay.var("arg", shape=(8, 8))
    input_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    input_const = relay.const(input_data, "float32")
    binary_op = arg + input_const
    call_extern_func = relay.Call(global_var, [arg, binary_op])
    main_func = relay.Function([arg], call_extern_func, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var] = extern_func
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[main_var].body)
    assert (
        check_for_constants.num_constants_ == 1
    ), "main() should have same number of arguments as before"


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("external_compiler", ["cmsis-nn", "other_compiler"])
def test_multiple_functions_non_cmsisnn_compiler(external_compiler):
    """Tests the pass ExternConstants on non CMSIS-NN targets"""
    y20_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x20 = relay.var("x20", shape=(8, 8))
    y20_const = relay.const(y20_data, "float32")
    z20 = x20 + y20_const
    f20 = relay.Function([x20], z20, relay.TensorType((8, 8), "float32"))
    f20 = set_composite_func_attr(f20, "cmsis-nn.qnn_op_1")
    x10 = relay.var("x10", shape=(8, 8))
    call_local_func0 = relay.Call(f20, [x10])
    extern_func0 = relay.Function([x10], call_local_func0, relay.TensorType((8, 8), "float32"))

    y21_data = np.random.uniform(0, 1, (8, 8)).astype("float32")
    x21 = relay.var("x21", shape=(8, 8))
    y21_const = relay.const(y21_data, "float32")
    z21 = x21 + y21_const
    f21 = relay.Function([x21], z21, relay.TensorType((8, 8), "float32"))
    f21 = set_composite_func_attr(f21, "cmsis-nn.qnn_op_2")
    x11 = relay.var("x11", shape=(8, 8))
    call_local_func1 = relay.Call(f21, [x11])
    extern_func1 = relay.Function([x11], call_local_func1, relay.TensorType((8, 8), "float32"))

    input0 = relay.var("input0", shape=(8, 8))
    global_var0 = relay.GlobalVar("external_function_0")
    extern_func0 = set_external_func_attr(extern_func0, external_compiler, global_var0.name_hint)
    call_extern_func0 = relay.Call(global_var0, [input0])
    global_var1 = relay.GlobalVar("external_function_1")
    extern_func1 = set_external_func_attr(extern_func1, external_compiler, global_var1.name_hint)
    call_extern_func1 = relay.Call(global_var1, [call_extern_func0])
    main_func = relay.Function([input0], call_extern_func1, relay.TensorType((8, 8), "float32"))
    main_var = relay.GlobalVar("main")

    mod = tvm.IRModule()
    mod[global_var0] = extern_func0
    mod[global_var1] = extern_func1
    mod[main_var] = main_func

    mod = ExtractConstantsFromPartitionedFunction()(mod)
    check_for_constants = CheckFunctionsForConstants()
    check_for_constants.visit_call(mod[main_var].body)

    num_extracted_constants = 0
    if external_compiler == "cmsis-nn":
        num_extracted_constants = 2

    assert (
        check_for_constants.num_constants_ == num_extracted_constants
    ), "main() should have same number of arguments as before"


if __name__ == "__main__":
    tvm.testing.main()
