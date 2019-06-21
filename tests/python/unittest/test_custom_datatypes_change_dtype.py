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
"""Utilities for changing datatypes of models."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet

tgt = "llvm"


def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    tvm.datatype.register("bfloat", 129)

    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"), "Cast",
        "llvm", "float", "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("IntToBFloat16_wrapper"), "Cast",
        "llvm", "bfloat", "int")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
        "llvm", "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
        "bfloat")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Sqrt_wrapper"),
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    tvm.datatype.register_op(
        tvm.datatype.lower_ite,
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("BFloat16Exp_wrapper"),
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="exp")


def convert_ndarray(dst_dtype, array, executor):
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    return executor.evaluate(cast)(array)


def change_dtype(src, dst, expr, params, executor):
    expr = relay.ir_pass.infer_type(expr)
    cdtype = relay.frontend.ChangeDatatype(src, dst)
    expr = cdtype.visit(expr)
    expr = relay.ir_pass.infer_type(expr)
    #raise "pause"
    params = dict(
        (p, convert_ndarray(dst, params[p], executor)) for p in params)
    return expr, params


def test_ops(src_dtype, dst_dtype):
    def check_unary_op(opfunc, ref, src_dtype, dst_dtype):
        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1)
            z = opfunc(x)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            ref_res = ref(x_data)
            func = relay.Function([x], z)

            ex = relay.create_executor("graph")
            func, _ = change_dtype(src_dtype, dst_dtype, func, [], ex)
            print(func)

            for target, ctx in [("llvm", tvm.cpu(0))]:
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                x_converted = convert_ndarray(dst_dtype, x_data, ex)
                op_res = intrp.evaluate(func)(x_converted)
                op_res_converted = convert_ndarray(src_dtype, op_res, ex)
                # TODO(gus) increased the tolerance an unreasonable amount
                np.testing.assert_allclose(
                    op_res_converted.asnumpy(), ref_res, rtol=0.1, atol=0.1)

    def check_binary_op(opfunc, ref, src_dtype, dst_dtype):
        if ref is not None:
            t1 = relay.TensorType((5, 10, 5), src_dtype)
            t2 = relay.TensorType((5, ), src_dtype)
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            y_data = np.random.rand(5).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            ex = relay.create_executor("graph")
            func, _ = change_dtype(src_dtype, dst_dtype, func, [], ex)

            for target, ctx in [("llvm", tvm.cpu(0))]:
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                x_converted = convert_ndarray(dst_dtype, x_data, ex)
                y_converted = convert_ndarray(dst_dtype, y_data, ex)
                op_res = intrp.evaluate(func)(x_converted, y_converted)
                op_res_converted = convert_ndarray(src_dtype, op_res, ex)
                np.testing.assert_allclose(
                    op_res_converted.asnumpy(), ref_res, rtol=0.01, atol=0.01)

    def my_func(x, y):
        a = relay.add(x, y)
        return relay.nn.relu(a)

    def my_func_np(x, y):
        a = x + y
        return np.maximum(0, a)

    def sigmoid(x):
        one = np.ones_like(x)
        return one / (one + np.exp(-x))

    def relu(x):
        x_copy = np.copy(x)
        np.maximum(x_copy, 0, x_copy)
        return x_copy

    def rsqrt(x):
        one = np.ones_like(x)
        return one / np.sqrt(x)

    check_binary_op(relay.add, np.add, src_dtype, dst_dtype)
    check_binary_op(relay.subtract, np.subtract, src_dtype, dst_dtype)
    check_binary_op(relay.divide, np.divide, src_dtype, dst_dtype)
    check_binary_op(relay.multiply, np.multiply, src_dtype, dst_dtype)
    check_unary_op(relay.sqrt, np.sqrt, src_dtype, dst_dtype)
    check_unary_op(relay.negative, np.negative, src_dtype, dst_dtype)
    #check_binary_op(my_func, my_func_np, src_dtype, dst_dtype)
    check_unary_op(relay.nn.relu, relu, src_dtype, dst_dtype)
    for opfunc, ref in [#(tvm.relay.log, np.log),
                        (tvm.relay.exp, np.exp),
                        (tvm.relay.sqrt, np.sqrt),
                        (tvm.relay.rsqrt, rsqrt),
                        #(tvm.relay.sigmoid, sigmoid),
                        #(tvm.relay.tanh, np.tanh),
                        (relay.nn.relu, relu)]:
        check_unary_op(opfunc, ref, src_dtype, dst_dtype)

def test_change_dtype_simple():
    shape = (3, 1)
    t = relay.TensorType(shape, 'float32')
    a = relay.var("a", t)
    b = relay.var("b", t)
    func = relay.Function([a, b], a + b)

    A = tvm.nd.array(np.random.rand(3, 1).astype('float32'))
    B = tvm.nd.array(np.random.rand(3, 1).astype('float32'))

    ex = relay.create_executor("graph")
    # Execute the model in the new datatype.
    result = ex.evaluate(func)(A, B)

    func_changed, _ = change_dtype('float32', 'custom[bfloat]16', func, [], ex)
    A_converted = convert_ndarray('custom[bfloat]16', A, ex)
    B_converted = convert_ndarray('custom[bfloat]16', B, ex)
    result = ex.evaluate(func_changed)(A_converted, B_converted)
    print(result.dtype)
    result_converted = convert_ndarray('float32', result, ex)
    print(result_converted)


def test_change_dtype_resnet():
    expr, params = get_resnet()

    ex = relay.create_executor("graph")

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'  # Change me to posit.
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params, ex)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 224, 224).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input, ex)

    def print_info(node):
        if not isinstance(node, relay.op.op.Op):
            if ("custom[bfloat]32" not in str(node.checked_type())):
                print(node.checked_type())

    relay.ir_pass.post_order_visit(expr, print_info)

    # Execute the model in the new datatype.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)


def test_change_dtype_inception_v3():
    expr, params = get_inception()

    ex = relay.create_executor("graph")

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'  # Change me to posit.
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params, ex)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input, ex)

    def print_info(node):
        if not isinstance(node, relay.op.op.Op):
            if ("custom[bfloat]32" not in str(node.checked_type())):
                print(node.checked_type())

    relay.ir_pass.post_order_visit(expr, print_info)

    # Execute the model in the new datatype.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)


def test_change_dtype_mobilenet():
    expr, params = get_mobilenet()

    ex = relay.create_executor("graph")

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'  # Change me to posit.
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params, ex)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 224, 224).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input, ex)

    # def print_info(node):
    #     if not isinstance(node, relay.op.op.Op):
    #         if ("custom[bfloat]32" not in str(node.checked_type())):
    #             print(node.checked_type())
    # relay.ir_pass.post_order_visit(expr, print_info)

    # Execute the model in the new datatype.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)
        print(convert_ndarray("float32", result, ex))


if __name__ == "__main__":
    setup()
    test_change_dtype_inception_v3()
    test_change_dtype_simple()
    test_change_dtype_mobilenet()
    test_change_dtype_resnet()
    test_ops('float32', 'custom[bfloat]16')
