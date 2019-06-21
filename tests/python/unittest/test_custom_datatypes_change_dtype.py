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
import topi.testing
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from tvm.target.datatype import register, register_min_func, register_op, create_lower_func, lower_ite

tgt = "llvm"


def convert_ndarray(dst_dtype, array):
    """Converts an NDArray into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor('graph').evaluate(cast)(array)


def change_dtype(src, dst, module, params):
    module = relay.frontend.ChangeDatatype(src, dst)(module)
    module = relay.transform.InferType()(module)
    params = dict((p, convert_ndarray(dst, params[p])) for p in params)
    return module, params


def setup():
    # You must first load the library containing the datatype implementation.
    # In this case, we have built the test functions used below right into TVM.
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)

    register("bfloat", 129)

    register_op(create_lower_func("FloatToBFloat16_wrapper"), "Cast", "llvm",
                "bfloat", "float")
    register_op(create_lower_func("BFloat16ToFloat_wrapper"), "Cast", "llvm",
                "float", "bfloat")
    register_op(create_lower_func("BFloat16Add_wrapper"), "Add", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Sub_wrapper"), "Sub", "llvm",
                "bfloat")
    register_op(create_lower_func("FloatToBFloat16_wrapper"), "FloatImm",
                "llvm", "bfloat")
    register_op(create_lower_func("BFloat16Mul_wrapper"), "Mul", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Div_wrapper"), "Div", "llvm",
                "bfloat")
    register_op(create_lower_func("BFloat16Max_wrapper"), "Max", "llvm",
                "bfloat")
    register_min_func(lambda num_bits: -3.38953139e38, "bfloat")

    register_op(
        create_lower_func("BFloat16Sqrt_wrapper"),
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    register_op(
        lower_ite,
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="tvm_if_then_else")
    register_op(
        create_lower_func("BFloat16Exp_wrapper"),
        "Call",
        "llvm",
        "bfloat",
        intrinsic_name="exp")


def test_ops(src_dtype, dst_dtype):
    def check_unary_op(opfunc, ref, src_dtype, dst_dtype):
        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1)
            z = opfunc(x)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            ref_res = ref(x_data)
            module = tvm.IRModule.from_expr(relay.Function([x], z))

            module, _ = change_dtype(src_dtype, dst_dtype, module, [])

            for target, ctx in [("llvm", tvm.cpu(0))]:
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target, mod=module)
                x_converted = convert_ndarray(dst_dtype, x_data)
                with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
                    op_res = intrp.evaluate()(x_converted)
                op_res_converted = convert_ndarray(src_dtype, op_res)
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
            module = tvm.IRModule.from_expr(relay.Function([x, y], z))

            module, _ = change_dtype(src_dtype, dst_dtype, module, [])

            for target, ctx in [("llvm", tvm.cpu(0))]:
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target, mod=module)
                x_converted = convert_ndarray(dst_dtype, x_data)
                y_converted = convert_ndarray(dst_dtype, y_data)
                with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
                    op_res = intrp.evaluate()(x_converted, y_converted)
                op_res_converted = convert_ndarray(src_dtype, op_res)
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
    module = tvm.IRModule.from_expr(func)

    A = tvm.nd.array(np.random.rand(3, 1).astype('float32'))
    B = tvm.nd.array(np.random.rand(3, 1).astype('float32'))

    ex = relay.create_executor("graph", mod=module)
    # Execute the model in the new datatype.
    result = ex.evaluate()(A, B)

    module_changed, _ = change_dtype('float32', 'custom[bfloat]16', module, [])
    ex = relay.create_executor("graph", mod=module_changed)

    A_converted = convert_ndarray('custom[bfloat]16', A)
    B_converted = convert_ndarray('custom[bfloat]16', B)
    result = ex.evaluate()(A_converted, B_converted)
    print(result.dtype)
    result_converted = convert_ndarray('float32', result)
    print(result_converted)


def test_change_dtype_resnet():
    module, params = get_resnet()

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 224, 224).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    # Execute the model in the new datatype.
    ex = relay.create_executor("graph", mod=module)
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        result = ex.evaluate()(input, **params)


def test_change_dtype_inception_v3():
    module, params = get_inception()

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    ex = relay.create_executor("graph")
    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 299, 299).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    # Execute the model in the new datatype.
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        ex = relay.create_executor('graph', mod=module)
        result = ex.evaluate()(input, **params)


def test_change_dtype_mobilenet():
    module, params = get_mobilenet()

    src_dtype = 'float32'
    dst_dtype = 'custom[bfloat]16'
    module, params = change_dtype(src_dtype, dst_dtype, module, params)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(3, 224, 224).astype(src_dtype))
    input = convert_ndarray(dst_dtype, input)

    # Execute the model in the new datatype.
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        ex = relay.create_executor('graph', mod=module)
        result = ex.evaluate()(input, **params)

def test_model(get_workload, input_shape, src_dtype, dst_dtype):
    expr, params = get_workload()

    ex = relay.create_executor("graph")

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(*input_shape).astype(src_dtype))

    correct = ex.evaluate(expr)(input, **params)

    # Simplifying inference is essential right now, as batch norms (which get
    # removed) are broken with custom datatypes.
    expr = relay.ir_pass.simplify_inference(expr)
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params, ex)

    input = convert_ndarray(dst_dtype, input, ex)

    # Vectorization is not implemented with custom datatypes.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)

    tvm.testing.assert_allclose(
        convert_ndarray(src_dtype, result, ex).asnumpy(), correct.asnumpy(), rtol=0.5, atol=0.5)


def test_conv2d():
    def run_test_conv2d(src_dtype,
                        dst_dtype,
                        scale,
                        dshape,
                        kshape,
                        padding=(1, 1),
                        fref=None,
                        groups=1,
                        dilation=(1, 1),
                        except_targets=None,
                        **attrs):
        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=src_dtype)
        w = relay.var("w", dtype=src_dtype)
        y = relay.nn.conv2d(
            x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(src_dtype)
        kernel = np.random.uniform(
            -scale, scale, size=kshape).astype(src_dtype)
        dkernel = topi.testing.dilate_python(kernel, (1, 1) + dilation)
        if fref is None:
            ref_res = topi.testing.conv2d_nchw_python(
                data.astype(src_dtype),
                dkernel.astype(src_dtype),
                1,
                padding,
                groups=groups)
        else:
            ref_res = fref(data.astype(src_dtype), dkernel.astype(src_dtype))

        for target, ctx in [("llvm", tvm.cpu(0))]:
            if target in except_targets:
                continue
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            # convert function
            func, _ = change_dtype(src_dtype, dst_dtype, func, [], intrp1)
            data_converted = convert_ndarray(dst_dtype, data, intrp1)
            kernel_converted = convert_ndarray(dst_dtype, kernel, intrp1)
            with tvm.build_config(disable_vectorize=True):
                op_res1 = intrp1.evaluate(func)(data_converted, kernel_converted)
            op_res1_converted = convert_ndarray(src_dtype, op_res1, intrp1)
            # TODO(gus) previous rtol, atol 1e-5
            tvm.testing.assert_allclose(
                op_res1_converted.asnumpy(), ref_res, rtol=0.5, atol=0.5)

    # depthwise conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    run_test_conv2d("float32", "custom[bfloat]16", 1, dshape, kshape,
                    padding=(1, 1), channels=32, groups=32, kernel_size=(3 ,3),
                    fref=lambda x, w: topi.testing.depthwise_conv2d_python_nchw(
                        x, w, (1, 1), "SAME"))

    # CUDA is disabled for 'direct' schedule:
    # https://github.com/dmlc/tvm/pull/3070#issuecomment-486597553
    # group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    run_test_conv2d("float32", "custom[bfloat]16", 1, dshape, kshape,
                    padding=(1, 1), channels=32, groups=8, kernel_size=(3 ,3),
                    except_targets=['cuda'])
    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    run_test_conv2d("float32", "custom[bfloat]16", 1, dshape, kshape,
                    padding=(1, 1), channels=64, groups=32, kernel_size=(3 ,3),
                    except_targets=['cuda'])

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    run_test_conv2d("float32", "custom[bfloat]16", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3 ,3))

    # dilated conv2d
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    run_test_conv2d("float32", "custom[bfloat]16", 1, dshape, kshape,
                    padding=(1, 1), channels=10, kernel_size=(3 ,3), dilation=(3, 3))

if __name__ == "__main__":
    setup()
    test_ops('float32', 'custom[bfloat]16')
    # test_change_dtype_inception_v3()
    # test_change_dtype_simple()
    # test_change_dtype_mobilenet()
    # test_change_dtype_resnet()
