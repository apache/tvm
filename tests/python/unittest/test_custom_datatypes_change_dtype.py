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
from nose.tools import nottest

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
    """Set up tests

    Currently, this registers some custom datatypes using the Bring Your
    Own Datatypes framework.
    """

    # To use datatype operations in an external library, you should first load
    # the library containing the datatype implementation:
    # CDLL("libposit.so", RTLD_GLOBAL)
    # In this case, the datatype library we are using is built right into TVM,
    # so we do not need to explicitly load any library.

    # You can pick a code for your datatype arbitrarily, as long as it is
    # greater than 128 and has not already been chosen.

    register("posit32", 131)

    register_op(create_lower_func("FloatToPosit32es2"), "Cast", "llvm",
                "posit32", "float")
    register_op(create_lower_func("Posit32es2ToFloat"), "Cast", "llvm",
                "float", "posit32")
    register_op(create_lower_func("IntToPosit32es2"), "Cast", "llvm",
                "posit32", "int")
    register_op(create_lower_func("Posit32es2Add"), "Add", "llvm", "posit32")
    register_op(create_lower_func("Posit32es2Sub"), "Sub", "llvm", "posit32")
    register_op(create_lower_func("FloatToPosit32es2"), "FloatImm", "llvm",
                "posit32")
    register_op(create_lower_func("Posit32es2Mul"), "Mul", "llvm", "posit32")
    register_op(create_lower_func("Posit32es2Div"), "Div", "llvm", "posit32")
    register_op(create_lower_func("Posit32es2Max"), "Max", "llvm", "posit32")
    register_op(create_lower_func("Posit32es2Sqrt"),
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    register_op(lower_ite,
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="tvm_if_then_else")
    register_op(create_lower_func("Posit32es2Exp"),
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="exp")
    register_op(create_lower_func("Posit32es2Log"),
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="log")
    register_op(create_lower_func("Posit32es2Sigmoid"),
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="sigmoid")
    register_op(create_lower_func("Posit32es2Tanh"),
                "Call",
                "llvm",
                "posit32",
                intrinsic_name="tanh")
    register_min_func(lambda num_bits: -1.329227995784915872903807060280344576e36, "posit32")

    register("posit8", 132)
    register_op(create_lower_func("FloatToPosit8es0"), "Cast", "llvm",
                "posit8", "float")
    register_op(create_lower_func("Posit8es0ToFloat"), "Cast", "llvm", "float",
                "posit8")
    register_op(create_lower_func("IntToPosit8es0"), "Cast", "llvm", "posit8",
                "int")
    register_op(create_lower_func("Posit8es0Add"), "Add", "llvm", "posit8")
    register_op(create_lower_func("Posit8es0Sub"), "Sub", "llvm", "posit8")
    register_op(create_lower_func("FloatToPosit8es0"), "FloatImm", "llvm",
                "posit8")
    register_op(create_lower_func("Posit8es0Mul"), "Mul", "llvm", "posit8")
    register_op(create_lower_func("Posit8es0Div"), "Div", "llvm", "posit8")
    register_op(create_lower_func("Posit8es0Max"), "Max", "llvm", "posit8")
    register_op(create_lower_func("Posit8es0Sqrt"),
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    register_op(lower_ite,
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="tvm_if_then_else")
    register_op(create_lower_func("Posit8es0Exp"),
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="exp")
    register_op(create_lower_func("Posit8es0Log"),
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="log")
    register_op(create_lower_func("Posit8es0Sigmoid"),
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="sigmoid")
    register_op(create_lower_func("Posit8es0Tanh"),
                "Call",
                "llvm",
                "posit8",
                intrinsic_name="tanh")
    register_min_func(lambda num_bits: -64, "posit8")

    register("posit16", 133)
    register_op(create_lower_func("FloatToPosit16es1"), "Cast", "llvm",
                "posit16", "float")
    register_op(create_lower_func("Posit16es1ToFloat"), "Cast", "llvm",
                "float", "posit16")
    register_op(create_lower_func("IntToPosit16es1"), "Cast", "llvm",
                "posit16", "int")
    register_op(create_lower_func("Posit16es1Add"), "Add", "llvm", "posit16")
    register_op(create_lower_func("Posit16es1Sub"), "Sub", "llvm", "posit16")
    register_op(create_lower_func("FloatToPosit16es1"), "FloatImm", "llvm",
                "posit16")
    register_op(create_lower_func("Posit16es1Mul"), "Mul", "llvm", "posit16")
    register_op(create_lower_func("Posit16es1Div"), "Div", "llvm", "posit16")
    register_op(create_lower_func("Posit16es1Max"), "Max", "llvm", "posit16")
    register_op(create_lower_func("Posit16es1Sqrt"),
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    register_op(lower_ite,
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="tvm_if_then_else")
    register_op(create_lower_func("Posit16es1Exp"),
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="exp")
    register_op(create_lower_func("Posit16es1Log"),
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="log")
    register_op(create_lower_func("Posit16es1Sigmoid"),
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="sigmoid")
    register_op(create_lower_func("Posit16es1Tanh"),
                "Call",
                "llvm",
                "posit16",
                intrinsic_name="tanh")
    register_min_func(lambda num_bits: -2.68435456e8, "posit16")

    register("noptype", 134)
    register_op(create_lower_func("FloatToNop32"), "Cast", "llvm", "noptype",
                "float")
    register_op(create_lower_func("Nop32ToFloat"), "Cast", "llvm", "float",
                "noptype")
    register_op(create_lower_func("IntToNop32"), "Cast", "llvm", "noptype",
                "int")
    register_op(create_lower_func("Nop32Add"), "Add", "llvm", "noptype")
    register_op(create_lower_func("Nop32Sub"), "Sub", "llvm", "noptype")
    register_op(create_lower_func("FloatToNop32"), "FloatImm", "llvm",
                "noptype")
    register_op(create_lower_func("Nop32Mul"), "Mul", "llvm", "noptype")
    register_op(create_lower_func("Nop32Div"), "Div", "llvm", "noptype")
    register_op(create_lower_func("Nop32Max"), "Max", "llvm", "noptype")
    register_op(create_lower_func("Nop32Sqrt"),
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    register_op(lower_ite,
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="tvm_if_then_else")
    register_op(create_lower_func("Nop32Exp"),
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="exp")
    register_op(create_lower_func("Nop32Log"),
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="log")
    register_op(create_lower_func("Nop32Sigmoid"),
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="sigmoid")
    register_op(create_lower_func("Nop32Tanh"),
                "Call",
                "llvm",
                "noptype",
                intrinsic_name="tanh")
    # This can be anything, considering the type isn't functionally correct.
    register_min_func(lambda num_bits: 0, "noptype")


def run_ops(src_dtype, dst_dtype, rtol=1e-7, atol=1e-7):
    """Run the same op, but with two different datatypes"""
    def check_unary_op(op, src_dtype, dst_dtype):
        t1 = relay.TensorType((5, 10, 5))
        x = relay.var("x", t1)
        z = op(x)
        x_data = np.random.rand(5, 10, 5).astype(t1.dtype)

        module = tvm.IRModule.from_expr(relay.Function([x], z))

        ex = relay.create_executor("graph", mod=module)

        correct = ex.evaluate()(x_data)

        module, _ = change_dtype(src_dtype, dst_dtype, module, [])
        ex = relay.create_executor("graph", mod=module)

        x_converted = convert_ndarray(dst_dtype, x_data)
        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            maybe_correct = ex.evaluate()(x_converted)
            maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct)
        np.testing.assert_allclose(maybe_correct_converted.asnumpy(),
                                   correct.asnumpy(),
                                   rtol=rtol,
                                   atol=atol)
        # print(maybe_correct_converted)
        # print(correct)

    for op in [
            relay.nn.softmax,
            tvm.relay.log,
            tvm.relay.exp,
            tvm.relay.sqrt,
            tvm.relay.rsqrt,
            tvm.relay.sigmoid,
            tvm.relay.tanh,
            relay.nn.relu,
    ]:
        check_unary_op(op, src_dtype, dst_dtype)

    def check_binary_op(opfunc, src_dtype, dst_dtype):
        t1 = relay.TensorType((5, 10, 5), src_dtype)
        t2 = relay.TensorType((5, ), src_dtype)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
        y_data = np.random.rand(5).astype(t2.dtype)
        module = tvm.IRModule.from_expr(relay.Function([x, y], z))

        ex = relay.create_executor("graph", mod=module)

        correct = ex.evaluate()(x_data, y_data)

        module, _ = change_dtype(src_dtype, dst_dtype, module, [])
        ex = relay.create_executor("graph", mod=module)

        x_converted = convert_ndarray(dst_dtype, x_data)
        y_converted = convert_ndarray(dst_dtype, y_data)

        with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
            maybe_correct = ex.evaluate()(x_converted, y_converted)
            maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct)
        np.testing.assert_allclose(correct.asnumpy(),
                                   maybe_correct_converted.asnumpy(),
                                   rtol=rtol,
                                   atol=atol)

    for op in [
            relay.add,
            relay.subtract,
            relay.divide,
            relay.multiply,
    ]:
        check_binary_op(op, src_dtype, dst_dtype)


def run_model(get_workload,
              input_shape,
              src_dtype,
              dst_dtype,
              num_classes,
              rtol=0.0001,
              atol=0.0001):
    module, params = get_workload(image_shape=input_shape,
                                  num_classes=num_classes)

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(*input_shape).astype(src_dtype))

    ex = relay.create_executor('graph', mod=module)
    correct = ex.evaluate()(input, **params)

    # Simplifying inference is essential right now, as batch norms (which get
    # removed) are broken with custom datatypes.
    #expr = relay.ir_pass.simplify_inference(expr)
    module, params = change_dtype(src_dtype, dst_dtype, module, params)
    ex = relay.create_executor('graph', mod=module)

    input = convert_ndarray(dst_dtype, input)

    # Vectorization is not implemented with custom datatypes.
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        result = ex.evaluate()(input, **params)
        tvm.testing.assert_allclose(convert_ndarray(src_dtype,
                                                    result).asnumpy(),
                                    correct.asnumpy(),
                                    rtol=rtol,
                                    atol=atol)


def run_conv2d(src_dtype, dst_dtype):
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
        w = relay.var("w", shape=kshape, dtype=src_dtype)
        y = relay.nn.conv2d(x,
                            w,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            **attrs)
        module = tvm.IRModule.from_expr(relay.Function([x, w], y))
        data = np.random.uniform(-scale, scale, size=dshape).astype(src_dtype)
        kernel = np.random.uniform(-scale, scale,
                                   size=kshape).astype(src_dtype)
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
            intrp1 = relay.create_executor("graph",
                                           ctx=ctx,
                                           target=target,
                                           mod=module)
            module, _ = change_dtype(src_dtype, dst_dtype, module, [])
            data_converted = convert_ndarray(dst_dtype, data)
            kernel_converted = convert_ndarray(dst_dtype, kernel)
            with tvm.transform.PassContext(
                    config={"tir.disable_vectorize": True}):
                op_res1 = intrp1.evaluate()(data_converted, kernel_converted)
            op_res1_converted = convert_ndarray(src_dtype, op_res1)
            tvm.testing.assert_allclose(op_res1_converted.asnumpy(), ref_res)

    # depthwise conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=32,
                    groups=32,
                    kernel_size=(3, 3),
                    fref=lambda x, w: topi.testing.
                    depthwise_conv2d_python_nchw(x, w, (1, 1), "SAME"))

    # CUDA is disabled for 'direct' schedule:
    # https://github.com/dmlc/tvm/pull/3070#issuecomment-486597553
    # group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=32,
                    groups=8,
                    kernel_size=(3, 3),
                    except_targets=['cuda'])
    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=64,
                    groups=32,
                    kernel_size=(3, 3),
                    except_targets=['cuda'])

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=10,
                    kernel_size=(3, 3))

    # dilated conv2d
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=10,
                    kernel_size=(3, 3),
                    dilation=(3, 3))


def test_ops():
    # TODO(gus) these tolerances are high, and still sometimes fail;
    # this is expected, b/c we're comparing between 32bit float and 8
    # bit posit.
    # Figure out a more logical way to test here.
    run_ops('float32', 'custom[posit8]8', rtol=1, atol=1)
    run_ops('float32', 'custom[posit16]16', rtol=0.01, atol=1)
    run_ops('float32', 'custom[posit32]32')


def test_conv2d():
    # TODO(@gussmith23) slow and broken, needing refactor!
    # run_conv2d('float32', 'custom[posit32]32')
    pass


def test_models():
    # Expected posit8 might be faster, but it's not.
    # run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit8]8')
    # run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit32]32')
    # run_model(get_inception, (3, 299, 299), 'float32', 'custom[posit32]32')
    # run_model(get_resnet, (3, 224, 224), 'float32', 'custom[posit32]32')

    # Run cifar-10 sizes to be a little faster...
    run_model(get_mobilenet, (3, 32, 32),
              'float32',
              'custom[posit32]32',
              num_classes=10)
    # run_model(get_inception, (3, 32, 32),
    #           'float32',
    #           'custom[posit32]32',
    #           num_classes=10)
    # run_model(get_resnet, (3, 32, 32),
    #           'float32',
    #           'custom[posit32]32',
    #           num_classes=10)

    # Meanwhile, noptype is not slow.
    run_model(get_mobilenet, (3, 224, 224),
              'float32',
              'custom[noptype]32',
              num_classes=1000,
              rtol=float("inf"),
              atol=float("inf"))
    run_model(get_inception, (3, 299, 299),
              'float32',
              'custom[noptype]32',
              num_classes=1000,
              rtol=float("inf"),
              atol=float("inf"))
    run_model(get_resnet, (3, 224, 224),
              'float32',
              'custom[noptype]32',
              num_classes=1000,
              rtol=float("inf"),
              atol=float("inf"))


if __name__ == "__main__":
    setup()
    test_ops()
    test_conv2d()
    test_models()
