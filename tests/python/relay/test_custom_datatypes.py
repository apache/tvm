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
"""Unit tests for the Bring Your Own Datatype framework.

TODO(@gussmith23 @hypercubestart) link to documentation"""
import numpy as np
import pytest
import tvm
import tvm.topi.testing
import tvm.testing
from tvm import relay
from tvm.relay.testing.layers import batch_norm_infer
from tvm.target.datatype import (
    create_lower_func,
    create_min_lower_func,
    lower_call_pure_extern,
    lower_ite,
    register,
    register_min_func,
    register_op,
)
from tvm.tir.op import call_pure_extern
from tvm.script import tir as T


# note: we can't use relay.testing models because params are randomly initialized,
# which lead the output to have the same values
# get mobilenet model from Gluon CV
# because: https://discuss.tvm.apache.org/t/mobilenet-intermediate-values-are-0/7812
def get_mobilenet():
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenet0.25", pretrained=True)
    shape_dict = {"data": dshape}
    return relay.frontend.from_mxnet(block, shape_dict)


# use real image instead of random data for end-to-end model training
# or else output would all be around the same value
def get_cat_image(dimensions):
    from PIL import Image
    from tvm.contrib.download import download_testdata

    url = "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize(dimensions)
    # CoreML's standard model image format is BGR
    img_bgr = np.array(img)[:, :, ::-1]
    img = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
    return np.asarray(img, dtype="float32")


# we use a random seed to generate input_data
# to guarantee stable tests
np.random.seed(0)


def convert_ndarray(dst_dtype, array):
    """Converts NDArray(s) into the specified datatype"""
    x = relay.var("x", shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor("graph").evaluate(cast)(array)


def change_dtype(src, dst, module, params):
    """Convert constants and functions in module from src type to dst type.
    Returns changed module and converted params of type dst_type.
    """
    module = relay.frontend.ChangeDatatype(src, dst)(module)
    module = relay.transform.InferType()(module)
    params = {k: convert_ndarray(dst, v) for k, v in params.items()}
    return module, params


def compare(module, input, src_dtype, dst_dtype, rtol, atol, params={}, target="llvm"):
    module = relay.transform.InferType()(module)
    module = relay.transform.SimplifyInference()(module)

    correct = relay.create_executor("graph", mod=module).evaluate()(*input, **params)
    module, converted_params = change_dtype(src_dtype, dst_dtype, module, params)
    # converts all inputs to dst_dtype
    x_converted = [convert_ndarray(dst_dtype, arr) for arr in input]

    # Vectorization is not implemented with custom datatypes
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        maybe_correct = relay.create_executor("graph", mod=module, target=target).evaluate()(
            *x_converted, **converted_params
        )
        # currently this only works for comparing single output
        maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct)
    np.testing.assert_allclose(
        maybe_correct_converted.numpy(), correct.numpy(), rtol=rtol, atol=atol
    )


def setup_myfloat():
    """Set up tests for myfloat (a custom datatype that under the hood is float)

    Currently, this registers some custom datatypes using the Bring Your
    Own Datatypes framework.
    """

    def _setup_myfloat_inner():
        # To use datatype operations in an external library, you should first load
        # the library containing the datatype implementation:
        # CDLL("libposit.so", RTLD_GLOBAL)
        # In this case, the datatype library we are using is built right into TVM,
        # so we do not need to explicitly load any library.

        # You can pick a code for your datatype arbitrarily, as long as it is
        # greater than 128 and has not already been chosen.
        register("myfloat", 131)

        register_op(
            create_lower_func({(32, 32): "FloatToCustom32"}), "Cast", "llvm", "float", "myfloat"
        )
        register_op(
            create_lower_func({(32, 32): "Custom32ToFloat"}), "Cast", "llvm", "myfloat", "float"
        )
        register_op(create_lower_func({32: "Custom32Add"}), "Add", "llvm", "myfloat")
        register_op(
            create_lower_func(
                {
                    32: "Custom32Sub",
                }
            ),
            "Sub",
            "llvm",
            "myfloat",
        )
        register_op(create_lower_func({32: "Custom32Mul"}), "Mul", "llvm", "myfloat")
        register_op(
            create_lower_func(
                {
                    32: "FloatToCustom32",
                }
            ),
            "FloatImm",
            "llvm",
            "myfloat",
        )
        register_op(
            create_lower_func(
                {
                    32: "Custom32Div",
                }
            ),
            "Div",
            "llvm",
            "myfloat",
        )
        register_op(create_lower_func({32: "Custom32Max"}), "Max", "llvm", "myfloat")
        register_op(
            create_lower_func({32: "Custom32Sqrt"}),
            "Call",
            "llvm",
            "myfloat",
            intrinsic_name="tir.sqrt",
        )
        register_op(
            create_lower_func({32: "Custom32Exp"}),
            "Call",
            "llvm",
            "myfloat",
            intrinsic_name="tir.exp",
        )
        register_op(
            create_lower_func({32: "Custom32Log"}),
            "Call",
            "llvm",
            "myfloat",
            intrinsic_name="tir.log",
        )
        register_op(
            create_lower_func({32: "Custom32Sigmoid"}),
            "Call",
            "llvm",
            "myfloat",
            intrinsic_name="tir.sigmoid",
        )
        register_op(
            create_lower_func({32: "Custom32Tanh"}),
            "Call",
            "llvm",
            "myfloat",
            intrinsic_name="tir.tanh",
        )
        register_op(lower_ite, "Call", "llvm", "myfloat", intrinsic_name="tir.if_then_else")
        register_op(
            lower_call_pure_extern, "Call", "llvm", "myfloat", intrinsic_name="tir.call_pure_extern"
        )

        register_min_func(create_min_lower_func({32: "MinCustom32"}, "myfloat"), "myfloat")

    try:
        _setup_myfloat_inner()
    except tvm._ffi.base.TVMError as e:
        # Ignore this specific error which can happen if another test
        # that uses "myfloat" has already run.
        if "float is already registered" not in str(e):
            raise e


def setup_posites2():
    """Set up tests for posites2
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

    register("posites2", 132)

    register_op(
        create_lower_func(
            {
                (32, 32): "FloatToPosit32es2",
                (32, 16): "FloatToPosit16es2",
                (32, 8): "FloatToPosit8es2",
            }
        ),
        "Cast",
        "llvm",
        "float",
        "posites2",
    )
    register_op(
        create_lower_func(
            {
                (32, 32): "Posit32es2ToFloat",
                (16, 32): "Posit16es2ToFloat",
                (8, 32): "Posit8es2ToFloat",
            }
        ),
        "Cast",
        "llvm",
        "posites2",
        "float",
    )
    register_op(
        create_lower_func({32: "Posit32es2Add", 16: "Posit16es2Add", 8: "Posit8es2Add"}),
        "Add",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Sub", 16: "Posit16es2Sub", 8: "Posit8es2Sub"}),
        "Sub",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func(
            {32: "FloatToPosit32es2", 16: "FloatToPosit16es2", 8: "FloatToPosit8es2"}
        ),
        "FloatImm",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Mul", 16: "Posit16es2Mul", 8: "Posit8es2Mul"}),
        "Mul",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Div", 16: "Posit16es2Div", 8: "Posit8es2Div"}),
        "Div",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Max", 16: "Posit16es2Max", 8: "Posit8es2Max"}),
        "Max",
        "llvm",
        "posites2",
    )
    register_op(
        create_lower_func({32: "Posit32es2Sqrt", 16: "Posit16es2Sqrt", 8: "Posit8es2Sqrt"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.sqrt",
    )
    register_op(lower_ite, "Call", "llvm", "posites2", intrinsic_name="tir.if_then_else")
    register_op(
        lower_call_pure_extern, "Call", "llvm", "posites2", intrinsic_name="tir.call_pure_extern"
    )
    register_op(
        create_lower_func({32: "Posit32es2Exp", 16: "Posit16es2Exp", 8: "Posit8es2Exp"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.exp",
    )
    register_op(
        create_lower_func({32: "Posit32es2Log", 16: "Posit16es2Log", 8: "Posit8es2Log"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.log",
    )
    register_op(
        create_lower_func(
            {32: "Posit32es2Sigmoid", 16: "Posit16es2Sigmoid", 8: "Posit8es2Sigmoid"}
        ),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.sigmoid",
    )
    register_op(
        create_lower_func({32: "Posit32es2Tanh", 16: "Posit16es2Tanh", 8: "Posit8es2Tanh"}),
        "Call",
        "llvm",
        "posites2",
        intrinsic_name="tir.tanh",
    )

    register_min_func(
        create_min_lower_func(
            {32: "MinPosit32es2", 16: "MinPosit16es2", 8: "MinPosit8es2"}, "posites2"
        ),
        "posites2",
    )


def run_ops(src_dtype, dst_dtype, rtol=1e-7, atol=1e-7):
    """Run the same op, but with two different datatypes"""
    # used for unary ops, first shape in binary ops
    shape1 = (5, 10, 5)
    # second shape for binary ops
    shape2 = (5,)

    def check_unary_op(op, src_dtype, dst_dtype, shape):
        t1 = relay.TensorType(shape, src_dtype)
        x = relay.var("x", t1)
        z = op(x)
        x_data = np.random.rand(*shape).astype(t1.dtype)

        module = tvm.IRModule.from_expr(relay.Function([x], z))

        compare(module, (x_data,), src_dtype, dst_dtype, rtol, atol)

    # test unary ops
    for op in [
        relay.nn.softmax,
        tvm.relay.log,
        tvm.relay.exp,
        tvm.relay.sqrt,
        tvm.relay.rsqrt,
        tvm.relay.sigmoid,
        tvm.relay.tanh,
        relay.nn.relu,
        relay.nn.batch_flatten,
    ]:
        check_unary_op(op, src_dtype, dst_dtype, shape1)

    # test unary ops over 4d data
    for op in [relay.nn.max_pool2d, relay.nn.avg_pool2d, relay.nn.global_avg_pool2d]:
        shape_2d = (3, 32, 32, 32)
        check_unary_op(op, src_dtype, dst_dtype, shape_2d)

    def check_binary_op(opfunc, src_dtype, dst_dtype):
        t1 = relay.TensorType(shape1, src_dtype)
        t2 = relay.TensorType(shape2, src_dtype)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        x_data = np.random.rand(*shape1).astype(t1.dtype)
        y_data = np.random.rand(*shape2).astype(t2.dtype)
        module = tvm.IRModule.from_expr(relay.Function([x, y], z))

        compare(module, (x_data, y_data), src_dtype, dst_dtype, rtol, atol)

    for op in [
        relay.add,
        relay.subtract,
        relay.divide,
        relay.multiply,
    ]:
        check_binary_op(op, src_dtype, dst_dtype)

    # we would like to test tvm_if_then_else
    # but Relay.IfNode is not lowered to this intrinsic,
    # so to keep our tests consistent with relay, we decide to not unit test
    # Note: tvm_if_then_else is tested as part of the mobile_net model


def run_model(get_workload, input, src_dtype, dst_dtype, rtol=1e-4, atol=1e-4):
    module, params = get_workload()

    # we don't generate random data here
    # because then the output data would all be around the same value
    compare(module, input, src_dtype, dst_dtype, rtol, atol, params)


def run_conv2d(src_dtype, dst_dtype, rtol=1e-7, atol=1e-4):
    def run_test_conv2d(
        src_dtype,
        dst_dtype,
        scale,
        dshape,
        kshape,
        padding=(1, 1),
        groups=1,
        dilation=(1, 1),
        **attrs,
    ):
        x = relay.var("x", shape=dshape, dtype=src_dtype)
        w = relay.var("w", shape=kshape, dtype=src_dtype)
        y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        module = tvm.IRModule.from_expr(relay.Function([x, w], y))
        data = np.random.uniform(-scale, scale, size=dshape).astype(src_dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(src_dtype)

        compare(module, (data, kernel), src_dtype, dst_dtype, rtol, atol)

    # depthwise conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    run_test_conv2d(
        src_dtype,
        dst_dtype,
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=32,
        groups=32,
        kernel_size=(3, 3),
    )

    # CUDA is disabled for 'direct' schedule:
    # https://github.com/dmlc/tvm/pull/3070#issuecomment-486597553
    # group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    run_test_conv2d(
        src_dtype,
        dst_dtype,
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=32,
        groups=8,
        kernel_size=(3, 3),
    )
    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    run_test_conv2d(
        src_dtype,
        dst_dtype,
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=64,
        groups=32,
        kernel_size=(3, 3),
    )

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(
        src_dtype, dst_dtype, 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(3, 3)
    )

    # dilated conv2d
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(
        src_dtype,
        dst_dtype,
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=10,
        kernel_size=(3, 3),
        dilation=(3, 3),
    )


def run_batchnorm(src_dtype, dst_dtype, rtol=1e-6, atol=1e-6):
    shape = (3, 32, 32)
    t = relay.TensorType(shape, src_dtype)
    x = relay.var("x", t)
    bn = batch_norm_infer(data=x, epsilon=2e-5, scale=False, name="bn_x")
    f = relay.Function(relay.analysis.free_vars(bn), bn)

    x_data = np.random.rand(*shape).astype(t.dtype)
    module = tvm.IRModule.from_expr(f)

    zero_data = np.zeros((32), "float32")
    compare(
        module,
        (x_data, zero_data, zero_data, zero_data, zero_data),
        src_dtype,
        dst_dtype,
        rtol,
        atol,
    )


def test_myfloat():
    setup_myfloat()

    run_ops("float32", "custom[myfloat]32", rtol=1e-6, atol=1e-6)
    run_conv2d("float32", "custom[myfloat]32", rtol=1e-6, atol=1e-6)
    run_batchnorm("float32", "custom[myfloat]32", rtol=1e-6, atol=1e-6)

    # mxnet python package not available
    # run_model(get_mobilenet, (get_cat_image((224, 224)), ),
    #           'float32',
    #           'custom[myfloat]32')


class TestMyfloatLowering(tvm.testing.CompareBeforeAfter):
    setup_myfloat()

    transform = tvm.tir.transform.LowerCustomDatatypes()

    def before(self):
        dtype = "custom[myfloat]32"

        @T.prim_func
        def func(A_data: T.handle(dtype)):
            T.func_attr({"target": T.target("llvm")})
            A = T.Buffer(16, dtype=dtype, data=A_data)
            B_data = T.allocate([16], dtype=dtype)
            B = T.Buffer(16, dtype=dtype, data=B_data)
            for i in range(16):
                B[i] = A[i] + 1.0

        return func

    def expected(self):
        dtype = "custom[myfloat]32"

        @T.prim_func
        def func(A_data: T.handle(dtype)):
            T.func_attr({"target": T.target("llvm")})
            A_uint32 = T.Buffer(16, "uint32", data=A_data)
            B_data = T.allocate([16], dtype="uint32")
            B_uint32 = T.Buffer(16, "uint32", data=B_data)
            for i in range(16):
                B_uint32[i] = T.call_pure_extern(
                    "uint32",
                    "FloatToCustom32",
                    T.call_pure_extern("float32", "Custom32ToFloat", A_uint32[i]) + T.float32(1),
                )

        return func


class TestMyfloatLoweringDeclBuffer(tvm.testing.CompareBeforeAfter):
    """Like TestMyfloatLoweringDeclBuffer, but using DeclBuffer"""

    setup_myfloat()

    transform = tvm.tir.transform.LowerCustomDatatypes()

    def before(self):
        dtype = "custom[myfloat]32"

        @T.prim_func
        def func(A_data: T.handle(dtype)):
            T.func_attr({"target": T.target("llvm")})
            A = T.decl_buffer(16, dtype=dtype, data=A_data)
            B = T.decl_buffer(16, dtype=dtype)
            for i in range(16):
                B[i] = A[i] + 1.0

        return func

    def expected(self):
        dtype = "custom[myfloat]32"

        @T.prim_func
        def func(A_data: T.handle(dtype)):
            T.func_attr({"target": T.target("llvm")})
            A_uint32 = T.decl_buffer(16, "uint32", data=A_data)
            B_uint32 = T.decl_buffer(16, dtype="uint32")
            for i in range(16):
                B_uint32[i] = T.call_pure_extern(
                    "uint32",
                    "FloatToCustom32",
                    T.call_pure_extern("float32", "Custom32ToFloat", A_uint32[i]) + T.float32(1),
                )

        return func


def _has_posit():
    return tvm.support.libinfo()["USE_BYODT_POSIT"] == "ON"


@pytest.mark.skipif(not _has_posit(), reason="compiled with USE_BYODT_POSIT flag OFF")
def test_posites2():
    setup_posites2()
    run_ops("float32", "custom[posites2]8", rtol=1, atol=1)
    run_ops("float32", "custom[posites2]16", rtol=0.01, atol=1)
    run_ops("float32", "custom[posites2]32", rtol=1e-6, atol=1e-6)

    run_conv2d("float32", "custom[posites2]8", rtol=1, atol=1)
    run_conv2d("float32", "custom[posites2]16", rtol=0.01, atol=1)
    run_conv2d("float32", "custom[posites2]32")

    run_batchnorm("float32", "custom[posites2]8", rtol=1, atol=1)
    run_batchnorm("float32", "custom[posites2]16", rtol=0.01, atol=1)
    run_batchnorm("float32", "custom[posites2]32", rtol=1e-4, atol=1e-4)
    # Expected posit8 might be faster, but it's not.
    # run_model(get_mobilenet, (get_cat_image((224, 224)), ), 'float32', 'custom[posit8]8')
    # run_model(get_mobilenet, (get_cat_image((224, 224)), ), 'float32', 'custom[posit32]32')
    # run_model(get_inception, (get_cat_image((229, 229)), ), 'float32', 'custom[posit32]32')
    # run_model(get_resnet, (get_cat_image((224, 224)), ), 'float32', 'custom[posit32]32')

    # can't run cifar-10 sizes because dimensions
    # don't match pretrained weights

    # runs on the order of minutes...
    # run_model(get_inception, (get_cat_image((229, 229)), ),
    #           'float32',
    #           'custom[posites2]32')
    # run_model(get_resnet, (get_cat_image((224, 224)), ),
    #           'float32',
    #           'custom[posites2]32')


if __name__ == "__main__":
    tvm.testing.main()
