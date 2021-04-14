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

"""Relay to ONNX serialization test cases"""
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import numpy as np
import onnxruntime as rt

import tvm
from tvm import relay
from tvm.contrib.target.onnx import to_onnx


def func_to_onnx(func, name):
    mod = tvm.IRModule()
    mod["main"] = func
    onnx_model = to_onnx(mod, {}, name, path=None)
    return onnx_model.SerializeToString()


def run_onnx(onnx_model, input_data):
    sess = rt.InferenceSession(onnx_model)
    input_names = {}
    for input, data in zip(sess.get_inputs(), input_data):
        input_names[input.name] = data
    output_names = [out.name for out in sess.get_outputs()]
    res = sess.run(output_names, input_names)
    return res


def run_relay(func, data_tuple):
    target = "llvm"
    dev = tvm.device("llvm", 0)
    intrp = relay.create_executor("graph", device=dev, target=target)
    relay_res = intrp.evaluate(func)(*data_tuple)

    result = []
    relay_res = relay_res if isinstance(relay_res, list) else [relay_res]
    for res in relay_res:
        result.append(res.asnumpy())

    return result


def verify_results(relay_func, indata, test_name, rtol=1e-7, atol=0):
    relay_results = run_relay(relay_func, indata)
    onnx_results = run_onnx(func_to_onnx(relay_func, test_name), indata)

    for relay_res, onnx_res in zip(relay_results, onnx_results):
        np.testing.assert_allclose(relay_res, onnx_res, rtol=rtol, atol=atol)


def test_add():
    dtype = "float32"
    t1 = relay.TensorType((5, 10, 5))
    t2 = relay.TensorType((5, 10, 5))
    x = relay.var("x", t1, dtype=dtype)
    y = relay.var("y", t2, dtype=dtype)
    z = relay.add(x, y)
    func = relay.Function([x, y], z)

    x_data = np.random.rand(5, 10, 5).astype(dtype)
    y_data = np.random.rand(5, 10, 5).astype(dtype)

    verify_results(func, [x_data, y_data], "test_add")


def test_bias_add():
    for dtype in ["float16", "float32"]:
        xshape = (10, 2, 3, 4)
        bshape = (2,)
        rtol = 1e-2 if dtype == "float16" else 1e-5
        x = relay.var("x", shape=xshape, dtype=dtype)
        bias = relay.var("bias", dtype=dtype)
        z = relay.nn.bias_add(x, bias)
        func = relay.Function([x, bias], z)

        x_data = np.random.uniform(size=xshape).astype(dtype)
        y_data = np.random.uniform(size=bshape).astype(dtype)

        verify_results(func, [x_data, y_data], "test_bias_add", rtol=rtol)


def test_conv2d():
    def verify_conv2d(
        dtype, scale, dshape, kshape, padding=(1, 1), groups=1, dilation=(1, 1), **attrs
    ):
        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", shape=kshape, dtype=dtype)
        y = relay.nn.conv2d(x, w, padding=padding, dilation=dilation, groups=groups, **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
        kernel = np.random.uniform(-scale, scale, size=kshape).astype(dtype)
        verify_results(func, [data, kernel], "test_conv2d", rtol=1e-5, atol=1e-5)

    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=32, groups=32, kernel_size=(3, 3)
    )

    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=32, groups=8, kernel_size=(3, 3)
    )

    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    verify_conv2d(
        "float32", 1, dshape, kshape, padding=(1, 1), channels=64, groups=32, kernel_size=(3, 3)
    )

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    verify_conv2d("float32", 1, dshape, kshape, padding=(2, 2), channels=10, kernel_size=(3, 3))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    verify_conv2d(
        "float32",
        1,
        dshape,
        kshape,
        padding=(1, 1),
        channels=10,
        kernel_size=(3, 3),
        dilation=(3, 3),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 2, 2)
    verify_conv2d(
        "float32",
        1,
        dshape,
        kshape,
        padding=(2, 2),
        channels=10,
        kernel_size=(2, 2),
        dilation=(1, 1),
    )

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))

    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 4, 4)
    verify_conv2d("float32", 1, dshape, kshape, padding=(1, 1), channels=10, kernel_size=(4, 4))


def test_reshape():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape=newshape)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(func, [x_data], "test_reshape", rtol=1e-5, atol=1e-5)

    verify_reshape((2, 3, 4), tuple(np.array([4, 2, 3], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([2, 0, 0], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([0, -1], dtype=np.int64)))
    verify_reshape((2, 3, 4), tuple(np.array([-1, 0], dtype=np.int64)))


def test_transpose():
    def verify_reshape(shape, newshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.transpose(x, newshape)
        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        verify_results(func, [x_data], "test_transpose", rtol=1e-5, atol=1e-5)

    verify_reshape((1, 2, 3, 4), (0, 2, 3, 1))
    verify_reshape((1, 2, 3, 4), (0, 3, 2, 1))


def test_dense():
    def verify_dense(d_shape, w_shape):
        data = relay.var("data", relay.TensorType(d_shape, "float32"))
        weight = relay.var("weight", relay.TensorType(w_shape, "float32"))
        func = relay.Function([data, weight], relay.nn.dense(data, weight))
        x_data = np.random.uniform(size=d_shape).astype("float32")
        w_data = np.random.uniform(size=w_shape).astype("float32")
        verify_results(func, [x_data, w_data], "test_dense", rtol=1e-5, atol=1e-5)

    verify_dense((1, 8), (16, 8))
    verify_dense((1, 4), (3, 4))


def test_max_pool():
    def verify_max_pool(x_shape, pool_size, strides, padding, ceil_mode):
        x = relay.var("x", relay.TensorType(x_shape, "float32"))
        y = tvm.relay.nn.max_pool2d(
            x, pool_size=pool_size, strides=strides, padding=padding, ceil_mode=ceil_mode
        )
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=x_shape).astype("float32")
        verify_results(func, [x_data], "test_max_pool", rtol=1e-5, atol=1e-5)

    verify_max_pool(
        (1, 4, 16, 16), pool_size=(2, 2), strides=(2, 2), padding=(0, 0), ceil_mode=False
    )


def test_batch_flatten():
    def verify_test_batch_flatten(d_shape):
        data = relay.var("data", relay.TensorType(d_shape, "float32"))
        func = relay.Function([data], relay.nn.batch_flatten(data))
        x_data = np.random.uniform(size=d_shape).astype("float32")
        verify_results(func, [x_data], "test_batch_flatten", rtol=1e-5, atol=1e-5)

    verify_test_batch_flatten((1, 2, 3, 4))
    verify_test_batch_flatten((1, 8))


def test_batch_norm():
    def verify_batch_norm(axis=1):
        for dtype in ["float16", "float32"]:
            data = relay.var("data", relay.TensorType((2, 4, 4, 1), dtype))
            gamma_shape = (data.type_annotation.shape[axis].value,)
            beta = relay.var("beta", relay.TensorType(gamma_shape, dtype))
            gamma = relay.var("gamma", relay.TensorType(gamma_shape, dtype))
            moving_mean = relay.var("moving_mean", relay.TensorType(gamma_shape, dtype))
            moving_var = relay.var("moving_var", relay.TensorType(gamma_shape, dtype))
            y = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var, axis=axis)
            func = relay.Function([data, gamma, beta, moving_mean, moving_var], y[0])

            x_data = np.random.uniform(size=(2, 4, 4, 1)).astype(dtype)
            beta = np.random.uniform(size=gamma_shape).astype(dtype)
            gamma = np.random.uniform(size=gamma_shape).astype(dtype)
            moving_mean = np.random.uniform(size=gamma_shape).astype(dtype)
            moving_var = np.random.uniform(size=gamma_shape).astype(dtype)
            verify_results(
                func,
                [x_data, gamma, beta, moving_mean, moving_var],
                "test_batch_norm",
                rtol=1e-1,
                atol=1e-1,
            )

    verify_batch_norm(axis=1)
    verify_batch_norm(axis=3)


def test_pad():
    def verify_pad():
        for dtype in ["float16", "float32"]:
            dshape = (4, 10, 7, 7)
            x = relay.var("x", shape=dshape, dtype=dtype)
            y = relay.nn.pad(x, ((1, 1), (2, 2), (3, 3), (4, 4)))
            func = relay.Function([x], y)
            x_data = np.random.uniform(size=dshape).astype(dtype)
            verify_results(func, [x_data], "test_pad", rtol=1e-5, atol=1e-5)

    verify_pad()


def test_sofmax():
    def verify_sofmax():
        for dtype in ["float32"]:
            shape = (10, 4)
            x = relay.var("x", shape=shape, dtype=dtype)
            y = relay.nn.softmax(x, axis=1)
            func = relay.Function([x], y)
            x_data = np.random.uniform(size=shape).astype(dtype)
            verify_results(func, [x_data], "test_softmax", rtol=1e-5, atol=1e-5)

    verify_sofmax()


def test_squeeze():
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        z = relay.squeeze(x, axis=axis)
        func = relay.Function([x], z)
        x_data = np.random.random_sample(shape).astype(dtype)
        verify_results(func, [x_data], "test_squeeze", rtol=1e-5, atol=1e-5)

    verify_squeeze((1, 3, 2, 5), "float32", None)
    verify_squeeze(
        (1, 3, 1),
        "float32",
        [
            2,
        ],
    )
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


def test_mean():
    def verify_mean(data_shape, axis, exclude, keepdims):
        dtype = "float32"
        x = relay.var("x", shape=data_shape, dtype=dtype)
        y = relay.mean(x, axis, keepdims, exclude)
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=data_shape).astype(dtype)
        verify_results(func, [x_data], "test_mean", rtol=1e-5, atol=1e-5)

    verify_mean((1, 2), 0, False, False)
    verify_mean((1, 2), 0, True, False)
    verify_mean((1, 2), 0, True, True)
    verify_mean((1, 2), 1, True, True)
    verify_mean((3, 2, 1), 1, False, True)


def test_split():
    def verify_split(dshape, indices_or_sections, axis=None):
        dtype = "float32"
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)
        func = relay.Function([x], y.astuple())
        x_data = np.random.uniform(size=dshape).astype(dtype)

        verify_results(func, [x_data], "test_split", rtol=1e-5, atol=1e-5)

    verify_split((5, 5, 2, 2), 5, axis=1)
    verify_split((5, 5, 2, 2), 5, axis=0)
    verify_split((5, 5, 2, 2), [1, 3, 4], axis=0)
    verify_split((5, 5, 2, 2), [1, 3, 4], axis=1)


def test_concatenate():
    def verify_concatenate(shapes, axis, dtype="float32"):
        in_vars = []
        in_data = []
        for i, shape in enumerate(shapes):
            in_vars.append(relay.var("x" + str(i), relay.ty.TensorType(shape, dtype)))
            in_data.append(np.random.uniform(size=shape).astype(dtype))

        out_tensor = relay.concatenate(in_vars, axis)
        func = relay.Function(in_vars, out_tensor)
        verify_results(func, in_data, "test_concatenate", rtol=1e-5, atol=1e-5)

    verify_concatenate([(2,), (2,), (2,)], -1)
    verify_concatenate([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1)
    verify_concatenate([(1, 2, 4), (1, 2, 3), (1, 2, 7), (1, 2, 8), (1, 2, 1)], -1)
    verify_concatenate([(5, 6, 7, 3), (16, 6, 7, 3), (12, 6, 7, 3), (8, 6, 7, 3), (2, 6, 7, 3)], 0)
    verify_concatenate([(1, 14400), (1, 2400), (1, 640), (1, 240)], 1)


def test_strided_slice():
    def verify_strided_slice(dshape, begin, end, strides, mode):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        if mode == "size":
            strides = None
        z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=mode)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=dshape).astype("float32")
        verify_results(func, [x_data], "test_strided_slice", rtol=1e-5, atol=1e-5)

    for mode in ["end", "size"]:
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 2, 3], None, mode)
        verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -1, 3], [1, 2], mode)
        verify_strided_slice(
            (3, 4, 3),
            [
                1,
            ],
            [4, -3],
            None,
            mode,
        )
        verify_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], mode)
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, -3], [2, 1, 1], mode)
        verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], mode)
        verify_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2], mode)
        verify_strided_slice((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], mode)

        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, mode)
        verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4], None, mode)
        verify_strided_slice((3, 4, 3), [1, 1], [4, 4, 3], None, mode)
        verify_strided_slice((3, 4, 3), [1, 1], [4, 4, 3], [1, 1, 2], mode)


def test_cmp_type():
    for op, ref in ((relay.greater, np.greater), (relay.less, np.less), (relay.equal, np.equal)):
        x_shape = (10, 4)
        y_shape = (5, 10, 1)
        t1 = relay.TensorType(x_shape)
        t2 = relay.TensorType(y_shape)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = op(x, y)
        x_data = np.random.rand(*x_shape).astype(t1.dtype)
        y_data = np.random.rand(*y_shape).astype(t2.dtype)
        func = relay.Function([x, y], z)
        verify_results(func, [x_data, y_data], "test_cmp_type", rtol=1e-5, atol=1e-5)


def test_unary_identity():
    for dtype in ["int16", "float32", "float64"]:
        for op, ref in [(relay.zeros_like, np.zeros_like), (relay.ones_like, np.ones_like)]:
            shape = (8, 9, 4)
            x = relay.var("x", relay.TensorType(shape, dtype))
            y = op(x)
            func = relay.Function(
                [
                    x,
                ],
                y,
            )
            x_data = np.random.rand(*shape).astype(dtype)
            verify_results(func, [x_data], "test_cmp_type", rtol=1e-5, atol=1e-5)


def test_binary_op():
    def check_binary_op(opfunc, dtype):
        t1 = relay.TensorType((5, 10, 5))
        t2 = relay.TensorType((5, 10, 5))
        x = relay.var("x", t1, dtype=dtype)
        y = relay.var("y", t2, dtype=dtype)
        z = opfunc(x, y)
        x_data = np.random.rand(5, 10, 5).astype(dtype)
        y_data = np.random.rand(5, 10, 5).astype(dtype)
        func = relay.Function([x, y], z)
        verify_results(func, [x_data, y_data], "test_binary_op", rtol=1e-5, atol=1e-5)

    for opfunc, ref in [
        (relay.add, np.add),
        (relay.subtract, np.subtract),
        (relay.multiply, np.multiply),
        (relay.divide, np.divide),
    ]:
        for dtype in ["float32"]:
            check_binary_op(opfunc, dtype)


def test_tuple_types():
    def verify_tuple_types(dshape, indices_or_sections, axis=None, dtype="float32"):
        x = relay.var("x", relay.ty.TensorType(dshape, dtype))
        y = relay.split(x, indices_or_sections, axis=axis)
        z = relay.concatenate(y, axis=axis)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=dshape).astype(dtype)
        verify_results(func, [x_data], "test_tuple_types", rtol=1e-5, atol=1e-5)

        split_z = relay.split(z, indices_or_sections, axis=axis)
        func = relay.Function([x], split_z.astuple())
        verify_results(func, [x_data], "test_tuple_types", rtol=1e-5, atol=1e-5)

        out = relay.Tuple([y[0] + y[1], y[0] - y[1]])
        func = relay.Function([x], out)
        verify_results(func, [x_data], "test_tuple_types", rtol=1e-5, atol=1e-5)

        z = relay.concatenate(out, axis=axis)
        func = relay.Function([x], z)
        verify_results(func, [x_data], "test_tuple_types", rtol=1e-5, atol=1e-5)

    verify_tuple_types((5, 5, 2, 2), 5, axis=1)
    verify_tuple_types((5, 5, 2, 2), 5, axis=0)
    verify_tuple_types((5, 5, 2, 2), [1, 3, 4], axis=0)
    verify_tuple_types((5, 5, 2, 2), [1, 3, 4], axis=1)


def test_layout_transform():
    def verify_layout_transform(dshape, src_layout, dst_layout, dtype="float32"):
        x = relay.var("x", relay.ty.TensorType(dshape, dtype))
        y = relay.layout_transform(x, src_layout, dst_layout)
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=dshape).astype(dtype)
        verify_results(func, [x_data], "test_layout_transform", rtol=1e-5, atol=1e-5)

    verify_layout_transform((1, 3, 8, 8), "NCHW", "NHWC")
    verify_layout_transform((1, 8, 8, 3), "NHWC", "NCHW")


def test_clip():
    def verify_clip(dshape, a_min, a_max, dtype="float32"):
        x = relay.var("x", relay.ty.TensorType(dshape, dtype))
        y = relay.clip(x, a_min, a_max)
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=dshape).astype(dtype)
        verify_results(func, [x_data], "test_clip", rtol=1e-5, atol=1e-5)

    verify_clip((5, 5, 2, 5), 0, 0.2)
    verify_clip((5, 5, 2, 5), 0.2, 0.5)


def test_expand_dims():
    def verify_expand_dims(dshape, axis, num_newaxis, dtype="float32"):
        x = relay.var("x", relay.ty.TensorType(dshape, dtype))
        y = relay.expand_dims(x, axis, num_newaxis)
        func = relay.Function([x], y)
        x_data = np.random.uniform(size=dshape).astype(dtype)
        verify_results(func, [x_data], "test_expand_dims", rtol=1e-5, atol=1e-5)

    verify_expand_dims((1, 1001), 0, 2)
    verify_expand_dims((1, 1, 1001), 2, 2)


if __name__ == "__main__":
    test_add()
    test_bias_add()
    test_conv2d()
    test_reshape()
    test_transpose()
    test_dense()
    test_max_pool()
    test_batch_flatten()
    test_batch_norm()
    test_pad()
    test_mean()
    test_split()
    test_concatenate()
    test_sofmax()
    test_squeeze()
    test_strided_slice()
    test_cmp_type()
    test_binary_op()
    test_tuple_types()
    test_layout_transform()
    test_clip()
    test_expand_dims()
