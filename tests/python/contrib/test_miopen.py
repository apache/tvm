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
import tvm
import tvm.testing
from tvm import te
from tvm.contrib import miopen
import numpy as np
import pytest


requires_miopen = pytest.mark.skipif(
    tvm.get_global_func("tvm.contrib.miopen.conv2d.setup", True) is None,
    reason="MIOpen is not enabled",
)


@tvm.testing.requires_rocm
@requires_miopen
def test_conv2d():
    in_channel = 3
    out_channel = 64
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [1, in_channel, 128, 128]
    wshape = (out_channel, in_channel, filter_h, filter_w)

    X = te.placeholder(xshape, name="X")
    W = te.placeholder(wshape, name="W")
    Y = miopen.conv2d_forward(
        X, W, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, conv_mode=0, data_type=1
    )

    yshape = [x.value for x in Y.shape]
    from tvm import topi

    s = te.create_schedule(Y.op)

    def verify():
        dev = tvm.rocm(0)
        f = tvm.build(s, [X, W, Y], "rocm --host=llvm", name="conv2d")
        x = tvm.nd.array(np.random.uniform(-1, 1, xshape).astype(np.float32), dev)
        w = tvm.nd.array(np.random.uniform(-1, 1, wshape).astype(np.float32), dev)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), dev)
        f(x, w, y)

        Y_ref = topi.nn.conv2d_nchw(
            X, W, (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w)
        )
        s_ref = te.create_schedule(Y_ref.op)
        f_ref = tvm.build(s_ref, [X, W, Y_ref], "rocm --host=llvm")
        y_ref = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.float32), dev)
        f_ref(x, w, y_ref)
        print("Max abs diff:", np.max(np.abs(y.numpy() - y_ref.numpy())))
        tvm.testing.assert_allclose(y.numpy(), y_ref.numpy(), atol=1e-3)

    verify()


def verify_softmax(shape, axis, dtype="float32", log_softmax=False):
    miopen_op = miopen.log_softmax if log_softmax else miopen.softmax
    testing_op = (
        tvm.topi.testing.log_softmax_python if log_softmax else tvm.topi.testing.softmax_python
    )

    A = te.placeholder(shape, dtype=dtype, name="A")
    B = miopen_op(A, axis)
    s = te.create_schedule([B.op])

    dev = tvm.rocm(0)
    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = testing_op(a_np)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    f = tvm.build(s, [A, B], target="rocm --host=llvm", name="softmax")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3)


def verify_softmax_4d(shape, dtype="float32", log_softmax=False):
    miopen_op = miopen.log_softmax if log_softmax else miopen.softmax
    testing_op = (
        tvm.topi.testing.log_softmax_python if log_softmax else tvm.topi.testing.softmax_python
    )

    A = te.placeholder(shape, dtype=dtype, name="A")
    B = miopen_op(A, axis=1)
    s = te.create_schedule([B.op])

    dev = tvm.rocm(0)
    n, c, h, w = shape
    a_np = np.random.uniform(size=shape).astype(dtype)
    b_np = testing_op(a_np.transpose(0, 2, 3, 1).reshape(h * w, c))
    b_np = b_np.reshape(n, h, w, c).transpose(0, 3, 1, 2)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    f = tvm.build(s, [A, B], target="rocm --host=llvm", name="softmax")
    f(a, b)
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3)


@tvm.testing.requires_rocm
@requires_miopen
def test_softmax():
    verify_softmax((32, 10), -1)
    verify_softmax((3, 4), -1)
    verify_softmax_4d((1, 16, 256, 256))
    verify_softmax_4d((1, 16, 256, 256))

    verify_softmax((32, 10), -1, log_softmax=True)
    verify_softmax((3, 4), -1, log_softmax=True)
    verify_softmax_4d((1, 16, 256, 256), log_softmax=True)


if __name__ == "__main__":
    test_conv2d()
