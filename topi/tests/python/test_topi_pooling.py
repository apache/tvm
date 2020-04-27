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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Test code for pooling"""
import math
import numpy as np
import tvm
from tvm import te
import topi
import topi.testing
from topi.util import get_const_tuple
from common import get_all_backend

_pool_schedule = {
    "generic": topi.generic.schedule_pool,
    "cpu": topi.x86.schedule_pool,
    "gpu": topi.cuda.schedule_pool,
    "hls": topi.hls.schedule_pool,
}

_adaptive_pool_schedule = {
    "generic": topi.generic.schedule_adaptive_pool,
    "cpu": topi.x86.schedule_adaptive_pool,
    "gpu": topi.cuda.schedule_adaptive_pool,
    "hls": topi.hls.schedule_adaptive_pool,
}

_pool_grad_schedule = {
    "generic": topi.generic.schedule_pool_grad,
    "gpu": topi.cuda.schedule_pool_grad,
}

def verify_pool(n, ic, ih, kh, sh, padding, pool_type, ceil_mode, count_include_pad=True):
    """verify function of pool"""
    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    layout = "NCHW"
    A = te.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode,
                     layout="NCHW", count_include_pad=count_include_pad)
    B = topi.nn.relu(B)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    pad_np = np.zeros(shape=(n, ic, ih+pt+pb, iw+pl+pr)).astype(dtype)
    no_zero = (range(n), range(ic), (range(pt, ih+pt)), (range(pl, iw+pl)))
    pad_np[np.ix_(*no_zero)] = a_np
    _, oc, oh, ow = get_const_tuple(B.shape)
    b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)

    if pool_type == 'avg':
        for i in range(oh):
            for j in range(ow):
                if count_include_pad:
                    b_np[:, :, i, j] = \
                            np.mean(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))
                else:
                    pad_count = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2, 3))
                    b_np[:, :, i, j] = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3)) \
                                       / np.maximum(pad_count, 1)

    elif pool_type == 'max':
        for i in range(oh):
            for j in range(ow):
                b_np[:, :, i, j] = np.max(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))
    b_np = np.maximum(b_np, 0.0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _pool_schedule)
            s = s_func(B, layout)

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=2e-5, atol=1e-5)

    for device in get_all_backend():
        check_device(device)

def verify_pool_grad(n, ic, ih, kh, sh, padding, pool_type, ceil_mode, count_include_pad=True,
                     add_relu=False):
    """verify function of pool_grad"""
    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    A = te.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode,
                     layout="NCHW", count_include_pad=count_include_pad)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)
    OutGrad = te.placeholder(bshape, name='OutGrad')
    PoolGrad = topi.nn.pool_grad(OutGrad, A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                                 pool_type=pool_type, ceil_mode=ceil_mode,
                                 layout="NCHW", count_include_pad=count_include_pad)
    if add_relu:
        PoolGrad = topi.nn.relu(PoolGrad)

    a_np = np.random.uniform(low=0.001, size=(n, ic, ih, iw)).astype(dtype)
    out_grad_np = np.random.uniform(low=0.001, size=bshape).astype(dtype)
    pool_grad_np = topi.testing.pool_grad_nchw(a_np, out_grad_np, pool_size=(kh, kw),
                                               strides=(sh, sw), padding=padding,
                                               pool_type=pool_type, ceil_mode=ceil_mode,
                                               count_include_pad=count_include_pad)
    if add_relu:
        pool_grad_np = np.maximum(pool_grad_np, 0.)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _pool_grad_schedule)
            s = s_func(PoolGrad)

        a = tvm.nd.array(a_np, ctx)
        out_grad = tvm.nd.array(out_grad_np, ctx)
        pool_grad = tvm.nd.array(np.zeros(get_const_tuple(PoolGrad.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, OutGrad, PoolGrad], device)
        f(a, out_grad, pool_grad)
        tvm.testing.assert_allclose(pool_grad.asnumpy(), pool_grad_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_pool():
    """test cases of pool"""
    verify_pool(1, 256, 32, 2, 2, [0, 0, 0, 0], 'avg', False, True)
    verify_pool(1, 256, 31, 3, 3, [1, 2, 1, 2], 'avg', False, True)
    verify_pool(1, 256, 32, 2, 2, [1, 2, 1, 2], 'avg', False, False)
    verify_pool(1, 256, 31, 4, 4, [3, 3, 3, 3], 'avg', False, False)
    verify_pool(1, 256, 31, 4, 4, [0, 0, 0, 0], 'avg', False, False)
    verify_pool(1, 256, 32, 2, 2, [0, 0, 0, 0], 'max', False)
    verify_pool(1, 256, 31, 3, 3, [2, 1, 2, 1], 'max', False)
    verify_pool(1, 256, 31, 3, 3, [2, 1, 2, 1], 'max', True)

    verify_pool(1, 256, 31, 3, 3, [2, 1, 0, 3], 'avg', False, True)
    verify_pool(1, 256, 32, 2, 2, [0, 3, 2, 1], 'avg', False, False)
    verify_pool(1, 256, 31, 3, 3, [1, 0, 3, 2], 'max', False)
    verify_pool(1, 256, 31, 3, 3, [3, 2, 1, 0], 'max', True)

def test_pool_grad():
    """test cases of pool_grad"""
    verify_pool_grad(1, 256, 32, 3, 2, [1, 1, 1, 1], 'avg', False, False)
    verify_pool_grad(1, 256, 32, 2, 2, [0, 0, 0, 0], 'avg', False, True)
    verify_pool_grad(1, 256, 31, 3, 3, [1, 2, 1, 2], 'avg', False, True)
    verify_pool_grad(1, 256, 32, 2, 2, [1, 2, 1, 2], 'avg', False, False)
    verify_pool_grad(1, 256, 31, 4, 4, [2, 2, 2, 2], 'avg', False, False)
    verify_pool_grad(1, 256, 31, 4, 4, [0, 0, 0, 0], 'avg', False, False)
    verify_pool_grad(1, 256, 32, 2, 2, [0, 0, 0, 0], 'max', False)
    verify_pool_grad(1, 256, 31, 3, 3, [2, 1, 2, 1], 'max', False)
    verify_pool_grad(1, 256, 31, 3, 3, [2, 1, 2, 1], 'max', True)

    verify_pool_grad(1, 256, 31, 3, 3, [2, 1, 0, 3], 'avg', False, True)
    verify_pool_grad(1, 256, 32, 2, 2, [0, 3, 2, 1], 'avg', False, False)
    verify_pool_grad(1, 256, 31, 3, 3, [1, 0, 3, 2], 'max', False)
    verify_pool_grad(1, 256, 31, 3, 3, [3, 2, 1, 0], 'max', True)
    verify_pool_grad(1, 256, 32, 3, 2, [1, 1, 1, 1], 'max', False)
    verify_pool_grad(1, 256, 32, 1, 2, [1, 1, 1, 1], 'avg', False, False)

    verify_pool_grad(1, 256, 31, 4, 4, [0, 0, 0, 0], 'avg', False, False, add_relu=True)
    verify_pool_grad(1, 256, 32, 2, 2, [0, 0, 0, 0], 'max', False, add_relu=True)


def verify_global_pool(dshape, pool_type, layout='NCHW'):
    """verify function of global_pool"""
    assert layout in ["NCHW", "NHWC"]
    A = te.placeholder(shape=dshape, name='A')
    B = topi.nn.global_pool(A, pool_type=pool_type, layout=layout)
    B = topi.nn.relu(B)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)

    axis = (layout.find('H'), layout.find('W'))
    if pool_type == 'avg':
        b_np = np.mean(a_np, axis=axis, keepdims=True)
    elif pool_type == 'max':
        b_np = np.max(a_np, axis=axis, keepdims=True)
    b_np = np.maximum(b_np, 0.0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _adaptive_pool_schedule)
            if device == "cuda":
                s = s_func(B, layout)
            else:
                s = s_func(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_global_pool():
    """test cases of global_pool"""
    verify_global_pool((1, 1024, 7, 7), 'avg')
    verify_global_pool((4, 1024, 7, 7), 'avg')
    verify_global_pool((1, 1024, 7, 7), 'max')
    verify_global_pool((4, 1024, 7, 7), 'max')
    verify_global_pool((1, 7, 7, 1024), 'avg', 'NHWC')
    verify_global_pool((4, 7, 7, 1024), 'avg', 'NHWC')
    verify_global_pool((1, 7, 7, 1024), 'max', 'NHWC')
    verify_global_pool((4, 7, 7, 1024), 'max', 'NHWC')


def verify_adaptive_pool(dshape, out_size, pool_type, layout="NCHW", dtype="float32"):
    """verify function of adaptive_pool"""
    np_data = np.random.uniform(low=0, high=255, size=dshape).astype(dtype)
    np_out = topi.testing.adaptive_pool(np_data, out_size, pool_type, layout)
    oshape = np_out.shape

    data = te.placeholder(dshape, name="data", dtype=dtype)
    if len(out_size) == 2:
        out = topi.nn.adaptive_pool(data, out_size, pool_type, layout)
    else:
        assert len(out_size) == 3
        out = topi.nn.adaptive_pool3d(data, out_size, pool_type, layout)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _adaptive_pool_schedule)
            if device == "cuda":
                s = s_func(out, layout)
            else:
                s = s_func(out)
        a = tvm.nd.array(np_data, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(oshape), dtype=out.dtype), ctx)
        f = tvm.build(s, [data, out], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), np_out, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def test_adaptive_pool():
    """test cases of adaptive_pool"""
    verify_adaptive_pool((1, 3, 224, 224), (1, 1), "max")
    verify_adaptive_pool((1, 3, 224, 224), (1, 1), "avg")
    verify_adaptive_pool((1, 14, 56, 78), (34, 13), "max")
    verify_adaptive_pool((1, 5, 46, 97), (4, 96), "avg")
    verify_adaptive_pool((1, 224, 224, 3), (1, 1), "max", layout="NHWC")
    verify_adaptive_pool((1, 5, 46, 97), (4, 96), "avg", layout="NHWC")
    verify_adaptive_pool((1, 16, 32, 32, 32), (1, 1, 1), "max", layout="NCDHW")
    verify_adaptive_pool((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NCDHW")
    verify_adaptive_pool((1, 16, 32, 32, 32), (2, 2, 2), "avg", layout="NCDHW")
    verify_adaptive_pool((1, 16, 64, 32, 32), (7, 8, 9), "avg", layout="NCDHW")
    verify_adaptive_pool((1, 16, 64, 32, 32), (8, 16, 16), "avg", layout="NCDHW")
    verify_adaptive_pool((1, 16, 32, 32, 32), (1, 1, 1), "avg", layout="NDHWC")
    verify_adaptive_pool((1, 16, 32, 32, 32), (2, 2, 2), "max", layout="NDHWC")
    verify_adaptive_pool((1, 16, 32, 32, 32), (2, 4, 4), "max", layout="NDHWC")


def verify_pool3d(n, ic, ih, kh, sh, padding, pool_type,
                  ceil_mode, count_include_pad=True, layout='NCDHW'):
    """verify function of pool3d"""
    id = iw = ih
    kd = kw = kh
    sd = sw = sh
    input_shape = (n, ic, id, ih, iw)
    kernel = [kd, kh, kw]
    stride = [sd, sh, sw]
    A = te.placeholder(input_shape, name='A')
    B = topi.nn.pool3d(A, kernel=kernel, stride=stride, padding=padding,
                       pool_type=pool_type, ceil_mode=ceil_mode,
                       layout=layout, count_include_pad=count_include_pad)
    B = topi.nn.relu(B)
    dtype = A.dtype
    output_shape = [int(i) for i in B.shape]

    input_np = np.random.uniform(low=0.001, size=input_shape).astype(dtype)
    ref_np = topi.testing.pool3d_ncdhw_python(input_np, kernel, stride, padding,
                                              output_shape, pool_type, count_include_pad, ceil_mode)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _pool_schedule)
            s = s_func(B, layout)

        a = tvm.nd.array(input_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), ref_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def test_pool3d():
    """test cases of pool3d"""
    verify_pool3d(1, 256, 32, 2, 2, [0, 0, 0, 0, 0, 0], 'avg', False, True)
    verify_pool3d(1, 256, 31, 3, 3, [1, 1, 2, 2, 2, 1], 'avg', False, True)
    verify_pool3d(1, 256, 32, 2, 2, [1, 1, 2, 2, 2, 1], 'avg', False, False)
    verify_pool3d(1, 256, 31, 4, 4, [3, 3, 3, 3, 3, 3], 'avg', False, False)
    verify_pool3d(1, 256, 31, 4, 4, [0, 0, 0, 0, 0, 0], 'avg', False, False)
    verify_pool3d(1, 256, 32, 2, 2, [0, 0, 0, 0, 0, 0], 'max', False)
    verify_pool3d(1, 256, 31, 3, 3, [2, 2, 1, 1, 1, 2], 'max', False)
    verify_pool3d(1, 256, 31, 3, 3, [2, 2, 1, 1, 1, 2], 'max', True)

    verify_pool3d(1, 256, 31, 3, 3, [2, 1, 0, 5, 4, 3], 'avg', False, True)
    verify_pool3d(1, 256, 32, 2, 2, [0, 5, 4, 3, 2, 1], 'avg', False, False)
    verify_pool3d(1, 256, 31, 3, 3, [1, 0, 5, 4, 3, 2], 'max', False)
    verify_pool3d(1, 256, 31, 3, 3, [3, 2, 1, 0, 5, 4], 'max', True)


def verify_pool1d(n, ic, iw, kw, sw, padding, pool_type,
                  ceil_mode, count_include_pad=True, layout='NCW'):
    """verify function of pool1d"""
    input_shape = (n, ic, iw)
    kernel = [kw]
    stride = [sw]
    A = te.placeholder(input_shape, name='A')
    B = topi.nn.pool1d(A, kernel=kernel, stride=stride, padding=padding,
                       pool_type=pool_type, ceil_mode=ceil_mode,
                       layout=layout, count_include_pad=count_include_pad)
    B = topi.nn.relu(B)
    dtype = A.dtype
    output_shape = [int(i) for i in B.shape]

    input_np = np.random.uniform(low=0.001, size=input_shape).astype(dtype)
    ref_np = topi.testing.pool1d_ncw_python(input_np, kernel, stride, padding,
                                            output_shape, pool_type, count_include_pad, ceil_mode)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s_func = topi.testing.dispatch(device, _pool_schedule)
            s = s_func(B, layout)

        a = tvm.nd.array(input_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), ref_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)


def test_pool1d():
    """test cases of pool1d"""
    verify_pool1d(1, 256, 32, 2, 2, [0, 0], 'avg', False, True)
    verify_pool1d(1, 256, 31, 3, 3, [1, 2], 'avg', False, True)
    verify_pool1d(1, 256, 32, 2, 2, [1, 2], 'avg', False, False)
    verify_pool1d(1, 256, 31, 4, 4, [3, 3], 'avg', False, False)
    verify_pool1d(1, 256, 31, 4, 4, [0, 0], 'avg', False, False)
    verify_pool1d(1, 256, 32, 2, 2, [0, 0], 'max', False)
    verify_pool1d(1, 256, 31, 3, 3, [2, 1], 'max', False)
    verify_pool1d(1, 256, 31, 3, 3, [2, 1], 'max', True)

    verify_pool1d(1, 256, 31, 3, 3, [2, 5], 'avg', False, True)
    verify_pool1d(1, 256, 32, 2, 2, [0, 3], 'avg', False, False)
    verify_pool1d(1, 256, 31, 3, 3, [1, 4], 'max', False)
    verify_pool1d(1, 256, 31, 3, 3, [3, 0], 'max', True)


if __name__ == "__main__":
    test_pool()
    test_pool1d()
    test_pool3d()
    test_pool_grad()
    test_global_pool()
    test_adaptive_pool()
