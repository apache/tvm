"""Test code for pooling"""
import numpy as np
import tvm
import topi
import math
from topi.util import get_const_tuple

from common import get_all_backend

def verify_pool(n, ic, ih, kh, sh, padding, pool_type, ceil_mode, count_include_pad=True):
    iw = ih
    kw = kh
    sw = sh
    pt, pl, pb, pr = padding
    layout = "NCHW"
    A = tvm.placeholder((n, ic, ih, iw), name='A')
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
                    b_np[:,:,i,j] = np.mean(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3))
                else:
                    pad_count = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2,3))
                    b_np[:,:,i,j] = np.sum(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3)) / np.maximum(pad_count, 1)

    elif pool_type =='max':
        for i in range(oh):
            for j in range(ow):
                b_np[:,:,i,j] = np.max(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3))
    b_np = np.maximum(b_np, 0.0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_pool(B, layout)

        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_pool():
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


def verify_global_pool(n, c, h, w, pool_type):
    A = tvm.placeholder((n, c, h, w), name='A')
    B = topi.nn.global_pool(A, pool_type=pool_type)
    B = topi.nn.relu(B)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    if pool_type == 'avg':
        b_np = np.mean(a_np, axis=(2,3), keepdims=True)
    elif pool_type =='max':
        b_np = np.max(a_np, axis=(2,3), keepdims=True)
    b_np = np.maximum(b_np, 0.0)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_global_pool(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_global_pool():
    verify_global_pool(1, 1024, 7, 7, 'avg')
    verify_global_pool(4, 1024, 7, 7, 'avg')
    verify_global_pool(1, 1024, 7, 7, 'max')
    verify_global_pool(4, 1024, 7, 7, 'max')


if __name__ == "__main__":
    test_pool()
    test_global_pool()
