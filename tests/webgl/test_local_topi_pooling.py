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
"""Test code for pooling
Copied from topi/tests/python/test_topi_pooling.py.
Should be removed once we fix OpenGL testing on Jenkins.
"""
import numpy as np
import tvm
import topi
import math
from topi.util import get_const_tuple

def verify_pool(n, ic, ih, kh, sh, padding, pool_type, ceil_mode):
    iw = ih
    kw = kh
    sw = sh
    ph, pw = padding
    A = tvm.placeholder((n, ic, ih, iw), name='A')
    B = topi.nn.pool(A, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                     pool_type=pool_type, ceil_mode=ceil_mode)
    B = topi.nn.relu(B)
    dtype = A.dtype

    bshape = get_const_tuple(B.shape)
    ashape = get_const_tuple(A.shape)
    if ceil_mode:
        assert bshape[2] == int(math.ceil(float(ashape[2] - kh + ph * 2) / sh) + 1)
        assert bshape[3] == int(math.ceil(float(ashape[3] - kw + pw * 2) / sw) + 1)
    else:
        assert bshape[2] == int(math.floor(float(ashape[2] - kh + ph * 2) / sh) + 1)
        assert bshape[3] == int(math.floor(float(ashape[3] - kw + pw * 2) / sw) + 1)


    a_np = np.random.uniform(size=(n, ic, ih, iw)).astype(dtype)
    pad_np = np.zeros(shape=(n, ic, ih+2*ph, iw+2*pw)).astype(dtype)
    no_zero = (range(n), range(ic), (range(ph, ih+ph)), (range(pw, iw+pw)))
    pad_np[np.ix_(*no_zero)] = a_np
    _, oc, oh, ow = get_const_tuple(B.shape)
    b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)

    if pool_type == 'avg':
        for i in range(oh):
            for j in range(ow):
                b_np[:,:,i,j] = np.mean(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3))
    elif pool_type =='max':
        for i in range(oh):
            for j in range(ow):
                b_np[:,:,i,j] = np.max(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3))
    b_np = np.maximum(b_np, 0.0)

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_pool(B)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=dtype), ctx)
        print(tvm.lower(s, [A, B], simple_mode=True))

        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['opengl']:
        check_device(device)

def test_pool():
    verify_pool(1, 256, 32, 2, 2, [0, 0], 'avg', False)
    verify_pool(1, 256, 31, 3, 3, [1, 2], 'avg', False)
    verify_pool(1, 256, 32, 2, 2, [0, 0], 'max', False)
    verify_pool(1, 256, 31, 3, 3, [2, 1], 'max', False)
    verify_pool(1, 256, 31, 3, 3, [2, 1], 'max', True)



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
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_global_pool(B)
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['opengl']:
        check_device(device)

def test_global_pool():
    verify_global_pool(1, 1024, 7, 7, 'avg')
    verify_global_pool(4, 1024, 7, 7, 'avg')
    verify_global_pool(1, 1024, 7, 7, 'max')
    verify_global_pool(4, 1024, 7, 7, 'max')


if __name__ == "__main__":
    test_pool()
    test_global_pool()
