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
"""Test code for softmax"""
import os
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
import logging
from tvm.topi.util import get_const_tuple


_softmax_schedule = {
    "generic": topi.generic.schedule_softmax,
    "cpu": topi.x86.schedule_softmax,
    "gpu": topi.cuda.schedule_softmax,
    "hls": topi.hls.schedule_softmax,
}


def check_device(A, B, a_np, b_np, device, ctx, name):
    print("Running on target: %s" % device)
    with tvm.target.Target(device):
        s_func = tvm.topi.testing.dispatch(device, _softmax_schedule)
        s = s_func(B)

    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
    f = tvm.build(s, [A, B], device, name=name)
    f(a, b)
    tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)


def verify_softmax(m, n, dtype="float32"):
    A = te.placeholder((m, n), dtype=dtype, name="A")
    B = topi.nn.softmax(A)
    # confirm lower works
    s = te.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)

    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = tvm.topi.testing.softmax_python(a_np)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(A, B, a_np, b_np, device, ctx, "softmax")


def verify_softmax_4d(shape, dtype="float32"):
    A = te.placeholder(shape, dtype=dtype, name="A")
    B = topi.nn.softmax(A, axis=1)

    _, c, h, w = shape
    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = tvm.topi.testing.softmax_python(a_np.transpose(0, 2, 3, 1).reshape(h * w, c))
    b_np = b_np.reshape(1, h, w, c).transpose(0, 3, 1, 2)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(A, B, a_np, b_np, device, ctx, "softmax")


@tvm.testing.uses_gpu
def test_softmax():
    verify_softmax(32, 10)
    verify_softmax(3, 4)
    verify_softmax(32, 10, "float64")
    verify_softmax_4d((1, 16, 256, 256))


def verify_log_softmax(m, n, dtype="float32"):
    A = te.placeholder((m, n), dtype=dtype, name="A")
    B = topi.nn.log_softmax(A)
    # confirm lower works
    s = te.create_schedule([B.op])
    tvm.lower(s, [A, B], simple_mode=True)
    a_np = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b_np = tvm.topi.testing.log_softmax_python(a_np)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(A, B, a_np, b_np, device, ctx, "log_softmax")


@tvm.testing.uses_gpu
def test_log_softmax():
    verify_log_softmax(32, 10)
    verify_log_softmax(3, 4)
    verify_log_softmax(32, 10, "float64")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_softmax()
    test_log_softmax()
