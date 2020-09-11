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
from tvm import te
import numpy as np
import pytest
from tvm.contrib import random
from tvm import rpc
import tvm.testing


def test_randint():
    m = 10240
    n = 10240
    A = random.randint(-127, 128, size=(m, n), dtype="int32")
    s = te.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.randint", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na)) < 0.3
        assert np.min(na) == -127
        assert np.max(na) == 127

    verify()


def test_uniform():
    m = 10240
    n = 10240
    A = random.uniform(0, 1, size=(m, n))
    s = te.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.uniform", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na) - 0.5) < 1e-1
        assert abs(np.min(na) - 0.0) < 1e-3
        assert abs(np.max(na) - 1.0) < 1e-3

    verify()


@pytest.mark.skip('neo-ai/tvm: skip due to MemoryError')
def test_normal():
    m = 10240
    n = 10240
    A = random.normal(3, 4, size=(m, n))
    s = te.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.normal", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na) - 3) < 1e-1
        assert abs(np.std(na) - 4) < 1e-2

    verify()


@tvm.testing.uses_gpu
def test_random_fill():
    def test_local(ctx, dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        np_ones = np.ones((512, 512), dtype=dtype)
        value = tvm.nd.empty(np_ones.shape, np_ones.dtype, ctx)
        random_fill = tvm.get_global_func("tvm.contrib.random.random_fill")
        random_fill(value)

        assert np.count_nonzero(value.asnumpy()) == 512 * 512

        # make sure arithmentic doesn't overflow too
        np_values = value.asnumpy()
        assert np.isfinite(np_values * np_values + np_values).any()

    def test_rpc(dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        if not tvm.testing.device_enabled("rpc") or not tvm.runtime.enabled("llvm"):
            return
        np_ones = np.ones((512, 512), dtype=dtype)
        server = rpc.Server("localhost")
        remote = rpc.connect(server.host, server.port)
        value = tvm.nd.empty(np_ones.shape, np_ones.dtype, remote.cpu())
        random_fill = remote.get_function("tvm.contrib.random.random_fill")
        random_fill(value)

        assert np.count_nonzero(value.asnumpy()) == 512 * 512

        # make sure arithmentic doesn't overflow too
        np_values = value.asnumpy()
        assert np.isfinite(np_values * np_values + np_values).any()

    for dtype in [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "int32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]:
        for _, ctx in tvm.testing.enabled_targets():
            test_local(ctx, dtype)
        test_rpc(dtype)


if __name__ == "__main__":
    test_randint()
    test_uniform()
    # TODO(trevmorr): Disabled in neo-ai/tvm due to MemoryError
    # test_normal()
    test_random_fill()
