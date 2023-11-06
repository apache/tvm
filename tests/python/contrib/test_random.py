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
"""Configure pytest"""
# pylint: disable=invalid-name
import threading
import numpy as np
import tvm
from tvm import te
from tvm.contrib import random
from tvm import rpc
import tvm.testing


def test_randint():
    """Tests randint function"""
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
        dev = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na)) < 0.3
        assert np.min(na) == -127
        assert np.max(na) == 127

    verify()


def test_uniform():
    """Tests uniform function"""
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
        dev = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 0.5) < 1e-1
        assert abs(np.min(na) - 0.0) < 1e-3
        assert abs(np.max(na) - 1.0) < 1e-3

    verify()


def test_normal():
    """Tests normal function"""
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
        dev = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 3) < 1e-1
        assert abs(np.std(na) - 4) < 1e-2

    verify()


@tvm.testing.uses_gpu
def test_random_fill():
    """Tests random_fill function"""

    def test_local(dev, dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        value = tvm.nd.empty((512, 512), dtype, dev)
        random_fill = tvm.get_global_func("tvm.contrib.random.random_fill")
        random_fill(value)

        assert np.count_nonzero(value.numpy()) == 512 * 512

        # make sure arithmentic doesn't overflow too
        np_values = value.numpy()
        assert np.isfinite(np_values * np_values + np_values).any()

    def test_rpc(dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        if not tvm.testing.device_enabled("rpc") or not tvm.runtime.enabled("llvm"):
            return

        def check_remote(server):
            remote = rpc.connect(server.host, server.port)
            value = tvm.nd.empty((512, 512), dtype, remote.cpu())
            random_fill = remote.get_function("tvm.contrib.random.random_fill")
            random_fill(value)

            assert np.count_nonzero(value.numpy()) == 512 * 512

            # make sure arithmentic doesn't overflow too
            np_values = value.numpy()
            assert np.isfinite(np_values * np_values + np_values).any()

        check_remote(rpc.Server("127.0.0.1"))

    for dtype in [
        "bool",
        "int4",
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
        for _, dev in tvm.testing.enabled_targets():
            test_local(dev, dtype)
        test_rpc(dtype)


def test_random_fill_mt():
    """Check random filler applicability in case of nontrivial thread pool configuration.
    Particularly when MaxConcurrency != num_workers_used_ which is actual for big-little systems.
    """
    no_exception_happened = True

    def test_body():
        try:
            num_thread_used = 1
            configure_threads = tvm.get_global_func("runtime.config_threadpool")
            configure_threads(1, num_thread_used)

            test_input = tvm.runtime.ndarray.empty((10, 10))
            random_fill = tvm.get_global_func("tvm.contrib.random.random_fill_for_measure")
            random_fill(test_input)
        except:  # pylint: disable=bare-except
            nonlocal no_exception_happened
            no_exception_happened = False

    # ThreadPool object is thread local. To eliminate effect on other test cases put it into thread
    x = threading.Thread(target=test_body)
    x.start()
    x.join()
    assert no_exception_happened


if __name__ == "__main__":
    test_randint()
    test_uniform()
    test_normal()
    test_random_fill()
    test_random_fill_mt()
