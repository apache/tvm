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
"""Test PopenPoolExecutor."""
import pytest
import os
import psutil
import time
from tvm.contrib.popen_pool import PopenWorker, PopenPoolExecutor
from tvm.testing import (
    identity_after,
    terminate_self,
    initializer,
    after_initializer,
    register_ffi,
    call_py_ffi,
    call_cpp_ffi,
    call_cpp_py_ffi,
    fast_summation,
    slow_summation,
    timeout_job,
)


def test_popen_worker():
    proc = PopenWorker()

    with pytest.raises(TimeoutError):
        proc.send(identity_after, [1, 100], timeout=0.01)
        proc.recv()

    with pytest.raises(ChildProcessError):
        proc.send(terminate_self)
        proc.recv()

    proc.send(identity_after, [2, 0])
    assert proc.recv() == 2

    proc.send(identity_after, [4, 0.0001])
    assert proc.recv() == 4


def test_popen_worker_reuses():
    proc = PopenWorker(maximum_uses=None)

    proc.send(os.getpid)
    initial_pid = proc.recv()

    proc.send(os.getpid)
    assert proc.recv() == initial_pid


def test_popen_worker_recycles():
    proc = PopenWorker(maximum_uses=2)

    proc.send(os.getpid)
    initial_pid = proc.recv()
    assert psutil.pid_exists(initial_pid)

    proc.send(os.getpid)
    assert proc.recv() == initial_pid
    assert psutil.pid_exists(initial_pid)

    proc.send(os.getpid)
    assert proc.recv() != initial_pid
    assert not psutil.pid_exists(initial_pid)


def test_popen_pool_executor():
    import tvm

    pool = PopenPoolExecutor(max_workers=2, timeout=0.01)
    value1 = pool.submit(identity_after, 1, 100)
    value2 = pool.submit(terminate_self)
    value3 = pool.submit(identity_after, 3, 0)
    value4 = pool.submit(tvm.runtime.String, "xyz")

    with pytest.raises(TimeoutError):
        value1.result()

    with pytest.raises(ChildProcessError):
        value2.result()

    assert value3.result() == 3
    value = value4.result()
    assert isinstance(value, tvm.runtime.String)
    assert value == "xyz"

    pool = PopenPoolExecutor(max_workers=4, timeout=None)
    values = pool.map_with_error_catching(lambda x: x, range(100))

    for idx, val in enumerate(values):
        assert val.value == idx


def test_popen_initializer():
    initargs = [1, 2, 3]
    proc = PopenWorker(initializer=initializer, initargs=initargs)
    proc.send(after_initializer)
    test_global_state_1, test_global_state_2, test_global_state_3 = proc.recv()
    assert test_global_state_1 == initargs[0]
    assert test_global_state_2 == initargs[1]
    assert test_global_state_3 == initargs[2]


def test_popen_worker_recycles_with_initializer():
    initargs = [1, 2, 3]
    proc = PopenWorker(initializer=initializer, initargs=initargs, maximum_uses=3)

    proc.send(os.getpid)
    initial_pid = proc.recv()

    proc.send(after_initializer)
    assert list(proc.recv()) == initargs

    proc.send(os.getpid)
    assert proc.recv() == initial_pid

    # The process should be recycled with this send.
    proc.send(os.getpid)
    assert proc.recv() != initial_pid

    # But the initializer should've run this time as well.
    proc.send(after_initializer)
    assert list(proc.recv()) == initargs


def test_popen_ffi():
    proc = PopenWorker(register_ffi)

    # call python function via ffi
    initargs = [0]
    proc.send(call_py_ffi, initargs)
    assert proc.recv() == initargs[0]

    # call cpp function via ffi
    initargs = [1]
    proc.send(call_cpp_ffi, initargs)
    assert proc.recv() == initargs[0]

    # call python function from cpp function via ffi
    initargs = [2]
    proc.send(call_cpp_py_ffi, initargs)
    assert proc.recv() == initargs[0]


def test_popen_pool_executor_timeout():
    timeout = 0.5

    pool = PopenPoolExecutor(timeout=timeout)

    f1 = pool.submit(timeout_job, timeout)
    while not f1.done():
        pass
    try:
        res = f1.result()
    except Exception as ex:
        assert isinstance(ex, TimeoutError)


def test_popen_pool_executor_recycles():
    pool = PopenPoolExecutor(max_workers=1, timeout=None, maximum_process_uses=2)

    initial_pid = pool.submit(os.getpid).result()
    assert initial_pid == pool.submit(os.getpid).result()
    assert initial_pid != pool.submit(os.getpid).result()


if __name__ == "__main__":
    test_popen_worker()
    test_popen_worker_recycles()
    test_popen_pool_executor()
    test_popen_initializer()
    test_popen_worker_recycles_with_initializer()
    test_popen_ffi()
    test_popen_pool_executor_timeout()
    test_popen_pool_executor_recycles()
