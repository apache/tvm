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
# ruff: noqa: F401
"""Test PopenPoolExecutor."""

import os
import time

import psutil
import pytest

from tvm.contrib.popen_pool import PopenPoolExecutor, PopenWorker
from tvm.testing import (
    identity_after,
    terminate_self,
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
    import tvm_ffi

    import tvm

    pool = PopenPoolExecutor(max_workers=2, timeout=0.01)
    value1 = pool.submit(identity_after, 1, 100)
    value2 = pool.submit(terminate_self)
    value3 = pool.submit(identity_after, 3, 0)
    value4 = pool.submit(tvm_ffi.core.String, "xyz")

    with pytest.raises(TimeoutError):
        value1.result()

    with pytest.raises(ChildProcessError):
        value2.result()

    assert value3.result() == 3
    value = value4.result()
    assert value == "xyz"

    pool = PopenPoolExecutor(max_workers=4, timeout=None)
    values = pool.map_with_error_catching(lambda x: x, range(100))

    for idx, val in enumerate(values):
        assert val.value == idx


def test_popen_pool_executor_recycles():
    pool = PopenPoolExecutor(max_workers=1, timeout=None, maximum_process_uses=2)

    initial_pid = pool.submit(os.getpid).result()
    assert initial_pid == pool.submit(os.getpid).result()
    assert initial_pid != pool.submit(os.getpid).result()


if __name__ == "__main__":
    test_popen_worker()
    test_popen_worker_recycles()
    test_popen_pool_executor()
    test_popen_pool_executor_recycles()
