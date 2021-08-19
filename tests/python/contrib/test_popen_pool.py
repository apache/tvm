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
import time
from tvm.contrib.popen_pool import PopenWorker, PopenPoolExecutor
from tvm.testing import identity_after, terminate_self, initializer, after_initializer


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
    pool = PopenPoolExecutor(
        max_workers=2, timeout=0.01, initializer=initializer, initargs=initargs
    )
    proc = PopenWorker(initializer=initializer, initargs=initargs)
    proc.send(after_initializer)
    a, b, c = proc.recv()
    assert a == 1
    assert b == 2
    assert c == 3


if __name__ == "__main__":
    test_popen_worker()
    test_popen_pool_executor()
