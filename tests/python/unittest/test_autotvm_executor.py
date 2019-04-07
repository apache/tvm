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
"""Test local executor"""
import time

from tvm.autotvm.measure import LocalExecutor, executor

def slow(n):
    r = 0
    for i in range(0, n+1):
        r += i
    return r

def fast(n):
    return n*(n+1)//2

def test_local_measure_async():
    ex = LocalExecutor()
    f1 = ex.submit(slow, 9999999)
    f2 = ex.submit(fast, 9999999)
    t1 = 0
    t2 = 0
    while True:
        if t1 == 0 and f1.done():
            t1 = time.time()
        if t2 == 0 and f2.done():
            t2 = time.time()
        if t1 != 0 and t2 != 0:
            break
    assert t2 < t1, "Expected fast async job to finish first!"
    assert f1.get() == f2.get()

def timeout_job(n):
    time.sleep(n * 1.5)

def test_timeout():
    timeout = 0.5

    ex = LocalExecutor(timeout=timeout)

    f1 = ex.submit(timeout_job, timeout)
    while not f1.done():
        pass
    res = f1.get()
    assert isinstance(res, executor.TimeoutError)

if __name__ == "__main__":
    test_local_measure_async()
    test_timeout()
