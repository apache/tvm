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
import time
import ctypes

import tvm
from tvm import te
from tvm.contrib.utils import tempdir
from tvm.runtime.module import BenchmarkResult


def test_min_repeat_ms():
    tmp = tempdir()
    filename = tmp.relpath("log")

    @tvm.register_func
    def my_debug(filename):
        """one call lasts for 100 ms and writes one character to a file"""
        time.sleep(0.1)
        with open(filename, "a") as fout:
            fout.write("c")

    X = te.compute((), lambda: tvm.tir.call_packed("my_debug", filename))
    s = te.create_schedule(X.op)
    func = tvm.build(s, [X])

    x = tvm.nd.empty((), dtype="int32")
    ftimer = func.time_evaluator(func.entry_name, tvm.cpu(), number=1, repeat=1)
    ftimer(x)

    with open(filename, "r") as fin:
        ct = len(fin.readline())

    assert ct == 2

    ftimer = func.time_evaluator(func.entry_name, tvm.cpu(), number=1, repeat=1, min_repeat_ms=1000)
    ftimer(x)

    # make sure we get more than 10 calls
    with open(filename, "r") as fin:
        ct = len(fin.readline())

    assert ct > 10 + 2


def test_benchmark_result():
    r = BenchmarkResult([1, 2, 2, 5])
    assert r.mean == 2.5
    assert r.median == 2.0
    assert r.min == 1
    assert r.max == 5
    assert r.std == 1.5


if __name__ == "__main__":
    test_min_repeat_ms()
    test_benchmark_result()
