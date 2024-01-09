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
import pytest

import tvm
from tvm.script import tir as T


@T.prim_func
def bad_load(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
    B[0, 0] = A[2, 2]


@T.prim_func
def bad_load_loop(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
    for i in range(3):
        B[i, 0] = A[i, 2]


@T.prim_func
def bad_store(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
    B[0, 3] = A[1, 2]


@T.prim_func
def bad_store_loop(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
    for i in range(3):
        B[0, i] = A[1, i]


@T.prim_func
def unknown_bounds(A: T.Buffer((2, 3), "float32"), B: T.Buffer((3, 2), "float32")):
    N = T.int32()
    for i in range(3):
        B[0, N] = A[1, i]


def test_oob_load():
    with pytest.raises(tvm.tir.ScheduleError) as err:
        tvm.tir.analysis.OOBChecker()(tvm.IRModule.from_expr(bad_load))
    assert "buffer A" in err.value.args[0]

    with pytest.raises(tvm.tir.ScheduleError) as err:
        tvm.tir.analysis.OOBChecker()(tvm.IRModule.from_expr(bad_load_loop))
    assert "buffer A" in err.value.args[0]


def test_oob_store():
    with pytest.raises(tvm.tir.ScheduleError) as err:
        tvm.tir.analysis.OOBChecker()(tvm.IRModule.from_expr(bad_store))
    assert "buffer B" in err.value.args[0]

    with pytest.raises(tvm.tir.ScheduleError) as err:
        tvm.tir.analysis.OOBChecker()(tvm.IRModule.from_expr(bad_store_loop))
    assert "buffer B" in err.value.args[0]


def test_unknown_bounds():
    # This should not return an error as we can't probe that N goes out of bounds
    tvm.tir.analysis.OOBChecker()(tvm.IRModule.from_expr(unknown_bounds))


if __name__ == "__main__":
    tvm.testing.main()
