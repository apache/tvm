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
from tvm import tir
from tvm.script import tir as T


def test_canonicalize_loop():
    @T.prim_func
    def before(A: T.Buffer[(128,), "float32"], B: T.Buffer[(128,), "float32"]):
        T.func_attr({"global_symbol": "main"})
        for i in range(1, 128, 5):
            B[i] = A[i] + 1.0

    @T.prim_func
    def expected(A: T.Buffer[(128,), "float32"], B: T.Buffer[(128,), "float32"]):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 26):
            B[i * 5 + 1] = A[i * 5 + 1] + 1.0

    mod = tvm.IRModule.from_expr(before)
    mod = tir.transform.CanonicalizeLoop()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_canonicalize_nested_loop():
    @T.prim_func
    def before(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "main"})
        for i in range(1, 128, 5):
            for j in range(2, 128, 3):
                B[i, j] = A[i, j] + 1.0

    @T.prim_func
    def expected(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 26):
            for j in T.serial(0, 42):
                B[i * 5 + 1, j * 3 + 2] = A[i * 5 + 1, j * 3 + 2] + 1.0

    mod = tvm.IRModule.from_expr(before)
    mod = tir.transform.CanonicalizeLoop()(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_canonicalize_negative_step():
    @T.prim_func
    def before(A: T.Buffer[(128,), "float32"], B: T.Buffer[(128,), "float32"]):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 127, step=-3):
            B[i] = A[i] + 1.0

    mod = tvm.IRModule.from_expr(before)
    with pytest.raises(tvm.error.InternalError):
        mod = tir.transform.CanonicalizeLoop()(mod)


def test_canonicalize_dynamic_step():
    """Currently we report error for dynamic step since we could not prove it is positive"""

    @T.prim_func
    def before(A: T.Buffer[(128,), "float32"], B: T.Buffer[(128,), "float32"], step: T.int32):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 128, step=step):
            B[i] = A[i] + 1.0

    mod = tvm.IRModule.from_expr(before)
    with pytest.raises(tvm.error.InternalError):
        mod = tir.transform.CanonicalizeLoop()(mod)


if __name__ == "__main__":
    tvm.testing.main()
