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
import tvm.testing
from tvm import te
from tvm.script import tir as T


def test_decl_buffer():
    """decl_buffer with texture scope."""

    @T.prim_func
    def func():
        A = T.decl_buffer([32, 32, 32, 4], dtype="float32", scope="global.texture")
        A[0, 0, 0, 0] = 0

    lowered = tvm.lower(func)["main"]
    assert isinstance(lowered.body, tvm.tir.LetStmt)
    assert isinstance(lowered.body.value, tvm.tir.expr.Call)
    assert lowered.body.value.op.name == "tir.nd_mem_alloc_with_scope"


def test_alloc_buffer():
    """alloc_buffer with texture scope."""

    @T.prim_func
    def func():
        A = T.alloc_buffer([32, 32, 32, 4], dtype="float32", scope="global.texture-weight")
        A[0, 0, 0, 0] = 0

    lowered = tvm.lower(func)["main"]
    assert isinstance(lowered.body, tvm.tir.LetStmt)
    assert isinstance(lowered.body.value, tvm.tir.expr.Call)
    assert lowered.body.value.op.name == "tir.nd_mem_alloc_with_scope"


def test_alloc_buffer_negative_test():
    """Shouldn't ave texture intrensic for general use."""

    @T.prim_func
    def func():
        A = T.alloc_buffer([32, 32, 32, 4], dtype="float32")
        A[0, 0, 0, 0] = 0

    lowered = tvm.lower(func)["main"]
    assert isinstance(lowered.body, tvm.tir.Allocate)


def test_with_block():
    """Scoped with block."""

    @T.prim_func
    def func(
        A: T.Buffer((T.int64(1), T.int64(16), T.int64(16)), "float16"),
        B: T.Buffer((T.int64(32), T.int64(32)), "float16"),
        C: T.Buffer((T.int64(1), T.int64(8), T.int64(8)), "float16"),
    ):
        with T.block("block"):
            A = T.alloc_buffer([1, 16, 16], dtype="float16")
            B = T.alloc_buffer([32, 32], dtype="float16")
            C = T.alloc_buffer([1, 8, 8], dtype="float16")
            D = T.alloc_buffer([32, 32, 32, 4], dtype="float16", scope="global.texture-weight")
            T.evaluate(D[0, 0, 0, 0])

    lowered = tvm.lower(func)["main"]
    assert isinstance(lowered.body, tvm.tir.LetStmt)
    assert isinstance(lowered.body.value, tvm.tir.expr.Call)
    assert lowered.body.value.op.name == "tir.nd_mem_alloc_with_scope"
    assert lowered.body.var.name == "D"


if __name__ == "__main__":
    tvm.testing.main()
