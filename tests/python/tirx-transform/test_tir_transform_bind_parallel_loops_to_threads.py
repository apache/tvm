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
"""Tests for tirx.transform.BindParallelLoopsToThreads."""

import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T


def test_bind_parallel_skips_without_target():
    """PrimFuncs without tvm::attr::kTarget must be left unchanged (no Target::Current guess)."""

    @I.ir_module
    class Mod:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32")):
            for i in T.parallel(4):
                A[i] = T.float32(1)

    after = tvm.tirx.transform.BindParallelLoopsToThreads()(Mod)
    tvm.ir.assert_structural_equal(after, Mod)


def test_bind_parallel_skips_non_gpu_target():
    @I.ir_module
    class Mod:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32")):
            T.func_attr({"target": T.target("llvm")})
            for i in T.parallel(4):
                A[i] = T.float32(1)

    after = tvm.tirx.transform.BindParallelLoopsToThreads()(Mod)
    tvm.ir.assert_structural_equal(after, Mod)


def test_bind_parallel_cuda_wraps_parallel_in_thread_extents():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((4,), "float32")):
            T.func_attr({"target": T.target("cuda")})
            for i in T.parallel(4):
                A[i] = T.float32(1)

    after = tvm.tirx.transform.BindParallelLoopsToThreads()(Before)
    body = after["main"].body
    assert isinstance(body, tvm.tirx.AttrStmt)
    assert body.node.thread_tag == "blockIdx.x"
    inner = body.body
    assert isinstance(inner, tvm.tirx.AttrStmt)
    assert inner.node.thread_tag == "threadIdx.x"
    assert isinstance(inner.body, tvm.tirx.IfThenElse)
    assert inner.body.else_case is None


def test_bind_parallel_nested_parallel_raises():
    @I.ir_module
    class Mod:
        @T.prim_func
        def main(A: T.Buffer((4, 4), "float32")):
            T.func_attr({"target": T.target("cuda")})
            for i in T.parallel(4):
                for j in T.parallel(4):
                    A[i, j] = T.float32(1)

    with pytest.raises(tvm.error.InternalError, match="nested parallel"):
        tvm.tirx.transform.BindParallelLoopsToThreads()(Mod)


def test_bind_parallel_respects_max_num_threads():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer((256,), "float32")):
            T.func_attr({"target": T.target({"kind": "cuda", "max_num_threads": 128})})
            for i in T.parallel(256):
                A[i] = T.float32(1)

    after = tvm.tirx.transform.BindParallelLoopsToThreads()(Before)
    inner = after["main"].body.body
    assert isinstance(inner, tvm.tirx.AttrStmt)
    assert inner.node.thread_tag == "threadIdx.x"
    assert isinstance(inner.value, tvm.tirx.IntImm)
    assert inner.value.value == 128


if __name__ == "__main__":
    tvm.testing.main()
