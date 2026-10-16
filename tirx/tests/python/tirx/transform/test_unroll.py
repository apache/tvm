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
from __future__ import annotations

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("max_step,explicit", [(16, True), (15, True), (16, False)])
def test_unroll_loop(max_step, explicit):
    @I.ir_module
    class Before:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    @I.ir_module
    class Expected:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            if T.constexpr(not explicit):
                for i in T.unroll(n, n + 2):
                    for j in T.unroll(8):
                        Ab[j + 1] = Ab[i] + T.int64(1)
            elif T.constexpr(max_step == 15):
                for i in T.serial(n, n + 2):
                    Ab[T.Add(0, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(1, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(2, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(3, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(4, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(5, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(6, 1)] = Ab[i] + T.int64(1)
                    Ab[T.Add(7, 1)] = Ab[i] + T.int64(1)
            else:
                Ab[T.Add(0, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(1, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(2, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(3, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(4, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(5, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(6, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(7, 1)] = Ab[n] + T.int64(1)
                Ab[T.Add(0, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(1, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(2, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(3, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(4, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(5, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(6, 1)] = Ab[n + T.int64(1)] + T.int64(1)
                Ab[T.Add(7, 1)] = Ab[n + T.int64(1)] + T.int64(1)

    with tvm_tirx.ir.transform.PassContext(
        config={"tirx.UnrollLoop": {"auto_max_step": max_step, "explicit_unroll": explicit}}
    ):
        After = tvm_tirx.tirx.transform.UnrollLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_unroll_loop_annotation():
    @I.ir_module
    class Before:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            for i in T.serial(n, n + 2, annotations={"auto_unroll_max_step": 16}):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    @I.ir_module
    class Expected:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            for i in T.unroll(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    with tvm_tirx.ir.transform.PassContext(
        config={"tirx.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        After = tvm_tirx.tirx.transform.UnrollLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_unroll_fake_loop():
    @I.ir_module
    class Before:
        @T.function
        def main(Ab: T.Tensor((n,), "int32"), n: T.int64):  # noqa: F821
            for i in T.serial(1):
                Ab[i * 2] = 3
                for j in T.serial(10):
                    Ab[j + 1] = Ab[i] + 1

    @I.ir_module
    class Expected:
        @T.function
        def main(Ab: T.Tensor((n,), "int32"), n: T.int64):  # noqa: F821
            Ab[T.Mul(0, 2)] = 3
            for j in T.serial(10):
                Ab[j + 1] = Ab[0] + 1

    with tvm_tirx.ir.transform.PassContext(
        config={
            "tirx.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        After = tvm_tirx.tirx.transform.UnrollLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_unroll_allocations():
    @I.ir_module
    class Before:
        @T.function
        def main():
            for i in T.unroll(2):
                buf = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
                buf[0] = 0.0

    @I.ir_module
    class Expected:
        @T.function
        def main():
            buf1 = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
            buf1[0] = 0.0
            buf2 = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
            buf2[0] = 0.0

    after = tvm_tirx.tirx.transform.UnrollLoop()(Before)

    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_unroll_local_access():
    @I.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((64,), "float32")):
            with T.launch_thread("blockIdx.x", 4) as bx:
                with T.launch_thread("threadIdx.x", 4) as tx:
                    A_local = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="local")])
                    for i in T.serial(4):
                        A_local[i] = T.float32(i)

    @I.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((64,), "float32")):
            with T.launch_thread("blockIdx.x", 4) as bx:
                with T.launch_thread("threadIdx.x", 4) as tx:
                    A_local = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="local")])
                    A_local[0] = T.float32(0)
                    A_local[1] = T.float32(1)
                    A_local[2] = T.float32(2)
                    A_local[3] = T.float32(3)

    with tvm_tirx.ir.transform.PassContext(
        config={
            "tirx.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = tvm_tirx.tirx.transform.UnrollLoop()(Before)
        after = tvm_tirx.tirx.transform.StmtSimplify()(after)

    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_pass_config():
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((1,), "int32")):
            A[0] = 1

    cfg = tvm_tirx.ir.transform.PassContext(
        opt_level=1,
        config={
            "tirx.UnrollLoop": {
                "auto_max_step": 10,
            }
        },
    )
    assert cfg.opt_level == 1

    assert cfg.config["tirx.UnrollLoop"].auto_max_step == 10
    # default option
    assert cfg.config["tirx.UnrollLoop"].explicit_unroll

    with cfg:
        After = tvm_tirx.tirx.transform.UnrollLoop()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)

    # schema checking for specific config key
    with pytest.raises(TypeError):
        cfg = tvm_tirx.ir.transform.PassContext(config={"tirx.UnrollLoop": {"invalid": 1}})

    # schema check for un-registered config
    with pytest.raises(AttributeError):
        cfg = tvm_tirx.ir.transform.PassContext(config={"inavlid-opt": True})

    # schema check for wrong type
    with pytest.raises(AttributeError):
        cfg = tvm_tirx.ir.transform.PassContext(config={"tirx.UnrollLoop": 1})


if __name__ == "__main__":
    tvm_tirx.testing.main()
