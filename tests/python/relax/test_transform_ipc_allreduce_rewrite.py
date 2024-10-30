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
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_ipc_allreduce_rewrite():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            lv1: R.Tensor((m, n), dtype="float16") = alloc  # type: ignore
            alloc1: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            _: R.Object = R.call_packed(
                "runtime.disco.allreduce", lv1, R.shape([0]), R.prim_value(True), alloc1
            )
            return alloc1

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("ipc_memory")
            )
            lv1: R.Tensor((m, n), dtype="float16") = alloc  # type: ignore
            alloc1: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            _: R.Object = R.call_packed(
                "runtime.disco.cuda_ipc.custom_allreduce", lv1, R.prim_value(1), alloc1
            )
            return alloc1

    allreduce_strategy = 1
    mod = relax.transform.IPCAllReduceRewrite(allreduce_strategy)(Module)
    tvm.ir.assert_structural_equal(
        mod,
        (
            Expected
            if tvm.get_global_func("runtime.disco.cuda_ipc.custom_allreduce", allow_missing=True)
            is not None
            else Module
        ),
    )


def test_ipc_allreduce_spread_along_reshape():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            lv1: R.Tensor((m * n,), dtype="float16") = R.reshape(alloc, (m * n,))  # type: ignore
            alloc1: R.Tensor((m * n,), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m * n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            _: R.Object = R.call_packed(
                "runtime.disco.allreduce", lv1, R.shape([0]), R.prim_value(False), alloc1
            )
            return alloc1

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(
            shape: R.Shape(["m", "n"]),  # type: ignore
        ) -> R.Tensor(("m * n",), dtype="float16"):  # type: ignore
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("ipc_memory")
            )
            lv1: R.Tensor((m * n,), dtype="float16") = R.reshape(  # type: ignore
                alloc, R.shape([m * n])
            )
            alloc1: R.Tensor((m * n,), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m * n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            _: R.Object = R.call_packed(
                "runtime.disco.cuda_ipc.custom_allreduce", lv1, R.prim_value(1), alloc1
            )
            return alloc1

    allreduce_strategy = 1
    mod = relax.transform.IPCAllReduceRewrite(allreduce_strategy)(Module)
    tvm.ir.assert_structural_equal(
        mod,
        (
            Expected
            if tvm.get_global_func("runtime.disco.cuda_ipc.custom_allreduce", allow_missing=True)
            is not None
            else Module
        ),
    )


def test_ipc_allreduce_skip_reducer_other_than_sum():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            lv1: R.Tensor((m, n), dtype="float16") = alloc  # type: ignore
            alloc1: R.Tensor((m, n), dtype="float16") = R.builtin.alloc_tensor(  # type: ignore
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("global")
            )
            _: R.Object = R.call_packed(
                "runtime.disco.allreduce", lv1, R.shape([1]), R.prim_value(True), alloc1
            )
            return alloc1

    allreduce_strategy = 1
    mod = relax.transform.IPCAllReduceRewrite(allreduce_strategy)(Module)
    tvm.ir.assert_structural_equal(mod, Module)


if __name__ == "__main__":
    tvm.testing.main()
