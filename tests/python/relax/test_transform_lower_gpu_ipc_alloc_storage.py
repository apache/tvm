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


def test_alloc_storage():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            storage: R.Object = R.memory.alloc_storage(
                R.shape([m, n]), R.prim_value(0), R.str("ipc_memory"), R.dtype("float16")
            )
            alloc: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                storage, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return alloc

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            storage: R.Object = R.call_packed(
                "runtime.disco.cuda_ipc.alloc_storage",
                R.shape([m, n]),
                R.dtype("float16"),
                sinfo_args=(R.Object,),
            )
            alloc: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                storage, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return alloc

    mod = relax.transform.LowerGPUIPCAllocStorage()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_builtin_alloc_tensor():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            tensor: R.Object = R.builtin.alloc_tensor(
                R.shape([m, n]), R.dtype("float16"), R.prim_value(0), R.str("ipc_memory")
            )
            return tensor

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(shape: R.Shape(["m", "n"])):  # type: ignore
            m = T.int64()
            n = T.int64()
            gv: R.Object = R.call_packed(
                "runtime.disco.cuda_ipc.alloc_storage",
                R.shape([m, n]),
                R.dtype("float16"),
                sinfo_args=(R.Object,),
            )
            tensor: R.Tensor((m, n), dtype="float16") = R.memory.alloc_tensor(  # type: ignore
                gv, R.prim_value(0), R.shape([m, n]), R.dtype("float16")
            )
            return tensor

    mod = relax.transform.LowerGPUIPCAllocStorage()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_alloc_storage()
    test_builtin_alloc_tensor()
