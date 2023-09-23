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
from tvm import relax

import tvm.script
from tvm.script import ir as I, relax as R, tir as T


def test_vm_builtin_lower_mem_alloc_storage():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()

            storage = R.memory.alloc_storage(R.shape([m * n * 4]), 0, "global", "uint8")
            alloc = R.memory.alloc_tensor(storage, 0, R.shape([m, n]), "float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            # we expected RemovePurityChecking to have been called first
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()

            storage = R.vm.alloc_storage(R.shape([m * n * 4]), R.prim_value(0), "uint8", "global")
            alloc = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([m, n]), "float32")

            _ = R.call_packed(
                "test.op.identity", x, alloc, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    After = relax.transform.VMBuiltinLower()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_vm_builtin_alloc_tensor_raises_error():
    """R.builtin.alloc_tensor should be handled earlier"""

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor:
            R.func_attr({"relax.force_pure": True})
            m, n = T.int64(), T.int64()

            alloc = R.builtin.alloc_tensor(R.shape([m, n]), runtime_device_index=0, dtype="float32")
            _ = R.call_packed(
                "test.op.identity", x, alloc, sinfo_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            gv0 = alloc
            return gv0

    with pytest.raises(tvm.TVMError):
        relax.transform.VMBuiltinLower()(Before)


if __name__ == "__main__":
    tvm.testing.main()
