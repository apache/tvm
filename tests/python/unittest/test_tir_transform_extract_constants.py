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
import numpy as np
import tvm
from tvm import tir
from tvm.script import tir as T
import tvm.testing


@tvm.script.ir_module
class Module4:
    @T.prim_func
    def constant1(a: T.handle) -> None:
        A = T.match_buffer(a, (10), "int32")
        B = T.alloc_buffer((10), "int32")
        K_data = T.allocate_const([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
        K = T.Buffer(shape=(10), dtype="int32", data=K_data)
        for x in T.serial(0, 10):
            B[x] = A[x] + K[x]

    @T.prim_func
    def constant2(a: T.handle) -> None:
        A = T.match_buffer(a, (10), "int32")
        B = T.alloc_buffer((10), "int32")
        K_data = T.allocate_const([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
        K = T.Buffer(shape=(10), dtype="int32", data=K_data)
        for x in T.serial(0, 10):
            B[x] = A[x] + K[x]

    @T.prim_func
    def constant3(a: T.handle) -> None:
        A = T.match_buffer(a, (10), "int32")
        B = T.alloc_buffer((10), "int32")
        K_data = T.allocate_const([1, 2, 3, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
        K = T.Buffer(shape=(10), dtype="int32", data=K_data)
        for x in T.serial(0, 10):
            B[x] = A[x] + K[x]


def test_const_extraction():
    mod = tvm.tir.transform.ExtractPrimFuncConstants()(Module4)
    constants = mod.attrs["constants"]
    assert len(constants) == 2

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.AllocateConst):
            assert np.array_equal(stmt.data.numpy(), constants[int(stmt.irmod_storage_idx)].numpy())

    for n, f in mod.functions.items():
        tvm.tir.stmt_functor.post_order_visit(f.body, _visit)

    tvm.lower(mod)


if __name__ == "__main__":
    tvm.testing.main()
