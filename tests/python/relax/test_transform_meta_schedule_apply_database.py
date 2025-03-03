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
from tvm import tir
from tvm import meta_schedule as ms
from tvm import relax
from tvm.script import ir as I, tir as T

target = tvm.target.Target("llvm --num-cores=16")


def test_apply_to_func_with_different_block_name():
    @I.ir_module
    class RecordModule:
        @T.prim_func
        def main(A: T.Buffer((2,), "float32"), B: T.Buffer((2,), "float32")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            for i in T.serial(2):
                with T.block("block"):
                    vi = T.axis.spatial(2, i)
                    B[vi] = A[vi]

    @I.ir_module
    class BlockRenamedModule:
        @T.prim_func
        def main(A: T.Buffer((2,), "float32"), B: T.Buffer((2,), "float32")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            for i in T.serial(2):
                with T.block("renamed_block"):
                    vi = T.axis.spatial(2, i)
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer((2,), "float32"), B: T.Buffer((2,), "float32")):
            T.func_attr(
                {
                    "tir.is_scheduled": T.bool(True),
                    "global_symbol": "main",
                    "tir.noalias": T.bool(True),
                }
            )
            for i in T.serial(2):
                with T.block("renamed_block"):
                    vi = T.axis.spatial(2, i)
                    B[vi] = A[vi]

    def create_trace(mod: tvm.IRModule):
        sch = tir.Schedule(mod)
        _ = sch.get_block("block")
        return sch.trace

    db = ms.database.create(kind="memory")
    db.commit_workload(RecordModule)
    db.commit_tuning_record(
        ms.database.TuningRecord(
            create_trace(RecordModule), ms.database.Workload(RecordModule), [0.0], target
        )
    )

    with db, target:
        mod = relax.transform.MetaScheduleApplyDatabase()(BlockRenamedModule)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
