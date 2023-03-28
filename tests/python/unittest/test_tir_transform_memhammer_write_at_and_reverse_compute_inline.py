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
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir import Schedule


@I.ir_module
class Module:
    @T.prim_func
    def main(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in T.thread_binding(32, thread="threadIdx.x"):
            for j in range(4):
                with T.block("B"):
                    v0, v1 = T.axis.remap("SS", [i, j])
                    T.reads([a[v0 * 4 + v1]])
                    T.writes([b[v0 * 4 + v1]])
                    b[v0 * 4 + v1] = a[v0 * 4 + v1] * 2.0


expected_script = """# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(a: T.Buffer((128,), "float32"), b: T.Buffer((128,), "float32")):
        T.func_attr({"global_symbol": "main"})
        # with T.block("root"):
        b_reindex_shared_dyn = T.alloc_buffer((32, 4), scope="shared.dyn")
        for i in T.thread_binding(32, thread="threadIdx.x"):
            for j in range(4):
                with T.block("B"):
                    v0, v1 = T.axis.remap("SS", [i, j])
                    T.reads(a[v0 * 4 + v1])
                    T.writes(b_reindex_shared_dyn[v0, v1])
                    b_reindex_shared_dyn[v0, v1] = a[v0 * 4 + v1] * T.float32(2)
                with T.block("b_reindex_shared.dyn"):
                    v0 = T.axis.spatial(4, j)
                    T.reads(b_reindex_shared_dyn[0:32, v0])
                    T.writes(b[v0:v0 + 125])
                    T.block_attr({"auto_copy": 1})
                    for ax0, ax1 in T.grid(32, 1):
                        b[ax0 * 4 + (v0 + ax1)] = b_reindex_shared_dyn[ax0, v0 + ax1]"""


def test_write_at_and_reverse_compute_inline():
    sch = Schedule(Module)
    sch.work_on("main")
    block_B = sch.get_block("B")
    block_B_reindex = sch.reindex(block_B, buffer=("write", 0))
    _, j = sch.get_loops(block_B)
    sch.write_at(j, block_B, 0, "shared.dyn")
    sch.reverse_compute_inline(block_B_reindex)
    assert sch.mod.script() == expected_script


if __name__ == "__main__":
    test_write_at_and_reverse_compute_inline()