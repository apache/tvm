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
"""VTCM Tests"""

import tvm.testing
from tvm import tir
from tvm.script import tir as T


@T.prim_func
def scale_by_two(buffer_a: T.Buffer[(8192,), "int8"], buffer_c: T.Buffer[(8192,), "int8"]):
    for i in T.serial(
        0,
        8192,
    ):
        with T.block("C"):
            buffer_c[i] = buffer_a[i] * T.int8(2)


def test_vtcm_lowering():
    """Test lowering with vtcm mem scope"""
    mod = tvm.IRModule.from_expr(scale_by_two.with_attr("global_symbol", "main"))
    sch = tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_block("C")
    (flat,) = sch.get_loops(block_c)
    outer, _, _, _ = sch.split(flat, factors=[8, 4, 2, 128])
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, outer)
    lowered = tvm.lower(sch.mod["main"])

    def ir_module_has_allocate_nodes(irmod):
        nallocs = 0

        def _visit(stmt):
            nonlocal nallocs
            if isinstance(stmt, tvm.tir.Allocate):
                nallocs += 1

        tvm.tir.stmt_functor.post_order_visit(irmod["main"].body, _visit)
        return nallocs

    assert not ir_module_has_allocate_nodes(lowered), (
        "AllocateNode found in lowered IRModule, "
        "VTCM allocations should have been lowered to tir.nd_mem_alloc_with_scope"
    )


if __name__ == "__main__":
    tvm.testing.main()
