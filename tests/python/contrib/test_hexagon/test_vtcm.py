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

import pytest
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from .infrastructure import get_hexagon_target


@T.prim_func
def scale_by_two(buffer_a: T.Buffer((8192,), "int8"), buffer_c: T.Buffer((8192,), "int8")):
    for i in T.serial(
        0,
        8192,
    ):
        with T.block("C"):
            buffer_c[i] = buffer_a[i] * T.int8(2)


def get_scale_by_two_schedule():
    mod = tvm.IRModule.from_expr(scale_by_two.with_attr("global_symbol", "main"))
    sch = tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_block("C")
    (flat,) = sch.get_loops(block_c)
    outer, _, _, _ = sch.split(flat, factors=[8, 4, 2, 128])
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, outer)
    return sch


@tvm.testing.requires_hexagon
def test_vtcm_building():
    """Test building with vtcm mem scope"""
    sch = get_scale_by_two_schedule()
    target = get_hexagon_target("v68")
    built = tvm.build(sch.mod, target=target)
    assert "global.vtcm" in built.get_source("asm")


@tvm.testing.requires_hexagon
@pytest.mark.parametrize("vtcm_capacity,limited", [(8192, False), (1024, False), (128, True)])
def test_vtcm_limit(vtcm_capacity, limited):
    """Test building with vtcm mem scope limit"""
    sch = get_scale_by_two_schedule()

    def _raises_exception(f):
        try:
            f()
        except tvm._ffi.base.TVMError:
            return True
        return False

    target = get_hexagon_target("v68", vtcm_capacity=vtcm_capacity)

    assert (
        _raises_exception(lambda: tvm.build(sch.mod, target=target)) == limited
    ), "Case 1 - arg. VTCM memory allocation limiter does not work correctly "

    with target:
        assert (
            _raises_exception(lambda: tvm.build(sch.mod)) == limited
        ), "Case 2 - with.VTCM memory allocation limiter does not work correctly "

    with tvm.transform.PassContext(config={"tir.vtcm_capacity": vtcm_capacity}):
        assert (
            _raises_exception(
                lambda: tvm.build(sch.mod, target=get_hexagon_target("v68", vtcm_capacity=0))
            )
            == limited
        ), "Case 3 - context. VTCM memory allocation limiter does not work correctly "


if __name__ == "__main__":
    tvm.testing.main()
