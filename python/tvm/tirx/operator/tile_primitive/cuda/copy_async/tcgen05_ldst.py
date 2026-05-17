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

"""copy_async dispatch: ``tcgen05.ld`` / ``tcgen05.st`` (tmem <-> local registers).

Both are inherently async; this dispatch emits the PTX instruction only and
leaves completion (``tcgen05.wait.ld`` / ``tcgen05.wait.st``) to the caller.
Callers that want sync semantics should issue the matching wait after the copy.
"""

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.layout import S, TCol, TileLayout, TLane, tid_in_wg
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import get_st_extent
from ..copy import _is_valid_copy, _scope_allowed
from ..exec_scope_utils import exec_scope_ok


def copy_tmem_local_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = op_call.dst, op_call.src
    dst: Buffer = dst_buffer_region.buffer
    src: Buffer = src_buffer_region.buffer

    if src.scope() == "tmem" and dst.scope() == "local":
        direction = "tmem2local"
        tmem_region, local_region = src_buffer_region, dst_buffer_region
    elif src.scope() == "local" and dst.scope() == "tmem":
        direction = "local2tmem"
        local_region, tmem_region = src_buffer_region, dst_buffer_region
    else:
        raise ValueError(f"Unsupported src scope {src.scope()} and dst scope {dst.scope()}")

    tmem_buf, local_buf = tmem_region.buffer, local_region.buffer

    assert tmem_buf.layout is not None
    assert local_buf.layout is not None
    assert tmem_buf.dtype == local_buf.dtype

    analyzer = Analyzer()
    elem_size = DataType(local_buf.dtype).bits
    elem_per_32b = 32 // elem_size
    assert len(local_buf.shape) == len(tmem_buf.shape) == 2
    # local: 128xWIDTH <-> tmem: 128xSHAPE[1]
    assert analyzer.can_prove_equal(local_buf.shape[0], 128)
    assert analyzer.can_prove_equal(tmem_buf.shape[0], 128)

    # Check width is valid for 32x32b, and determine num
    width = local_region.region[1].extent
    candidates = [1, 2, 4, 8, 16, 32, 64, 128]

    if not analyzer.can_prove_equal(tvm.tirx.floormod(width, elem_per_32b), 0):
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    num = None
    for n in candidates:
        if analyzer.can_prove_equal(tvm.tirx.floordiv(width, elem_per_32b), n):
            num = n
            break
    else:
        raise ValueError(f"Width {width} is not valid for tcgen05.ld/st with shape 32x32b")

    tmem_st, tmem_extent = get_st_extent(tmem_region)
    local_st, local_extent = get_st_extent(local_region)
    # tmem layout (128, WIDTH):(1@TLane, 1@TCol)
    tmem_layout = TileLayout(S[(128, tmem_buf.shape[1]) : (1 @ TLane, 1 @ TCol)]).canonicalize()
    # local layout
    TileLayout(S[(128, width) : (1 @ tid_in_wg, 1)]).canonicalize()

    # tmem allocated addr is not None
    assert tmem_buf.allocated_addr is not None
    tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), tmem_layout)
    # tvm.ir.assert_structural_equal(local_buf.layout.canonicalize(), local_layout)
    # local: [0:128, 0:WIDTH] <-> tmem: [0:128, st:st+WIDTH]
    assert analyzer.can_prove_equal(tmem_st[0], 0)
    assert analyzer.can_prove_equal(tmem_extent[0], 128)

    assert analyzer.can_prove_equal(local_st[0], 0)
    assert analyzer.can_prove_equal(local_extent[0], 128)

    offset = tmem_st[1]
    assert analyzer.can_prove_equal(tvm.tirx.floormod(offset, elem_per_32b), 0)
    offset_32b = tvm.tirx.floordiv(offset, elem_per_32b)
    assert analyzer.can_prove_equal(tmem_extent[1], width), (
        f"tmem_extent[1]: {tmem_extent[1]}, width: {width}"
    )

    # assert analyzer.can_prove_equal(local_st[1], 0)
    assert analyzer.can_prove_equal(local_extent[1], width)

    op = Tx.ptx.tcgen05.ld if direction == "tmem2local" else Tx.ptx.tcgen05.st

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def impl():
        with Tx.warp():
            local_storage = local_buf.view(local_buf.shape[1] * elem_per_32b, layout=TileLayout(S[num * elem_per_32b]))  # noqa: E501
            local_32b = local_storage.view("uint32")
            op(tmem_buf.allocated_addr[0], *[local_32b[local_st[1] // elem_per_32b+i] for i in range(num)], shape="32x32b", num=num, row=0, col=offset_32b)  # noqa: E501
    # fmt: on
    return impl


# === Variant: copy_async/tmem<->local (priority=10) ===
#
# When: one buffer is in tmem (tensor memory, Blackwell SM100+) and the other
# is in local scope, at warpgroup exec scope.
#
# Emits: Tx.ptx.tcgen05.ld / Tx.ptx.tcgen05.st (async). The caller is
# responsible for issuing the matching ``Tx.ptx.tcgen05.wait.ld`` /
# ``Tx.ptx.tcgen05.wait.st`` when synchronization is required.
@register_dispatch(
    "copy_async",
    "cuda",
    variant="tmem<->local",
    priority=10,
    when=[
        predicate("validate_copy_op", _is_valid_copy),
        predicate("exec_scope", exec_scope_ok, expected_scopes=["warpgroup"]),
        predicate(
            "storage_scope", _scope_allowed, allowed_pairs=[("tmem", "local"), ("local", "tmem")]
        ),
    ],
)
def copy_async_schedule_tmem_local_async(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> PrimFunc:
    return copy_tmem_local_impl(op_call, sctx)
