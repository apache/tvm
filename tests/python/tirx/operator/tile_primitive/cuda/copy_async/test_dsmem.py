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
"""Tests for the DSMEM (shared::cta → shared::cluster) copy_async variant.

Split out from ``test_copy_async.py`` so the TMA-focused file stays focused
on the g2s/s2g TMA family. Any cross-cutting copy_async helper that both
files need should live in a shared module, not be duplicated.
"""

import functools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx import IntImm, Var
from tvm.tirx.cuda.operator.tile_primitive.copy_async.dsmem import copy_dsmem_impl
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.operator.tile_primitive.dispatch_context import DispatchContext
from tvm.tirx.operator.tile_primitive.dispatcher import DispatchFail
from tvm.tirx.operator.tile_primitive.ops import CopyAsync
from tvm.tirx.stmt_functor import StmtExprVisitor


def _make_dsmem_dispatch_call(shape, dtype, src_layout, dst_layout):
    """Call copy_dsmem_impl directly. Returns impl or raises DispatchFail."""
    from tvm.ir import Range
    from tvm.tirx.stmt import BufferRegion

    src_buf = tvm.tirx.decl_buffer(shape, dtype, "A", scope="shared.dyn", layout=src_layout)
    dst_buf = tvm.tirx.decl_buffer(shape, dtype, "B", scope="shared.dyn", layout=dst_layout)
    ranges = [Range.from_min_extent(0, s) for s in shape]
    config = {"mbar": Var("mbar", "handle"), "remote_cta_id": IntImm("int32", 1)}
    op_call = CopyAsync(BufferRegion(dst_buf, ranges), BufferRegion(src_buf, ranges), config=config)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {})
    return copy_dsmem_impl(op_call, sctx)


class _S2CCounter(StmtExprVisitor):
    """Count cp.async.bulk.shared_to_cluster calls including loop iterations."""

    def __init__(self):
        super().__init__()
        self._loop_extents = []
        self.total = 0

    def visit_for_(self, op):
        self._loop_extents.append(op.extent)
        self.visit_stmt(op.body)
        self._loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.tirx.Call):
            if op.value.op.name == "tirx.ptx.cp_async_bulk_shared_to_cluster":
                n = 1
                for e in self._loop_extents:
                    n *= e
                self.total += n


def _count_s2c_ops(impl):
    c = _S2CCounter()
    c.visit_stmt(impl.body)
    return c.total


# ---------------------------------------------------------------------------
# Parametrized DSMEM test: dispatch assertion + GPU correctness
# ---------------------------------------------------------------------------

# (shape, dtype, src_spec, dst_spec, expected_s2c_ops | "fail")
# Dispatch assertion uses src_spec/dst_spec as given.
# GPU correctness (all non-fail cases) uses src_spec as the layout for both CTAs.
DSMEM_CONFIGS = [
    pytest.param((128, 64), "float16", S[128, 64], S[128, 64], 1, id="contiguous-2d"),
    pytest.param((256,), "float16", S[256], S[256], 1, id="contiguous-1d"),
    # Stride gap: inner 128 contiguous, outer stride=256 (gap) → 8 bulk copies
    pytest.param(
        (8, 128), "float16", S[(8, 128) : (256, 1)], S[(8, 128) : (256, 1)], 8, id="stride-gap"
    ),
    # Different outer strides → 8 bulk copies in dispatch
    pytest.param(
        (8, 128),
        "float16",
        S[(8, 128) : (256, 1)],
        S[(8, 128) : (512, 1)],
        8,
        id="partial-contiguity-diff-stride",
    ),
    # Incompatible: row-major vs column-major → DispatchFail
    pytest.param(
        (4, 64), "float16", S[4, 64], S[(4, 64) : (1, 4)], "fail", id="incompatible-row-vs-col"
    ),
]


def _layout_physical_elements(layout):
    """Compute number of physical elements needed for a TileLayout."""
    max_offset = 0
    for shard in layout.shard:
        if shard.axis.is_memory():
            max_offset += int(shard.stride) * (int(shard.extent) - 1)
    return max_offset + 1


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("shape,dtype,src_spec,dst_spec,expected", DSMEM_CONFIGS)
def test_dsmem(shape, dtype, src_spec, dst_spec, expected):
    """Dispatch assertion + GPU correctness for DSMEM copy.

    Always tests dispatch (s2c op count or DispatchFail).
    For non-fail cases: also runs a 2-CTA cluster kernel via T.copy_async
    dispatch (using src_spec as layout for both CTAs) and verifies correctness.
    """
    from tvm.tirx.lang.pipeline import MBarrier

    src_layout = TileLayout(src_spec)
    dst_layout = TileLayout(dst_spec)

    # --- Dispatch assertion ---
    if expected == "fail":
        with pytest.raises(DispatchFail):
            _make_dsmem_dispatch_call(shape, dtype, src_layout, dst_layout)
        return

    impl = _make_dsmem_dispatch_call(shape, dtype, src_layout, dst_layout)
    assert _count_s2c_ops(impl) == expected

    # --- GPU correctness ---
    # Allocate two separate smem buffers: src_smem (src_layout) and dst_smem
    # (dst_layout). CTA 0 loads global→src_smem, copy_async copies src_smem→
    # dst_smem on CTA 1. CTA 1 reads dst_smem and writes to global output.

    CLUSTER_N = 2
    n_elements = functools.reduce(lambda a, b: a * b, shape, 1)
    copy_bytes = n_elements * tvm.DataType(dtype).bits // 8
    src_phys = _layout_physical_elements(src_layout)
    dst_phys = _layout_physical_elements(dst_layout)
    r = tuple(slice(0, s) for s in shape)

    # fmt: off
    @T.prim_func
    def dsmem_copy(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, shape, dtype)
        B = T.match_buffer(B_ptr, shape, dtype)

        T.device_entry()
        cbx = T.cta_id_in_cluster([CLUSTER_N])
        T.cta_id([CLUSTER_N])
        tid = T.thread_id([1])
        pool = T.SMEMPool()
                # src_smem: CTA 0 writes here, dispatch reads from here
        src_raw = pool.alloc([src_phys], dtype, align=128)
        src_smem = T.decl_buffer(
            list(shape), dtype, src_raw.data,
            elem_offset=0, scope="shared.dyn", layout=src_layout,
        )
                # dst_smem: dispatch writes here (on remote CTA), CTA 1 reads
        dst_raw = pool.alloc([dst_phys], dtype, align=128)
        dst_smem = T.decl_buffer(
            list(shape), dtype, dst_raw.data,
            elem_offset=0, scope="shared.dyn", layout=dst_layout,
        )
        mbar = MBarrier(pool, 1)
        pool.commit()

        mbar.init(1)
        T.ptx.fence.mbarrier_init()
        T.cuda.cluster_sync()

        if tid == 0:
            if cbx == 0:
                Tx.copy(src_smem[r], A[r])
                T.ptx.fence.proxy_async("shared::cta")

                Tx.copy_async(
                    dst_smem[r], src_smem[r],
                    dispatch="dsmem",
                    mbar=mbar.ptr_to([0]),
                    remote_cta_id=T.int32(1),
                )
            else:
                T.ptx.mbarrier.arrive.expect_tx(mbar.ptr_to([0]), copy_bytes)
                mbar.wait(0, 0)

                Tx.copy(B[r], dst_smem[r])
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": dsmem_copy})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        cuda_src = mod.mod.imports[0].inspect_source()
        assert "cp.async.bulk.shared::cluster.shared::cta" in cuda_src

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, shape)
        B_np = np.zeros(shape, dtype=np_dtype)

        A_tvm = tvm.runtime.tensor(A_np, dev)
        B_tvm = tvm.runtime.tensor(B_np, dev)
        mod(A_tvm, B_tvm)
        np.testing.assert_allclose(A_np, B_tvm.numpy())


def test_dsmem_dispatch_missing_config():
    """Dispatch fails when required config keys are missing."""
    from tvm.ir import Range
    from tvm.tirx.stmt import BufferRegion

    layout = TileLayout(S[64])
    buf = tvm.tirx.decl_buffer((64,), "float16", "A", scope="shared.dyn", layout=layout)
    br = BufferRegion(buf, [Range.from_min_extent(0, 64)])
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {})

    with pytest.raises(DispatchFail, match="remote_cta_id"):
        copy_dsmem_impl(CopyAsync(br, br, config={"mbar": Var("m", "handle")}), sctx)
    with pytest.raises(DispatchFail, match="mbar"):
        copy_dsmem_impl(CopyAsync(br, br, config={"remote_cta_id": IntImm("int32", 1)}), sctx)


if __name__ == "__main__":
    tvm.testing.main()
