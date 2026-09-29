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
"""Tests for CUDA allocation pool validation."""

import pytest
import tvm_ffi

import tvm
from tvm.script import tirx as T
from tvm.tirx.cuda.lang.alloc_pool import _validate_mma_alloc_shape
from tvm.tirx.cuda.tile_primitive.tma_utils import SwizzleMode

# ---------------------------------------------------------------------------
# alloc_tcgen05_mma_AB shape validation: bad inputs raise actionable ValueError instead of
# the opaque "Divide by zero" diagnostic that ``Layout.tile_to`` would emit.
# ---------------------------------------------------------------------------


class TestAllocMmaValidationRowBytes:
    """row width (cols * itemsize) must be a positive multiple of swizzle atom bytes."""

    def test_bf16_32cols_128b_swizzle_too_narrow(self):
        # The exact case that bit gdn-prefill v1_0 / v1_2 (eval R10).
        # Row = 32 * 2B = 64B < 128B atom.
        with pytest.raises(ValueError, match=r"64B rows.*128B swizzle atom"):
            _validate_mma_alloc_shape((128, 32), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)

    def test_error_suggests_smaller_swizzle(self):
        try:
            _validate_mma_alloc_shape((128, 32), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)
        except ValueError as e:
            assert "SWIZZLE_64B_ATOM" in str(e), f"missing fix-it hint: {e}"
        else:
            pytest.fail("should have raised")

    def test_error_suggests_widening_cols(self):
        try:
            _validate_mma_alloc_shape((128, 32), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)
        except ValueError as e:
            assert "multiple of 64 elements" in str(e), f"missing widen hint: {e}"
        else:
            pytest.fail("should have raised")

    def test_fp32_16cols_128b_swizzle_too_narrow(self):
        # Row = 16 * 4B = 64B < 128B atom.
        with pytest.raises(ValueError, match=r"64B rows.*128B swizzle atom"):
            _validate_mma_alloc_shape((128, 16), "float32", SwizzleMode.SWIZZLE_128B_ATOM)

    def test_3d_shape_validates_last_dim(self):
        # Validation must consider shape[-1], not shape[0].
        with pytest.raises(ValueError, match=r"64B rows"):
            _validate_mma_alloc_shape((2, 128, 32), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)


class TestAllocMmaValidationRowCount:
    """rows (shape[-2]) must be a positive multiple of the 8-row atom."""

    def test_rows_below_atom_rejected(self):
        with pytest.raises(ValueError, match=r"shape\[-2\]=4.*multiple of 8"):
            _validate_mma_alloc_shape((4, 64), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)

    def test_rows_not_multiple_of_8_rejected(self):
        with pytest.raises(ValueError, match=r"shape\[-2\]=12.*multiple of 8"):
            _validate_mma_alloc_shape((12, 64), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)


class TestAllocMmaValidationRank:
    """rank-1 shapes cannot be tiled with a 2-D swizzle atom."""

    def test_rank_one_rejected(self):
        with pytest.raises(ValueError, match=r"fewer than 2 dimensions"):
            _validate_mma_alloc_shape((128,), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM)


class TestAllocMmaValidationValid:
    """combinations that should succeed must not be rejected."""

    @pytest.mark.parametrize(
        "shape,dtype,mode",
        [
            # The fix path the agent should pick when row_bytes >= 128.
            ((128, 64), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM),
            ((128, 128), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM),
            # Or downgrade to a swizzle whose atom matches the row.
            ((128, 32), "bfloat16", SwizzleMode.SWIZZLE_64B_ATOM),
            ((128, 16), "bfloat16", SwizzleMode.SWIZZLE_32B_ATOM),
            # 3-D request validates the last two dims only.
            ((2, 128, 64), "bfloat16", SwizzleMode.SWIZZLE_128B_ATOM),
            # fp32 with row width >= atom.
            ((128, 32), "float32", SwizzleMode.SWIZZLE_128B_ATOM),
            # fp8 (1B) with row width >= atom.
            ((128, 128), "float8_e4m3", SwizzleMode.SWIZZLE_128B_ATOM),
        ],
    )
    def test_valid_combinations_accepted(self, shape, dtype, mode):
        _validate_mma_alloc_shape(shape, dtype, mode)

    def test_swizzle_none_skips_validation(self):
        # SWIZZLE_NONE has no atom — even otherwise-bad shapes are allowed.
        _validate_mma_alloc_shape((128, 32), "bfloat16", SwizzleMode.SWIZZLE_NONE)
        _validate_mma_alloc_shape((3, 5), "bfloat16", SwizzleMode.SWIZZLE_NONE)
        _validate_mma_alloc_shape((128,), "bfloat16", SwizzleMode.SWIZZLE_NONE)


@pytest.mark.parametrize("size", [None, 128])
def test_smem_pool_commits_allocation_extent(size):
    @T.prim_func
    def kernel():
        T.func_attr({"target": T.target("cuda", host="c")})
        T.attr(T.target("cuda"), "target", 0)
        pool = T.SMEMPool()
        first = pool.alloc((3,), "float4_e2m1fn")
        second = pool.alloc((4,), "float32", align=16)
        pool.move_base_to(64)
        pool.commit(size)
        T.evaluate(first.data)
        T.evaluate(second.data)

    allocations = []
    views = []

    def collect(node):
        if isinstance(node, tvm.tirx.AllocBuffer):
            allocations.append(node)
        if isinstance(node, tvm.tirx.DeclBuffer):
            views.append(node)

    tvm_ffi.structural_walk(kernel.body, collect)
    assert len(allocations) == 1
    backing = allocations[0].buffer
    assert int(backing.shape[0]) == (64 if size is None else size)
    assert len(views) == 2
    assert all(view.data.args[0].same_as(backing) for view in views)
    assert int(views[1].buffer.elem_offset) == 4
    tvm.ir.assert_structural_equal(
        kernel, tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    )

    split = tvm.tirx.transform.SplitHostDevice()(tvm.IRModule({"kernel": kernel}))
    calls = []

    def collect_launch(node):
        if isinstance(node, tvm.ir.Call) and node.op.name == "tirx.call_ffi_kernel":
            calls.append(node)

    tvm_ffi.structural_walk(split["kernel"].body, collect_launch)
    assert len(calls) == 1
    analyzer = tvm.sym.Analyzer()
    assert int(analyzer.simplify(calls[0].args[-1])) == (64 if size is None else size)


def test_smem_pool_requires_commit():
    with pytest.raises(ValueError, match=r"SMEMPool.commit\(\) must be called"):

        @T.prim_func
        def kernel():
            pool = T.SMEMPool()
            view = pool.alloc((16,), "uint8")
            T.evaluate(view.data)


def test_smem_pool_commit_rejects_small_size():
    with pytest.raises(AssertionError, match="smaller than"):

        @T.prim_func
        def kernel():
            pool = T.SMEMPool()
            view = pool.alloc((16,), "uint8")
            pool.commit(8)
            T.evaluate(view.data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
