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
"""Tests for tvm.tirx.lang.alloc_pool validation."""

import pytest

from tvm.tirx.lang.alloc_pool import _validate_mma_alloc_shape
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import SwizzleMode

# ---------------------------------------------------------------------------
# alloc_mma shape validation: bad inputs raise actionable ValueError instead of
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
