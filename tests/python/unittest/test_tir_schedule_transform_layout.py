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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16


@T.prim_func
def two_elementwise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_intermediate_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((8, 8, 16, 16), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0


@T.prim_func
def two_elementwise_transformed_input_buffer(
    A: T.Buffer[(8, 8, 16, 16), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi // 16, vj // 16, vi % 16, vj % 16] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_output_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 8, 16, 16), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi // 16, vj // 16, vi % 16, vj % 16] = B[vi, vj] + 1.0


@T.prim_func
def elementwise(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]) -> None:
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def elementwise_transformed(A: T.Buffer[(128, 128), "float32"], B: T.Buffer[(128, 128), "float32"]) -> None:
    for i in range(16384):
        with T.block("B"):
            vi = T.axis.remap("S", [i])
            B[vi // 128, vi % 128] = A[vi // 128, vi % 128] * 2.0


@T.prim_func
def conv2d_nhwc(
    Input: T.Buffer[(1, 224, 224, 3), "float32"],
    Weight: T.Buffer[(7, 7, 3, 64), "float32"],
    Conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"],
) -> None:
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                ((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)),
                Input[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1],
                T.float32(0),
                dtype="float32",
            )
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with T.block("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with T.init():
                Conv2d_nhwc[n, h, w, co] = T.float32(0)
            Conv2d_nhwc[n, h, w, co] = Conv2d_nhwc[n, h, w, co] + (
                PadInput[n, ((h * 2) + rh), ((w * 2) + rw), ((T.floordiv(co, 64) * 3) + rc)]
                * Weight[rh, rw, rc, co]
            )


@T.prim_func
def conv2d_nhwc_transformed(
    Input: T.Buffer[(1, 224, 224, 3), "float32"],
    Weight: T.Buffer[(7, 7, 3, 64), "float32"],
    Conv2d_nhwc: T.Buffer[(1, 112, 112, 64), "float32"],
) -> None:
    PadInput = T.alloc_buffer([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with T.block("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(Input[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            T.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                i1_1 >= 3 and i1_1 < 227 and i2_1 >= 3 and i2_1 < 227,
                Input[i0_1, i1_1 - 3, i2_1 - 3, i3_1],
                T.float32(0),
                dtype="float32",
            )
    for ax0, ax1, ax2 in T.grid(12544, 64, 147):
        with T.block("conv2d_nhwc"):
            v0, v1, v2 = T.axis.remap("SSR", [ax0, ax1, ax2])
            T.reads(PadInput[v0 // 12544, v0 // 112 * 2 + v2 // 21, v0 % 112 * 2 + v2 % 21 // 3, v2 % 3], Weight[v2 // 21, v2 % 21 // 3, v2 % 3, v1])
            T.writes(Conv2d_nhwc[v0 // 12544, v0 // 112, v0 % 112, v1])
            with T.init():
                Conv2d_nhwc[v0 // 12544, v0 // 112, v0 % 112, v1] = T.float32(0)
            Conv2d_nhwc[v0 // 12544, v0 // 112, v0 % 112, v1] = Conv2d_nhwc[v0 // 12544, v0 // 112, v0 % 112, v1] + PadInput[v0 // 12544, v0 // 112 * 2 + v2 // 21, v0 % 112 * 2 + v2 % 21 // 3, v2 % 3] * Weight[v2 // 21, v2 % 21 // 3, v2 % 3, v1]


@T.prim_func
def two_elementwise_unit_dim(A: T.Buffer[(1, 128), "float32"], C: T.Buffer[(1, 128), "float32"]) -> None:
    B = T.alloc_buffer((1, 128), "float32")
    for i, j in T.grid(1, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(1, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})


def test_two_elementwise_transform_intermediate_buffer(use_block_name):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            block="B",
            buffer="B",
            index_map=packed_index_map_func,
        )
    else:
        block = sch.get_block("B")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_intermediate_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_input_buffer(use_block_name):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="B",
            buffer="A",
        )
    else:
        block = sch.get_block("B")
        sch.transform_layout(block, ("read", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_input_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_output_buffer(use_block_name):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="C",
            buffer="C",
        )
    else:
        block = sch.get_block("C")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_output_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_unit_dim(use_block_name):
    sch = tir.Schedule(two_elementwise_unit_dim, debug_mask="all")
    index_map = lambda i, j: (i, j)

    if use_block_name:
        sch.transform_layout(
            index_map=index_map,
            block="B",
            buffer="B",
        )
    else:
        block = sch.get_block("B")
        sch.transform_layout(block, ("write", 0), index_map)

    tvm.ir.assert_structural_equal(two_elementwise_unit_dim, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise_unit_dim)


def test_simplify():
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    i, j = sch.get_loops(sch.get_block("C"))
    i, i_inner = sch.split(i, factors=[None, 16])
    j, j_inner = sch.split(j, factors=[None, 16])

    sch.reorder(
        i,
        j,
        i_inner,
        j_inner,
    )

    block_outer = sch.blockize(i_inner)

    B = sch.cache_read(block_outer, 0, "global")
    sch.transform_layout(B, ("write", 0), lambda i, j: (i // 16, j // 16, i % 16, j % 16))

    @T.prim_func
    def ref(B: T.Buffer[(8, 8, 16, 16), "float32"], C: T.Buffer[(128, 128), "float32"]):
        for i_0, j_0 in T.grid(8, 8):
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o, vj_o, 0:16, 0:16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o, vj_o, vi, vj])
                        T.writes(C[vi_o * 16 + vi, vj_o * 16 + vj])
                        C[vi_o * 16 + vi, vj_o * 16 + vj] = B[vi_o, vj_o, vi, vj] + T.float32(1)

                        # Without simplification
                        # T.reads(B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16])
                        # C[...] = B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16] + T.float32(1)

    tvm.ir.assert_structural_equal(ref.body.block.body, sch.get(sch.get_loops(block_outer)[0]))


def test_var_args_sugar():
    @T.prim_func
    def summation_3d(
        A: T.Buffer[(1024, 1024, 32), "float32"], B: T.Buffer[(1,), "float32"]
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with T.block("compute"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk]

    @T.prim_func
    def summation_3d_split(
        A: T.Buffer[(1024, 1024, 8, 4), "float32"], B: T.Buffer[(1,), "float32"]
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with T.block("compute"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk // 4, vk % 4]

    sch = tir.Schedule(summation_3d, debug_mask="all")
    sch.transform_layout(
        index_map=lambda *indices, k: [*indices, k // 4, k % 4], block="compute", buffer="A"
    )
    tvm.ir.assert_structural_equal(summation_3d_split, sch.mod["main"])


def test_transform_block_layout_basic(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_block("B")
    sch.transform_block_layout(block, lambda i, j: (i * 128 + j,))
    tvm.ir.assert_structural_equal(elementwise_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_transform_block_layout_conv2d_nhwc(use_block_name):
    sch = tir.Schedule(conv2d_nhwc, debug_mask="all")
    block = "conv2d_nhwc" if use_block_name else sch.get_block("conv2d_nhwc")
    sch.transform_block_layout(
        block,
        lambda n, h, w, co, rh, rw, rc: (n * 112 * 112 + h * 112 + w, co, rh * 7 * 3 + rw * 3 + rc),
    )
    tvm.ir.assert_structural_equal(conv2d_nhwc_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_transform_block_layout_unit_dim(use_block_name):
    sch = tir.Schedule(two_elementwise_unit_dim, debug_mask="all")
    block = "B" if use_block_name else sch.get_block("B")
    sch.transform_block_layout(block, lambda i, j: (j, i))

    @T.prim_func
    def two_elementwise_unit_dim_transformed(
        A: T.Buffer[(1, 128), "float32"], C: T.Buffer[(1, 128), "float32"]
    ) -> None:
        B = T.alloc_buffer((1, 128), "float32")
        for j, i in T.grid(128, 1):
            with T.block("B"):
                vj, vi = T.axis.remap("SS", [j, i])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(1, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    tvm.ir.assert_structural_equal(two_elementwise_unit_dim_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise_unit_dim)


def test_transform_block_layout_fail_non_affine(use_block_name):
    sch = tir.Schedule(elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_block("B")
    with pytest.raises(tir.ScheduleError):
        sch.transform_block_layout(block, lambda i, j: (i + j,))


def test_transform_block_layout_fail_mixed_iter_type(use_block_name):
    sch = tir.Schedule(conv2d_nhwc, debug_mask="all")
    block = "conv2d_nhwc" if use_block_name else sch.get_block("conv2d_nhwc")
    with pytest.raises(tir.ScheduleError):
        sch.transform_block_layout(
            block,
            lambda n, h, w, co, rh, rw, rc: (n * 112 * 112 + h * 112 + w, co * 7 + rh, rw * 3 + rc),
        )


def test_transform_block_layout_int64_extent(use_block_name):
    @T.prim_func
    def elementwise_int64_extent(
        A: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0

    @T.prim_func
    def elementwise_int64_extent_transformed(
        A: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
        B: T.Buffer[(T.int64(128), T.int64(128)), "float32"],
    ) -> None:
        for i in range(T.int64(16384)):
            with T.block("B"):
                vi = T.axis.remap("S", [i])
                B[vi // T.int64(128), vi % T.int64(128)] = (
                    A[vi // T.int64(128), vi % T.int64(128)] * 2.0
                )

    sch = tir.Schedule(elementwise_int64_extent, debug_mask="all")
    block = "B" if use_block_name else sch.get_block("B")
    sch.transform_block_layout(block, lambda i, j: (i * 128 + j,))
    print(
        tvm.ir.base.get_first_structural_mismatch(
            elementwise_int64_extent_transformed, sch.mod["main"]
        )
    )
    tvm.ir.assert_structural_equal(elementwise_int64_extent_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_int64_extent)


class BasePaddingCompare(tvm.testing.CompareBeforeAfter):
    pad_value = tvm.testing.parameter(None)

    transformed_buffer = tvm.testing.parameter("A")

    index_map = tvm.testing.parameter(lambda i: [i // 4, i % 4])

    @pytest.fixture
    def transform(self, pad_value, transformed_buffer, index_map):
        def transform(mod):
            sch = tir.Schedule(mod)
            sch.transform_layout("block", transformed_buffer, index_map, pad_value=pad_value)
            return sch.mod

        return transform


class TestNoPadding(BasePaddingCompare):
    """Transformations without padding do not depend on pad_value."""

    pad_value = tvm.testing.parameter(None, 42)

    def before():
        A = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi] = 0

    def expected():
        A = T.alloc_buffer([4, 4], "int32")
        for i in T.serial(16):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi // 4, vi % 4] = 0


class TestNoPaddingMultipleUsage(BasePaddingCompare):
    """Transformations without padding do not depend on pad_value.

    Like TestNoPadding, but the buffer A shows up in multiple
    locations.  To remain internally consistent, all instances of the
    buffer should be rewritten.
    """

    pad_value = tvm.testing.parameter(None, 42)

    def before():
        A = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi] = 0

        B = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("other"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    def expected():
        A = T.alloc_buffer([4, 4], "int32")
        for i in T.serial(16):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi // 4, vi % 4] = 0

        B = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("other"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi // 4, vi % 4]


class TestNoPaddingOpaqueBlock(BasePaddingCompare):
    """Transformations without padding do not depend on pad_value.

    Like TestNoPadding, but buffer access is done in an opaque block.
    """

    pad_value = tvm.testing.parameter(None, 42)

    def before():
        A = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("block"):
                A[i] = 0

    def expected():
        A = T.alloc_buffer([4, 4], "int32")
        for i in T.serial(16):
            with T.block("block"):
                A[i // 4, i % 4] = 0


class TestErrorIfPaddingForbidden(BasePaddingCompare):
    """Unless padding is explicitly enabled, should raise error"""

    def before():
        A = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi] = 0

    expected = tvm.tir.schedule.schedule.ScheduleError


class TestErrorOnWrongPaddingType(BasePaddingCompare):
    """The padding must have the same dtype as the buffer"""

    pad_value = tvm.testing.parameter(tir.IntImm("int8", 0))

    def before():
        A = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi] = 0

    expected = tvm.tir.schedule.schedule.ScheduleError


class TestPaddedTransformIfThenElse(BasePaddingCompare):
    """Use if_then_else to represent padding, if possible.

    For a block that is a producer of the pre-transformation buffer,
    which visits all indices according to a row-major traversal, and
    which has no effect other than producing the transformed buffer,
    transform the loop iterators to be a row-major traversal of the
    post-transformation buffer, with padding represented by
    `T.if_then_else`.
    """

    pad_value = tvm.testing.parameter(0)
    transformed_buffer = tvm.testing.parameter("B")
    dtype = tvm.testing.parameter("int32", "int8")

    @tvm.testing.fixture
    def before(self, dtype):
        @T.prim_func
        def func(A: T.Buffer[14, dtype]):
            B = T.alloc_buffer(14, dtype)
            for i in T.serial(14):
                with T.block("block"):
                    vi = T.axis.remap("S", [i])
                    B[vi] = A[vi]

        return func

    @tvm.testing.fixture
    def expected(self, dtype, pad_value):
        pad_value = tir.IntImm(dtype, pad_value)

        @T.prim_func
        def func(A: T.Buffer[14, dtype]):
            B = T.alloc_buffer([4, 4], dtype)
            for i, j in T.grid(4, 4):
                with T.block("block"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = T.if_then_else(
                        vi == 3 and 2 <= vj, pad_value, A[vi * 4 + vj], dtype=dtype
                    )

        return func


class TestPaddedTransformWithoutLoop(BasePaddingCompare):
    """Handle padded writes without a loop

    The statement being replaced may be something other than a
    for-loop, such as if a loop has already been unrolled.
    """

    pad_value = tvm.testing.parameter(0)

    def before(A: T.Buffer[14, "int32"]):
        with T.block("root"):
            T.reads()
            T.writes()
            with T.block("block"):
                A[0] = 0

    def expected(A: T.Buffer[(4, 4), "int32"]):
        with T.block("block"):
            A[0, 0] = 0

        for i, j in T.grid(4, 4):
            with T.block("buffer_A_padding"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.where(i == 3 and 2 <= j)
                A[vi, vj] = 0


class TestPaddedTransformIfThenElseReduction(BasePaddingCompare):
    """Like TestPaddedTransformIfThenElse, but with a reduction axis"""

    pad_value = tvm.testing.parameter(0)
    transformed_buffer = tvm.testing.parameter("B")

    def before(A: T.Buffer[(14, 32), "int32"]):
        B = T.alloc_buffer(14, "int32")
        for i, k in T.grid(14, 32):
            with T.block("block"):
                vi, vk = T.axis.remap("SR", [i, k])
                with T.init():
                    B[vi] = 0
                B[vi] = B[vi] + A[vi, vk]

    def expected(A: T.Buffer[(14, 32), "int32"]):
        B = T.alloc_buffer([4, 4], "int32")
        for i, j, k in T.grid(4, 4, 32):
            with T.block("block"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    B[vi, vj] = T.if_then_else(vi == 3 and 2 <= vj, 0, 0, dtype="int32")
                B[vi, vj] = T.if_then_else(
                    vi == 3 and 2 <= vj, 0, B[vi, vj] + A[vi * 4 + vj, vk], dtype="int32"
                )


class TestPaddedTransformIfThenElseReductionOpaque(BasePaddingCompare):
    """Like TestPaddedTransformIfThenElseReduction, but with opaque blocks"""

    pad_value = tvm.testing.parameter(0)
    transformed_buffer = tvm.testing.parameter("B")

    def before(A: T.Buffer[(14, 32), "int32"]):
        B = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            B[i] = 0
            for k in T.serial(32):
                with T.block("block"):
                    B[i] = B[i] + A[i, k]

    def expected(A: T.Buffer[(14, 32), "int32"]):
        B = T.alloc_buffer([4, 4], "int32")
        for i, j in T.grid(4, 4):
            B[i, j] = T.if_then_else(i == 3 and 2 <= j, 0, 0, dtype="int32")
            for k in T.serial(32):
                with T.block("block"):
                    B[i, j] = T.if_then_else(
                        i == 3 and 2 <= j, 0, B[i, j] + A[i * 4 + j, k], dtype="int32"
                    )


class TestPaddedTransformPostProcIfRequiredDueToSideEffects(BasePaddingCompare):
    """Set the transformation padding in a post-processing block.

    Like TestPaddedTransformIfThenElse, but the block that produces B
    also has the effect of setting `C`.
    """

    pad_value = tvm.testing.parameter(0)
    transformed_buffer = tvm.testing.parameter("B")

    def before(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer(14, "int32")
        C = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]
                C[vi] = 0

    def expected(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer([4, 4], "int32")
        C = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi // 4, vi % 4] = A[vi]
                C[vi] = 0

        for i, j in T.grid(4, 4):
            with T.block("block_pad_B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.where(i == 3 and 2 <= j)
                B[vi, vj] = 0


class TestPaddedTransformOfInputCreatesAssumption(BasePaddingCompare):
    """Transformation of an input buffer places T.assume locally"""

    pad_value = tvm.testing.parameter(42)

    def before(A: T.Buffer[14, "int32"], B: T.Buffer[14, "int32"]):
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    def expected(A: T.Buffer[(4, 4), "int32"], B: T.Buffer[14, "int32"]):
        for i, j in T.grid(4, 4):
            with T.block("buffer_A_assumption"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.evaluate(T.assume(not (vi == 3 and 2 <= vj) or A[vi, vj] == 42))

        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi // 4, vi % 4]


class TestPaddedTransformNonConstantValue(tvm.testing.CompareBeforeAfter):
    """Allow an expression to specify the pad value.

    Like TestPaddedTransformIfThenElse, but the pad value depends on
    the indices.
    """

    @pytest.fixture
    def transform(self):
        def transform(mod):
            sch = tir.Schedule(mod)
            sch.transform_layout(
                "block",
                "B",
                lambda i: [i // 4, i % 4],
                pad_value=lambda i, j: i + j,
            )
            return sch.mod

        return transform

    def before(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    def expected(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer([4, 4], "int32")
        for i, j in T.grid(4, 4):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.if_then_else(
                    vi == 3 and 2 <= vj, vi + vj, A[vi * 4 + vj], dtype="int32"
                )


@pytest.mark.xfail(reason="Not yet implemented")
class TestPaddedTransformRepeatedBufferElement(tvm.testing.CompareBeforeAfter):
    """Allow an expression to specify the pad value.

    Like TestPaddedTransformOfInputCreatesAssumption, but the pad
    value depends on another portion of the buffer.  In this case, the
    padding at the end of A contains repeated elements from the
    beginning of A.
    """

    @pytest.fixture
    def transform(self):
        def transform(mod):
            sch = tir.Schedule(mod)

            A = sch.get(sch.get_block("block")).reads[0].buffer
            sch.transform_layout(
                "block",
                "A",
                lambda i: [i // 4, i % 4],
                pad_value=lambda i, j: A[(4 * i + j) % 14],
            )
            return sch.mod

        return transform

    def before(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    def expected(A: T.Buffer[(4, 4), "int32"]):
        for i, j in T.grid(4, 4):
            with T.block("buffer_A_assumption"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.evaluate(
                    T.assume(
                        not (vi == 3 and 2 <= vj)
                        or A[vi, vj] == A[((4 * vi + j) % 14) // 4, ((4 * vi + j) % 14) % 4]
                    )
                )

        B = T.alloc_buffer(14, "int32")
        for i in T.grid(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi // 4, vi % 4]


class TestPadValueMayNotReferenceOtherBuffer(tvm.testing.CompareBeforeAfter):
    """Allow an expression to specify the pad value.

    Like TestPaddedTransformRepeatedBufferElement, but the pad value depends on
    a different buffer, which is not allowed.
    """

    @pytest.fixture
    def transform(self):
        def transform(mod):
            sch = tir.Schedule(mod)

            A = sch.get(sch.get_block("block")).reads[0].buffer
            other = tir.decl_buffer(1, A.dtype, name="other")
            sch.transform_layout(
                "block",
                "A",
                lambda i: [i // 4, i % 4],
                pad_value=lambda i, j: other[0],
            )
            return sch.mod

        return transform

    def before(A: T.Buffer[14, "int32"]):
        B = T.alloc_buffer(14, "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    expected = tvm.tir.schedule.schedule.ScheduleError


class TestTransformLayoutWithVar(tvm.testing.CompareBeforeAfter):
    """Layout transform with dynamic parameter in transform"""

    @pytest.fixture
    def transform(self):
        def transform(mod):
            sch = tir.Schedule(mod)

            n = sch.mod["main"].params[1]

            sch.transform_layout(
                "block",
                "B",
                lambda i: [i // n, i % n],
                pad_value=0,
            )
            return sch.mod

        return transform

    def before(A: T.Buffer[16, "int32"], n: T.int32):
        B = T.alloc_buffer(16, "int32")
        for i in T.serial(16):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi]

    def expected(A: T.Buffer[16, "int32"], n: T.int32):
        B = T.alloc_buffer([(-16 % n + 16) // n, n], dtype="int32")
        for i, j in T.grid((-16 % n + 16) // n, n):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.if_then_else(
                    # Checks if the transform introduced padding
                    -16 % n != 0
                    and (
                        # If so, is vi in the last group (which may
                        # include padding).
                        (vj + vi * n) // n == 16 // n
                        # And is vj within the padding
                        and 16 % n <= (vj + vi * n) % n
                    ),
                    0,
                    A[vj + vi * n],
                    dtype="int32",
                )


class TestTransformWithAxisSeparators(BasePaddingCompare):
    """Axis separators may be specified in a transform"""

    index_map = tvm.testing.parameter(lambda i: [i // 4, tvm.tir.IndexMap.AXIS_SEPARATOR, i % 4])
    pad_value = tvm.testing.parameter(0)

    def before(a: T.handle):
        A = T.match_buffer(a, [14], "int32")
        for i in T.serial(14):
            with T.block("block"):
                vi = T.axis.remap("S", [i])
                A[vi] = 42

    def expected(a: T.handle):
        A = T.match_buffer(a, [4, 4], "int32", axis_separators=[1])
        for i, j in T.grid(4, 4):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                A[vi, vj] = T.if_then_else(vi == 3 and 2 <= vj, 0, 42, dtype="int32")


class TestTransformWithAxisSeparatorsOpaqueBlock(BasePaddingCompare):
    """Axis separators may be specified in a transform of opaque block"""

    index_map = tvm.testing.parameter(lambda i: [i // 4, tvm.tir.IndexMap.AXIS_SEPARATOR, i % 4])
    pad_value = tvm.testing.parameter(0)

    def before(a: T.handle):
        A = T.match_buffer(a, [14], "int32")
        for i in T.serial(14):
            with T.block("block"):
                A[i] = 42

    def expected(a: T.handle):
        A = T.match_buffer(a, [4, 4], "int32", axis_separators=[1])
        for i, j in T.grid(4, 4):
            with T.block("block"):
                A[i, j] = T.if_then_else(i == 3 and 2 <= j, 0, 42, dtype="int32")


if __name__ == "__main__":
    tvm.testing.main()
