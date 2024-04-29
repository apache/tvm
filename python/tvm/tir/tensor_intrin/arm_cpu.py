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
# pylint: disable=invalid-name,missing-function-docstring,unused-import
"""Intrinsics for ARM tensorization."""
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder.tir import prim_func as build_prim_func
from tvm.target.codegen import llvm_version_major

from .. import TensorIntrin
from .dot_product_common import (
    DP4A_S8S8S32_INTRIN,
    DP4A_S8U8S32_INTRIN,
    DP4A_U8S8S32_INTRIN,
    DP4A_U8U8U32_INTRIN,
)


# TODO(masahi): Parametrize the TVMScript description of dot product by
# shape and dtype, and share the common description with x86.


@T.prim_func
def neon_4x4_i8i8i32_desc(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])
        for i in T.serial(0, 4):
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def neon_4x4_i8i8i32_impl(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_int8 = A.vload([0], "int8x4")
        re_int32 = T.reinterpret(A_int8, dtype="int32")
        vec_ai32 = T.broadcast(re_int32, 2)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x8")

        vec_b = B.vload([0, 0], dtype="int8x16")

        # TODO(masahi): Remove duplication when inlined function call is supported
        vec_b_low = T.vectorlow(vec_b, dtype="int8x8")

        multiply_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_low,
            dtype="int16x8",
        )

        pairwise_reduction_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_low,
            dtype="int32x4",
        )

        vec_b_high = T.vectorhigh(vec_b, dtype="int8x8")

        multiply_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_high,
            dtype="int16x8",
        )

        pairwise_reduction_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_high,
            dtype="int32x4",
        )

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.addp.v4i32"),
            T.uint32(2),
            pairwise_reduction_low,
            pairwise_reduction_high,
            dtype="int32x4",
        )


def get_dotprod_intrin(in_dtype, out_dtype):
    if in_dtype == "uint8":
        instr = "udot.v4u32.v16u8"
    else:  # if in_dtype == "int8"
        instr = "sdot.v4i32.v16i8"

    in_dtype_x4 = f"{in_dtype}x4"
    out_dtype_x4 = f"{out_dtype}x4"
    in_dtype_x16 = f"{in_dtype}x16"

    @T.prim_func
    def dot_prod_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), dtype=in_dtype, offset_factor=1)
        B = T.match_buffer(b, (4, 4), dtype=in_dtype, offset_factor=1)
        C = T.match_buffer(c, (4,), dtype=out_dtype, offset_factor=1)
        with T.block("root"):
            T.reads(C[0:4], A[0:4], B[0:4, 0:4])
            T.writes(C[0:4])
            for i in T.serial(0, 4):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], dtype=out_dtype) * T.cast(
                            B[vi, vk], dtype=out_dtype
                        )

    @T.prim_func
    def dot_prod_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), dtype=in_dtype, offset_factor=1)
        B = T.match_buffer(b, (4, 4), dtype=in_dtype, offset_factor=1)
        C = T.match_buffer(c, (4,), dtype=out_dtype, offset_factor=1)
        with T.block("root"):
            T.reads(C[0:4], A[0:4], B[0:4, 0:4])
            T.writes(C[0:4])

            A_i8x4 = A.vload([0], in_dtype_x4)
            A_i32 = T.reinterpret(A_i8x4, dtype=out_dtype)
            vec_ai32 = T.broadcast(A_i32, 4)
            vec_a = T.reinterpret(vec_ai32, dtype=in_dtype_x16)

            vec_b = B.vload([0, 0], dtype=in_dtype_x16)

            vec_c = C.vload([0], dtype=out_dtype_x4)

            C[T.ramp(T.int32(0), 1, 4)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(f"llvm.aarch64.neon.{instr}"),
                T.uint32(3),
                vec_c,
                vec_a,
                vec_b,
                dtype=out_dtype_x4,
            )

    return dot_prod_desc, dot_prod_impl


def get_sme_transpose_interleave_2svlx2svl_intrin():
    """
    Transpose a matrix of size 2SVL x 2SVL (where 'SVL' is the Scalable Vector Length) using
    the Scalable Matrix Extension (SME).

    This is completed by loading rows of the input matrix into the accumulator tile,
    then storing the columns. The SME accumulator tile is divided into a series of sub-tiles
    which must be loaded to / stored from independently.

    Note: currently only supports the fp32 datatype.

    Example
    -------
    An example case for float32. In this instance the accumulator tile is divided into 4
    sub-tiles of size SVLxSVL numbered 0-3. We start by loading rows of A, each SVL in length,
    into each of the sub-tiles. In the diagram below, each load for a sub-tile is sequenced by
    a, b, ... till the tile is full.

    The columns of each sub-tile are then stored into A_t. Note that to perform a transpose,
    the contents of sub-tile 1 and 2 are stored in opposite locations - see the diagram
    below.

    A:                                  Accumulator tile:                     A_t:
                2SVL                                2SVL                               2SVL
         +----------------+                 +-----------------+                +-------------------+
         | --0a--  --1a-- |                 |                 |                | |  |     |  |     |
         | --0b--  --1b-- |                 |    0       1    |                | 0a 0b .. 2a 2b .. |
         |   ...     ...  | ld1w.horiz      |                 | st1w.vert      | |  |     |  |     |
    2SVL | --2a--  --3a-- |   ====>    2SVL |                 |   ====>   2SVL | |  |     |  |     |
         | --2a--  --3b-- |                 |    2       3    |                | 1a 1b .. 3a 3b .. |
         |   ...     ...  |                 |                 |                | |  |     |  |     |
         +----------------+                 +-----------------+                +-------------------+

    Returns
    -------
    intrin : TensorIntrin
        The SME TensorIntrin that can be used in tensorizing a schedule.

    """
    SVF = 4 * T.vscale()
    SVF2 = 2 * SVF

    @T.prim_func
    def desc(a: T.handle, a_t: T.handle) -> None:
        A = T.match_buffer(a, (SVF2, SVF2), dtype="float32", offset_factor=1)
        A_t = T.match_buffer(a_t, (SVF2, SVF2), dtype="float32", offset_factor=1)
        with T.block("root"):
            T.reads(A[0:SVF2, 0:SVF2])
            T.writes(A_t[0:SVF2, 0:SVF2])
            for k, m in T.grid(SVF2, SVF2):
                with T.block("transpose"):
                    v_m, v_k = T.axis.remap("SS", [m, k])
                    A_t[v_k, v_m] = A[v_m, v_k]

    def impl():
        # Accumulation sub-tile count. For fp32 it is 4
        sub_tile_count = 4

        with IRBuilder() as ib:
            with build_prim_func():
                a = T.arg("a", T.handle())
                a_t = T.arg("a_t", T.handle())

                A = T.match_buffer(
                    a, (SVF2, SVF2), "float32", offset_factor=1, strides=[T.int32(), 1]
                )
                A_t = T.match_buffer(
                    a_t,
                    (SVF2, SVF2),
                    "float32",
                    offset_factor=1,
                    strides=[T.int32(), 1],
                )

                # Disable predication
                ptrue = T.broadcast(T.IntImm("int1", 1), T.vscale() * 4)

                with T.block("root"):
                    T.reads(A[0:SVF2, 0:SVF2])
                    T.writes(A_t[0:SVF2, 0:SVF2])

                    # Load rows of the input matrix
                    with T.serial(0, SVF) as slice_idx:
                        for sub_tile_idx in range(0, sub_tile_count):
                            row_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            col_offset = SVF if sub_tile_idx % 2 else 0
                            offset = (slice_idx + row_offset) * A.strides[0] + col_offset

                            input_ptr = A.access_ptr("r", offset=offset)
                            sub_tile = T.int32(sub_tile_idx)
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "void",
                                    "llvm.aarch64.sme.ld1w.horiz",
                                    T.uint32(4),
                                    ptrue,
                                    input_ptr,
                                    sub_tile,
                                    slice_idx,
                                )
                            )

                    # Store columns to the ouptut matrix
                    with T.serial(0, SVF) as slice_idx:
                        for sub_tile_idx in range(0, sub_tile_count):
                            col_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            row_offset = SVF if sub_tile_idx % 2 else 0
                            offset = (slice_idx + row_offset) * A_t.strides[0] + col_offset

                            output_ptr = A_t.access_ptr("w", offset=offset)
                            sub_tile = T.int32(sub_tile_idx)
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "void",
                                    "llvm.aarch64.sme.st1w.vert",
                                    T.uint32(4),
                                    ptrue,
                                    output_ptr,
                                    sub_tile,
                                    slice_idx,
                                )
                            )

        return ib.get()

    return desc, impl()


def get_sme_gemm_interleaved_mopa_2svlx2svl_intrin(K):
    """
    Compute a GEMM of size 2SVL x 2SVL (where 'SVL' is the Scalable Vector Length using
    outer product operations from the Scalable Matrix Extension (SME).

    The inputs A and B are expected to be of size K x 2SVL and produce a result C of
    size 2SVL x 2SVL.

    The SME accumulator tile is divided into sub-tiles, each of which is utilized to
    calculate the outer-product using columns / rows of A and B respectively. For each
    sub-tile, elements in the first column of input matrix A (accessed sequentially due
    to being transpose-interleaved) and first row of input matrix B are used to calculate
    an outer-product. This is then accumulated with the result of performing an
    outer-product on the second column and row of A and B respectively. This process is
    repeated K times. Finally, the results of the accumulation are stored.

    Note: The input tensor 'A' must be transpose-interleaved.
    Note: Currently only supports the fp32 datatype.

    Example
    -------

    Diagram showing outer-product performed on each of the accumulator sub-tiles
    for the fp32 datatype:

                       SVL           SVL
                +----------------------------+
                |       l     |       h      | K
            K   +----------------------------+
         +---+  +----------------------------+
         |   |  |  0:            1:          |-+
         |   |  |  mopa(l, l)    mopa(l, h)  | |-+
       l |   |  |                            | | |
         |   |  |                            | | |
         |---|  |                            | | |
         |   |  |  2:            3:          | | |
       h |   |  |  mopa(h, l)    mopa(h, h)  | | |
         |   |  |                            | | |
         |   |  |                            | | |
         +---+  +----------------------------+ | |
                  +----------------------------+ |
                     +---------------------------+
                                    (accumulate K times)

    Pseudo code computing 2SVL x 2SVL GEMM for fp32 inputs:

    .. code-block:: c

        // Number of fp32 elements in a scalable vector
        int SVF = SVL / 32;

        // Reset the accumulator tile
        sme.zero();

        // Calculate outer products and accumulate
        for (k = 0; k < K; k++) {
            float32xSVF A_row_0 = A[k][0];
            float32xSVF A_row_1 = A[k][SVF];
            float32xSVF B_row_0 = B[k][0];
            float32xSVF B_row_1 = B[k][SVF];

            float32xSVFxSVF sub_tile_0 += sme.mopa(A_row_0, B_row_0);
            float32xSVFxSVF sub_tile_1 += sme.mopa(A_row_0, B_row_1);
            float32xSVFxSVF sub_tile_2 += sme.mopa(A_row_1, B_row_0);
            float32xSVFxSVF sub_tile_3 += sme.mopa(A_row_1, B_row_1);
        }

        // Store the results of accumulation
        for (i = 0; i < SVF; i++) {
            C[i][0] = sme.horiz(sub_tile_0[i]);
            C[i][0] = sme.horiz(sub_tile_0[i + SVF]);
            C[i + SVF][0] = sme.horiz(sub_tile_0[i]);
            C[i + SVF][0] = sme.horiz(sub_tile_0[i + SVF]);
        }

    Notes:
    - Recall that A has been transposed beforehand such that each column is now accessed
      by row.
    - 'sme.zero' resets the accumulator tile to contain all zero's.
    - 'sme.mopa' is the outer product and accumulate intrinsic.
    - 'sme.horiz' stores rows of an accumulator sub-tile to memory.

    Returns
    -------
    intrin : TensorIntrin
        The SME TensorIntrin that can be used in tensorizing a schedule.

    """
    SVF = 4 * T.vscale()
    SVF2 = 2 * SVF

    @T.prim_func
    def desc(a: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (K, SVF2), dtype="float32", offset_factor=1)
        B = T.match_buffer(b, (K, SVF2), dtype="float32", offset_factor=1)
        C = T.match_buffer(c, (SVF2, SVF2), dtype="float32", offset_factor=1)

        with T.block("root"):
            T.reads(C[0:SVF2, 0:SVF2], A[0:K, 0:SVF2], B[0:K, 0:SVF2])
            T.writes(C[0:SVF2, 0:SVF2])
            for m, n, k in T.grid(SVF2, SVF2, K):
                with T.block("gemm"):
                    v_m, v_n, v_k = T.axis.remap("SSR", [m, n, k])
                    C[v_m, v_n] += A[v_k, v_m] * B[v_k, v_n]

    def impl():
        # Accumulation sub-tile count. For fp32 it is 4
        sub_tile_count = 4

        with IRBuilder() as ib:
            with build_prim_func():
                a = T.arg("a", T.handle())
                b = T.arg("b", T.handle())
                c = T.arg("c", T.handle())

                A = T.match_buffer(a, (K, SVF2), "float32", offset_factor=1, strides=[T.int32(), 1])
                B = T.match_buffer(b, (K, SVF2), "float32", offset_factor=1, strides=[T.int32(), 1])
                C = T.match_buffer(
                    c, (SVF2, SVF2), "float32", offset_factor=1, strides=[T.int32(), 1]
                )

                ptrue = T.broadcast(T.IntImm("int1", 1), T.vscale() * 4)

                with T.block("root"):
                    T.reads(C[0:SVF2, 0:SVF2], A[0:K, 0:SVF2], B[0:K, 0:SVF2])
                    T.writes(C[0:SVF2, 0:SVF2])

                    # Iterate over the reduction axis applying outer product and accumulate
                    with T.serial(K) as k:
                        a_low = T.BufferLoad(A, [k, T.Ramp(0, 1, T.vscale() * 4)])
                        a_high = T.BufferLoad(A, [k, T.Ramp(SVF, 1, T.vscale() * 4)])
                        b_low = T.BufferLoad(B, [k, T.Ramp(0, 1, T.vscale() * 4)])
                        b_high = T.BufferLoad(B, [k, T.Ramp(SVF, 1, T.vscale() * 4)])

                        input_combinations = [
                            (a_low, b_low),
                            (a_low, b_high),
                            (a_high, b_low),
                            (a_high, b_high),
                        ]
                        for sub_tile_idx in range(0, sub_tile_count):
                            sub_tile = T.int32(sub_tile_idx)
                            input_1 = input_combinations[sub_tile_idx][0]
                            input_2 = input_combinations[sub_tile_idx][1]

                            T.evaluate(
                                T.call_llvm_intrin(
                                    "void",
                                    "llvm.aarch64.sme.mopa.nxv4f32",
                                    T.uint32(5),
                                    sub_tile,
                                    ptrue,
                                    ptrue,
                                    input_1,
                                    input_2,
                                )
                            )

                    # Store the accumulated tile results
                    with T.serial(SVF) as slice_idx:
                        for sub_tile_idx in range(sub_tile_count):
                            vert_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            horiz_offset = SVF if sub_tile_idx % 2 else 0
                            local_offset = (slice_idx + vert_offset) * C.strides[0] + horiz_offset
                            output_ptr = C.access_ptr("w", offset=local_offset, extent=SVF)

                            T.evaluate(
                                T.call_llvm_intrin(
                                    "void",
                                    "llvm.aarch64.sme.st1w.horiz",
                                    T.uint32(4),
                                    ptrue,
                                    output_ptr,
                                    T.int32(sub_tile_idx),
                                    T.int32(slice_idx),
                                )
                            )

            return ib.get()

    return desc, impl()


def get_sme_init_intrin():
    """
    Reset the entire matrix tile storage to 0.
    """
    SVF2 = 2 * 4 * T.vscale()

    @T.prim_func
    def desc(c: T.handle) -> None:
        C = T.match_buffer(c, (SVF2, SVF2), "float32", offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(C[0:SVF2, 0:SVF2])
            for m, n in T.grid(SVF2, SVF2):
                with T.block("init"):
                    v_m, v_n = T.axis.remap("SS", [m, n])
                    C[v_m, v_n] = T.float32(0)

    @T.prim_func
    def impl(c: T.handle) -> None:
        C = T.match_buffer(c, (SVF2, SVF2), "float32", offset_factor=1)
        with T.block("root"):
            T.reads()
            T.writes(C[0:SVF2, 0:SVF2])
            clear_all_tiles = T.int32(255)
            T.evaluate(
                T.call_llvm_intrin("void", "llvm.aarch64.sme.zero", T.uint32(1), clear_all_tiles)
            )

    return desc, impl


ARM_DOT_4x4_i8_NEON_INTRIN = "dot_4x4_i8i8s32_neon"
ARM_DOT_4x4_i8_SDOT_INTRIN = "dot_4x4_i8i8s32_sdot"
ARM_DOT_4x4_u8_UDOT_INTRIN = "dot_4x4_u8u8u32_udot"
ARM_DOT_4x4_u8_HDOT_INTRIN = "dot_4x4_u8u8i32_hdot"

TensorIntrin.register(ARM_DOT_4x4_i8_NEON_INTRIN, neon_4x4_i8i8i32_desc, neon_4x4_i8i8i32_impl)
TensorIntrin.register(ARM_DOT_4x4_i8_SDOT_INTRIN, *get_dotprod_intrin("int8", "int32"))
TensorIntrin.register(ARM_DOT_4x4_u8_UDOT_INTRIN, *get_dotprod_intrin("uint8", "uint32"))
TensorIntrin.register(ARM_DOT_4x4_u8_HDOT_INTRIN, *get_dotprod_intrin("uint8", "int32"))

ARM_SME_INIT = "sme_init"
ARM_SME_2SVLx2SVL_TRANSPOSE_INTERLEAVE = "sme_2svlx2svl_transpose_interleave"
ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA = "sme_2svlx2svl_gemm_interleaved_mopa"

# The following tensor intrinsics use LLVM intrinsics that are only available
# in versions of LLVM >= 15. Installations with older versions of LLVM will
# not be able to use them.
if llvm_version_major() >= 15:
    TensorIntrin.register(
        ARM_SME_2SVLx2SVL_TRANSPOSE_INTERLEAVE, *get_sme_transpose_interleave_2svlx2svl_intrin()
    )
    TensorIntrin.register(ARM_SME_INIT, *get_sme_init_intrin())
