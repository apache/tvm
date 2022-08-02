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

from tvm.script import tir as T

w_spit__b = "llvm.hexagon.S2.vsplatrb"  # Q6_R_vsplatb_R
v_spit__w = "llvm.hexagon.V6.lvsplatw.128B"
v_rmpy__uv_uw_acc = "llvm.hexagon.V6.vrmpyub.acc.128B"
v_rmpy__uv_uw = "llvm.hexagon.V6.vrmpyub.128B"
v_sub = "llvm.hexagon.V6.vsubw.128B"

def get_mm_uint8_intrin(in_m, in_n, in_k):
    blocks = in_k // 32
    unrolled_rows = in_m // 16

    @T.prim_func
    def mm_uint8_intrinsic(a: T.handle, b: T.handle, c: T.handle, offsets: T.handle):
        A = T.match_buffer(a, [T.cast(in_n, dtype="int32") * T.cast(in_m, dtype="int32")], dtype="uint8")
        B = T.match_buffer(b, [T.cast(in_m, dtype="int32") * T.cast(in_k, dtype="int32")], dtype="uint8")
        C = T.match_buffer(c, [T.cast(in_n, dtype="int32") * T.cast(in_k, dtype="int32")], dtype="int32")
        OFFSETS = T.match_buffer(offsets, [2], dtype="uint8")
        with T.block("root"): 
            T.reads(A[0: T.cast(in_n, dtype="int32") * T.cast(in_m, dtype="int32")], B[0: T.cast(in_m, dtype="int32") * T.cast(in_k, dtype="int32")], OFFSETS[0:2])
            T.writes(C[0: T.cast(in_n, dtype="int32") * T.cast(in_k, dtype="int32")])
            for i in T.serial(in_n):
                for s in T.serial(blocks):
                    C[T.ramp(((s * 32) + (i * T.cast(in_k, dtype="int32"))), 1, 32)] = T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.lvsplatw.128B"), T.uint32(1), (( T.cast(OFFSETS[0], dtype="int32") * T.cast(OFFSETS[1], dtype="int32")) * T.cast(in_m, dtype="int32")), dtype="int32x32")
                for blok, ro  in T.grid(blocks, unrolled_rows):
                    b_offset =  T.cast(OFFSETS[1], dtype="int32")
                    a_offset =  T.cast(OFFSETS[0], dtype="int32")
                    out_index = blok * 32 + i * T.cast(in_k, dtype="int32")
                    
                    B_index_unrolled = blok * 128 + (ro * 16 * T.cast(in_k, dtype="int32"))
                    B_index_unrolled_2 = blok * 128 + (ro * 16 + 4) * T.cast(in_k, dtype="int32")
                    B_index_unrolled_3 = blok * 128 + (ro * 16 + 8) * T.cast(in_k, dtype="int32")
                    B_index_unrolled_4 = blok * 128 + (ro * 16 + 12) * T.cast(in_k, dtype="int32")
                    
                    A_index_unrolled = ro * 16 + i * T.cast(in_m, dtype="int32")
                    A_index_unrolled_2 = A_index_unrolled + 4
                    A_index_unrolled_3 = A_index_unrolled + 8
                    A_index_unrolled_4 = A_index_unrolled + 12
                    
                    a_b_vrmpy_accumulation_unrolled = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"), # instruction
                        T.uint32(3), # number of inputs
                        C[T.ramp(out_index, 1, 32)], # accumulation location
                        T.reinterpret(B[T.ramp(B_index_unrolled, 1, 128)], dtype = "int32x32"), # 32 4 byte inputs (Vu) to vrmpy 
                        T.reinterpret(A[T.ramp(A_index_unrolled, 1, 4)], dtype = "int32"), # 4 byte input (Rt) to vrmpy
                        dtype = "int32x32" # output datatype
                    )

                    a_b_vrmpy_accumulation_unrolled_1 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_vrmpy_accumulation_unrolled,
                        T.reinterpret(B[T.ramp(B_index_unrolled_2, 1, 128)], dtype = "int32x32"),
                        T.reinterpret(A[T.ramp(A_index_unrolled_2, 1, 4)], dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_vrmpy_accumulation_unrolled_2 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_vrmpy_accumulation_unrolled_1,
                        T.reinterpret(B[T.ramp(B_index_unrolled_3, 1, 128)], dtype = "int32x32"),
                        T.reinterpret(A[T.ramp(A_index_unrolled_3, 1, 4)], dtype = "int32"),
                        dtype = "int32x32"
                    )

                    la_b_vrmpy_accumulation_unrolled_3 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_vrmpy_accumulation_unrolled_2,
                        T.reinterpret(B[T.ramp(B_index_unrolled_4, 1, 128)], dtype = "int32x32"),
                        T.reinterpret(A[T.ramp(A_index_unrolled_4, 1, 4)], dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_b = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.128B"),
                        T.uint32(2),
                        T.reinterpret(B[T.ramp(B_index_unrolled, 1, 128)], dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"), T.uint32(1), a_offset, dtype = "int32"), 
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_a = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_b,
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.lvsplatw.128B"),T.uint32(1),T.reinterpret(A[T.ramp(A_index_unrolled, 1, 4)], dtype = "int32"),dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),b_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_b1 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_a,
                        T.reinterpret(B[T.ramp(B_index_unrolled_2, 1, 128)], dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),a_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_a1 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_b1,
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.lvsplatw.128B"),T.uint32(1),T.reinterpret(A[T.ramp(A_index_unrolled_2, 1, 4)], dtype = "int32"),dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),b_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_b2 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_a1,
                        T.reinterpret(B[T.ramp(B_index_unrolled_3, 1, 128)], dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),a_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_a2 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_b2,
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.lvsplatw.128B"),T.uint32(1),T.reinterpret(A[T.ramp(A_index_unrolled_3, 1, 4)], dtype = "int32"),dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),b_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_b3 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                        T.uint32(3),
                        a_b_offsets_vrmpy_accumulation_unrolled_a2,
                        T.reinterpret(B[T.ramp(B_index_unrolled_4, 1, 128)], dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"),T.uint32(1),a_offset,dtype = "int32"),
                        dtype = "int32x32"
                    )

                    a_b_offsets_vrmpy_accumulation_unrolled_a3 = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"), 
                        T.uint32(3), 
                        a_b_offsets_vrmpy_accumulation_unrolled_b3,
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.lvsplatw.128B"), T.uint32(1), T.reinterpret(A[T.ramp(A_index_unrolled_4, 1, 4)], dtype = "int32"), dtype = "int32x32"),
                        T.call_llvm_intrin(T.llvm_lookup_intrinsic_id("llvm.hexagon.S2.vsplatrb"), T.uint32(1), b_offset, dtype = "int32"),
                        dtype = "int32x32"
                    )

                    C[T.ramp(out_index, 1, 32)] = T.call_llvm_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vsubw.128B"), 
                        T.uint32(2), 
                        la_b_vrmpy_accumulation_unrolled_3,
                        a_b_offsets_vrmpy_accumulation_unrolled_a3,
                        dtype = "int32x32"
                    )


    @T.prim_func
    def mmul_desc(a: T.handle, b: T.handle, c: T.handle, offsets: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [in_n, in_m], dtype="uint8")
        B = T.match_buffer(b, [in_m, in_k], dtype="uint8")
        C = T.match_buffer(c, [in_n, in_k], dtype="int32")
        OFFSETS = T.match_buffer(offsets, [2], dtype="uint8")
        # body
        with T.block("root"): 
            for i0, i1, i2 in T.grid(in_m, in_n, in_k):
                with T.block("C"):
                    y, x, j = T.axis.remap("SSR", [i0, i1, i2])
                    C[y, x] = C[y, x] + T.cast(A[y, j] - OFFSETS[0], "int32") * T.cast(B[j, x] - OFFSETS[1], "int32")

    return mmul_desc, mm_uint8_intrinsic


