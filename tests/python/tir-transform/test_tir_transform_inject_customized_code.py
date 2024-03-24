# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group

@T.prim_func
def main_opt_m_16(A: T.handle("float16", "global"), B: T.handle("int8", "global"), Scale: T.handle("float16", "global"), Bias: T.handle("float16", "global"), E: T.handle("float16", "global"), m: T.int32):
    T.func_attr({"dequantize_info": {"B_decode": {"decode_block": "B_decode", "fast_decoding": T.bool(True), "group_size": 128, "source_format": {"bits": 4, "format": "uint"}, "storage_dtype": "int8", "target_format": "float16", "with_scaling": T.bool(True), "with_zeros": T.bool(False), "zeros_type": "original"}}, "dlight.tensorcore_prenormlized": T.bool(True), "opt_shapes": {"m": 16}, "tir.noalias": T.bool(True)})
    E_1 = T.decl_buffer((m, 768), "float16", data=E)
    Bias_1 = T.decl_buffer((768,), "float16", data=Bias)
    Scale_1 = T.decl_buffer((768, 6), "float16", data=Scale)
    B_1 = T.decl_buffer((768, 384), "int8", data=B)
    A_1 = T.decl_buffer((m, 768), "float16", data=A)
    with T.block("root"):
        T.reads()
        T.writes()
        A_reindex_pad_shared = T.alloc_buffer((1, (m + 15) // 16 * 16, 768), "float16", scope="shared")
        B_decode_reindex_shared = T.alloc_buffer((1, 768, 768), "float16", scope="shared")
        B_decode_reindex_local = T.alloc_buffer((1, 768, 768), "float16", scope="local")
        B_local = T.alloc_buffer((768, 384), "int8", scope="local")
        B_shared = T.alloc_buffer((768, 384), "int8", scope="shared")
        A_reindex_pad_shared_warp = T.alloc_buffer((1, (m + 15) // 16, 48, 32, 8), "float16", scope="warp")
        B_decode_reindex_shared_warp = T.alloc_buffer((1, 48, 48, 32, 8), "float16", scope="warp")
        C_reindex_pad_shared = T.alloc_buffer((1, (m + 15) // 16, 48, 16, 16), "float16", scope="shared")
        C_reindex_pad_shared_warp = T.alloc_buffer((1, (m + 15) // 16, 48, 32, 8), "float16", scope="warp")
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((m + 15) // 16, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(12, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(1, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(4, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(1, 1):
                                with T.block("C_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial((m + 15) // 16, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(48, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("C_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in T.serial(6, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2, 3], "software_pipeline_stage": [0, 0, 1, 1]}):
                                for ax0_ax1_ax2_fused_0 in T.unroll(2):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_2 in T.thread_binding(4, thread="threadIdx.z", annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial((m + 15) // 16 * 16, ax1_0_0_ax2_0_0_fused * 16 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 1024 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 128)
                                                        v2 = T.axis.spatial(768, ax3_0_0 * 128 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 1024 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 128)
                                                        T.reads(A_1[v1, v2])
                                                        T.writes(A_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, A_1[v1, v2], T.float16(0))
                                for ax0_ax1_fused_0 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                    for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.z"):
                                        for ax0_ax1_fused_2 in T.thread_binding(1, thread="threadIdx.y"):
                                            for ax0_ax1_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_fused_4 in T.vectorized(16):
                                                    with T.block("B_shared"):
                                                        v0 = T.axis.spatial(768, ax1_0_1_ax2_0_1_fused * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) // 64)
                                                        v1 = T.axis.spatial(384, ax3_0_0 * 64 + (ax0_ax1_fused_0 * 2048 + ax0_ax1_fused_1 * 512 + ax0_ax1_fused_2 * 512 + ax0_ax1_fused_3 * 16 + ax0_ax1_fused_4) % 64)
                                                        T.reads(B_1[v0, v1])
                                                        T.writes(B_shared[v0, v1])
                                                        B_shared[v0, v1] = B_1[v0, v1]
                                for ax0_1, ax1_ax2_0_fused_0 in T.grid(1, 8):
                                    for ax1_ax2_0_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
                                        for ax1_ax2_0_fused_2 in T.thread_binding(4, thread="threadIdx.z"):
                                            for ax1_ax2_0_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax2_1 in range(1):
                                                    for ax0_2 in range(1):
                                                        for ax1 in T.vectorized(4):
                                                            with T.block("B_local"):
                                                                v0 = T.axis.spatial(768, ax1_0_1_ax2_0_1_fused * 64 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) // 16 + ax0_2)
                                                                v1 = T.axis.spatial(384, ax3_0_0 * 64 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) % 16 * 4 + ax1)
                                                                T.reads(B_shared[v0, v1])
                                                                T.writes(B_local[v0, v1])
                                                                B_local[v0, v1] = B_shared[v0, v1]
                                                    for ax0_2, ax1 in T.grid(1, 1):
                                                        with T.block("B_decode_reindex_local_o"):
                                                            v0_o = T.axis.spatial(1, ax0_2)
                                                            v1_o = T.axis.spatial(768, ax1_0_1_ax2_0_1_fused * 64 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) // 16 + ax1)
                                                            v2_o = T.axis.spatial(96, ax3_0_0 * 16 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) % 16)
                                                            T.reads(B_local[v1_o, v2_o * 4:v2_o * 4 + 4], Scale_1[v1_o, v2_o // 16])
                                                            T.writes(B_decode_reindex_local[v0_o, v1_o, v2_o * 8:v2_o * 8 + 8])
                                                            Compressed = T.match_buffer(B_local[v1_o, v2_o * 4:v2_o * 4 + 4], (4,), "int8", scope="local")
                                                            Decompressed = T.match_buffer(B_decode_reindex_local[v0_o, v1_o, v2_o * 8:v2_o * 8 + 8], (8,), "float16", scope="local")
                                                            Scale_2 = T.match_buffer(Scale_1[v1_o, v2_o // 16], (1,), "float16", strides=("Scale_s0",), offset_factor=1)
                                                            T.call_extern("handle", "decode_i4u_to_f16_scale", Compressed.data, Decompressed.data, T.tvm_access_ptr(T.type_annotation("float16"), Scale_2.data, Scale_2.elem_offset, Scale_2.strides[0], 1), 8)
                                                    for ax2_2 in T.vectorized(8):
                                                        with T.block("B_decode_reindex_shared"):
                                                            v0 = T.axis.spatial(1, ax0_1)
                                                            v1 = T.axis.spatial(768, ax1_0_1_ax2_0_1_fused * 64 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) // 16)
                                                            v2 = T.axis.spatial(768, ax3_0_0 * 128 + (ax1_ax2_0_fused_0 * 128 + ax1_ax2_0_fused_1 * 128 + ax1_ax2_0_fused_2 * 32 + ax1_ax2_0_fused_3) % 16 * 8 + ax2_1 * 8 + ax2_2)
                                                            T.reads(B_decode_reindex_local[v0, v1, v2])
                                                            T.writes(B_decode_reindex_shared[v0, v1, v2])
                                                            T.block_attr({"permuted_layout": 1})
                                                            B_decode_reindex_shared[v0, v1, v2] = B_decode_reindex_local[v0, v1, v2]
                                for ax3_0_1 in range(8):
                                    for ax0_1, ax1_0, ax2_0 in T.grid(1, 1, 1):
                                        with T.block("A_reindex_pad_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, ax0_1)
                                            v1_o = T.axis.spatial((m + 15) // 16, ax1_0_0_ax2_0_0_fused + ax1_0)
                                            v2_o = T.axis.spatial(48, ax3_0_0 * 8 + ax3_0_1 + ax2_0)
                                            T.reads(A_reindex_pad_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(A_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_pad_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_1, ax1_0, ax2_0 in T.grid(1, 1, 1):
                                        with T.block("B_decode_reindex_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, ax0_1)
                                            v1_o = T.axis.spatial(48, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 + ax1_0)
                                            v2_o = T.axis.spatial(48, ax3_0_0 * 8 + ax3_0_1 + ax2_0)
                                            T.reads(B_decode_reindex_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_decode_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(B_decode_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_decode_reindex_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * 8 * (tx // 16) + shared.strides[0] * (tx % 8) + 8 * (tx % 16 // 8))
                                    for ax1_0_3, ax2_0_3 in T.grid(1, 1):
                                        with T.block("C_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial((m + 15) // 16, ax1_0_0_ax2_0_0_fused + ax1_0_2 + ax1_0_3)
                                            v2_o = T.axis.spatial(48, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 + ax2_0_3)
                                            v3_o = T.axis.reduce(48, ax3_0_0 * 8 + ax3_0_1)
                                            T.reads(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], B_decode_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8])
                                            T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("C_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], B_decode_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8])
                                                T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_2 = T.match_buffer(A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_2 = T.match_buffer(B_decode_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C = T.match_buffer(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_2.data, A_2.elem_offset + tx * 8, B_2.data, B_2.elem_offset + tx * 8, C.data, C.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_2.data, A_2.elem_offset + tx * 8, B_2.data, B_2.elem_offset + tx * 8 + 4, C.data, C.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_0, ax1_0 in T.grid(1, 1):
                                with T.block("C_reindex_pad_shared_warp_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial((m + 15) // 16, ax1_0_0_ax2_0_0_fused)
                                    v2_o = T.axis.spatial(48, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2)
                                    v3_o, v4_o = T.axis.remap("SS", [ax0_0, ax1_0])
                                    T.reads(C_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_pad_shared[v0_o, v1_o, v2_o, 0:16, 0:16])
                                    C_warp = T.match_buffer(C_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C = T.match_buffer(C_reindex_pad_shared[v0_o, v1_o, v2_o, 0:16, 0:16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C.strides[0])
                            for ax0_ax1_ax2_ax3_ax4_fused_0 in T.unroll(1, annotations={"pragma_unroll_explicit": 0}):
                                for ax0_ax1_ax2_ax3_ax4_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                    for ax0_ax1_ax2_ax3_ax4_fused_2 in T.vectorized(8):
                                        with T.block("C_reindex_pad_shared"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial((m + 15) // 16, ax1_0_0_ax2_0_0_fused)
                                            v2 = T.axis.spatial(48, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2)
                                            v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 256 + ax0_ax1_ax2_ax3_ax4_fused_1 * 8 + ax0_ax1_ax2_ax3_ax4_fused_2) // 16)
                                            v4 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 256 + ax0_ax1_ax2_ax3_ax4_fused_1 * 8 + ax0_ax1_ax2_ax3_ax4_fused_2) % 16)
                                            T.reads(C_reindex_pad_shared[v0, v1, v2, v3, v4], Bias_1[v4 + v2 * 16])
                                            T.writes(E_1[v3 + v1 * 16, v4 + v2 * 16])
                                            if v1 * 16 + v3 < m:
                                                E_1[v3 + v1 * 16, v4 + v2 * 16] = C_reindex_pad_shared[v0, v1, v2, v3, v4] + Bias_1[v4 + v2 * 16]

mod = main_opt_m_16
sch = tvm.tir.Schedule(mod, debug_mask="all")
block_init = sch.get_block("C_o_init")
i = sch.get_loops(block_init)[-2]
b = sch.get_loops(block_init)[-3]
# sch.annotate(i, "inject_customized_code_prepend", "hello")
sch.annotate(b, "inject_customized_code_postpend", "hello")
with tvm.transform.PassContext(
            config={"tir.use_async_copy": True}
        ):
    dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")

print(dense_relu_0_rt_mod.imported_modules[0].get_source())
