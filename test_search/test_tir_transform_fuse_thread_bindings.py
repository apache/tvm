import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((256, 256), "float32"), B: T.Buffer((256, 256), "float32"), C: T.Buffer((256, 256), "float32"), D: T.Buffer((256, 256), "float32")):
        # with T.block("root"):
        local_c = T.alloc_buffer((256, 256))
        D_local = T.alloc_buffer((256, 256), scope="local")
        local_c_local = T.alloc_buffer((256, 256), scope="local")
        C_local = T.alloc_buffer((256, 256), scope="local")
        local_c_local_1 = T.alloc_buffer((256, 256), scope="local")
        A_local = T.alloc_buffer((256, 256), scope="local")
        B_local = T.alloc_buffer((256, 256), scope="local")
        for i_0_j_0_fused in T.thread_binding(32, thread="blockIdx.x"):
            for i_1_j_1_fused in T.thread_binding(2, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(32, thread="threadIdx.x"):
                    for k_0 in range(8):
                        for ax0_ax1_fused in range(128):
                            with T.block("A_local"):
                                v0 = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 16 * 4 + ax0_ax1_fused // 32)
                                v1 = T.axis.spatial(256, k_0 * 32 + ax0_ax1_fused % 32)
                                T.reads(A[v0, v1])
                                T.writes(A_local[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                A_local[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in range(256):
                            with T.block("B_local"):
                                v0 = T.axis.spatial(256, k_0 * 32 + ax0_ax1_fused // 8)
                                v1 = T.axis.spatial(256, i_1_j_1_fused * 128 + i_2_j_2_fused % 16 * 8 + ax0_ax1_fused % 8)
                                T.reads(B[v0, v1])
                                T.writes(B_local[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                B_local[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(16, 1, 4, 2, 4, 2):
                            with T.block("matmul1"):
                                vi = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 16 * 4 + i_3 * 4 + i_4)
                                vj = T.axis.spatial(256, i_1_j_1_fused * 128 + i_2_j_2_fused % 16 * 8 + j_3 * 2 + j_4)
                                vk = T.axis.reduce(256, k_0 * 32 + k_1 * 2 + k_2)
                                T.reads(A_local[vi, vk], B_local[vk, vj])
                                T.writes(local_c_local_1[vi, vj])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                with T.init():
                                    local_c_local_1[vi, vj] = T.float32(0.0)
                                local_c_local_1[vi, vj] = local_c_local_1[vi, vj] + A_local[vi, vk] * B_local[vk, vj]
                    for ax0, ax1 in T.grid(4, 8):
                        with T.block("local_c_local"):
                            v0 = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 16 * 4 + ax0)
                            v1 = T.axis.spatial(256, i_1_j_1_fused * 128 + i_2_j_2_fused % 16 * 8 + ax1)
                            T.reads(local_c_local_1[v0, v1])
                            T.writes(local_c[v0, v1])
                            local_c[v0, v1] = local_c_local_1[v0, v1]
        for i_0_j_0_fused in T.thread_binding(32, thread="blockIdx.x"):
            for i_1_j_1_fused in T.thread_binding(1, thread="vthread.x"):
                for i_2_j_2_fused in T.thread_binding(16, thread="threadIdx.x"):
                    for k_0_fused in T.serial(4, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 2]}):
                        for ax0_ax1_fused in range(64):
                            with T.block("local_c_local"):
                                v0 = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 2)
                                v1 = T.axis.spatial(256, k_0_fused * 64 + ax0_ax1_fused)
                                T.reads(local_c[v0, v1])
                                T.writes(local_c_local[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                local_c_local[v0, v1] = local_c[v0, v1]
                        for ax0_ax1_fused in range(8192):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(256, k_0_fused * 64 + ax0_ax1_fused // 128)
                                v1 = T.axis.spatial(256, i_2_j_2_fused % 2 * 128 + ax0_ax1_fused % 128)
                                T.reads(C[v0, v1])
                                T.writes(C_local[v0, v1])
                                T.block_attr({"meta_schedule.cooperative_fetch": 3})
                                C_local[v0, v1] = C[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(8, 1, 128, 8, 1, 1):
                            with T.block("matmul2"):
                                vi = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 2 + i_3 + i_4)
                                vj = T.axis.spatial(256, i_2_j_2_fused % 2 * 128 + j_3 + j_4)
                                vk = T.axis.reduce(256, k_0_fused * 64 + k_1 * 8 + k_2)
                                T.reads(local_c_local[vi, vk], C_local[vk, vj])
                                T.writes(D_local[vi, vj])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                with T.init():
                                    D_local[vi, vj] = T.float32(0.0)
                                D_local[vi, vj] = D_local[vi, vj] + local_c_local[vi, vk] * C_local[vk, vj]
                    for ax0, ax1 in T.grid(1, 128):
                        with T.block("D_local"):
                            v0 = T.axis.spatial(256, i_0_j_0_fused * 8 + i_2_j_2_fused // 2 + ax0)
                            v1 = T.axis.spatial(256, i_2_j_2_fused % 2 * 128 + ax1)
                            T.reads(D_local[v0, v1])
                            T.writes(D[v0, v1])
                            D[v0, v1] = D_local[v0, v1]


@T.prim_func
def element_wise_kernels_with_different_size(
    a: T.handle, b: T.handle, c: T.handle, d: T.handle
) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [256, 256])
    D = T.match_buffer(d, [256, 256])
    for i0 in T.thread_binding(0, 128, "blockIdx.x"):
        for j0 in T.thread_binding(0, 128, "threadIdx.x"):
            B[i0, j0] = A[i0, j0] * 2.0
    for i1 in T.thread_binding(0, 256, "blockIdx.x"):
        for j1 in T.thread_binding(0, 256, "threadIdx.x"):
            D[i1, j1] = C[i1, j1] + 1.0

@T.prim_func
def element_wise_thread_x(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i in T.thread_binding(0, 128, "blockIdx.x"):
        for j0_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.thread_binding(0, 4, "threadIdx.x"):
            for j1_1 in T.serial(0, 32):
                with T.block(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0

@T.prim_func
def element_wise_thread_x_different_dtype(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i in T.thread_binding(128, "blockIdx.x"):
        for j0_0 in T.thread_binding(4, "threadIdx.x"):
            for j0_1 in T.serial(0, 32):
                with T.block(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.thread_binding(T.int64(4), "threadIdx.x"):
            for j1_1 in T.serial(T.int64(32)):
                with T.block(""):
                    C[i, j1_0 * T.int64(32) + j1_1] = B[i, j1_0 * T.int64(32) + j1_1] + 1.0



if __name__ == "__main__":
    mod = tvm.tir.transform.FuseThreadBindings()(Module)
    mod = tvm.tir.transform.Simplify()(mod)
    print("[INFO]****************mod: ", mod)
    mod = tvm.IRModule.from_expr(element_wise_kernels_with_different_size.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.FuseThreadBindings()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod)
    
    mod = tvm.IRModule.from_expr(element_wise_thread_x.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.FuseThreadBindings()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod)
    
    mod = tvm.IRModule.from_expr(element_wise_thread_x_different_dtype.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.FuseThreadBindings()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod)