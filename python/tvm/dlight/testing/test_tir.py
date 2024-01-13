import tvm
from tvm.script import ir as I
from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 150528), "float32"), B: T.Buffer((128, 150528), "float32"), C: T.Buffer((128, 128), "float32")):
    T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
    # with T.block("root"):
    C_local = T.alloc_buffer((128, 128), scope="local")
    A_shared = T.alloc_buffer((128, 150528), scope="shared")
    B_shared = T.alloc_buffer((128, 150528), scope="shared")
    for ax0_0_ax1_0_fused in T.thread_binding(256, thread="blockIdx.x"):
        for ax0_1_ax1_1_fused in T.thread_binding(64, thread="threadIdx.y"):
            for ax2_1_1_fused in T.thread_binding(2, thread="threadIdx.x"):
                for ax2_0 in range(392):
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.serial(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 8 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 384)
                                        v1 = T.axis.spatial(150528, ax2_0 * 384 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 8 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 384)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                    for ax0_ax1_fused_0 in range(6):
                        for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.y"):
                            for ax0_ax1_fused_2 in T.thread_binding(2, thread="threadIdx.x"):
                                for ax0_ax1_fused_3 in T.serial(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 8 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 384)
                                        v1 = T.axis.spatial(150528, ax2_0 * 384 + (ax0_ax1_fused_0 * 512 + ax0_ax1_fused_1 * 8 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 384)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                    for ax2_1_0 in range(192):
                        with T.block("B"):
                            v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8)
                            v1 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8)
                            v2 = T.axis.reduce(150528, ax2_0 * 384 + ax2_1_0 * 2 + ax2_1_1_fused)
                            T.reads(A_shared[v0, v2], B_shared[v1, v2])
                            T.writes(C_local[v0, v1])
                            with T.init():
                                C_local[v0, v1] = T.float32(0)
                            C_local[v0, v1] = C_local[v0, v1] + A_shared[v0, v2] * B_shared[v1, v2]
            with T.block("C_local"):
                v0 = T.axis.spatial(128, ax0_0_ax1_0_fused // 16 * 8 + ax0_1_ax1_1_fused // 8)
                v1 = T.axis.spatial(128, ax0_0_ax1_0_fused % 16 * 8 + ax0_1_ax1_1_fused % 8)
                T.reads(C_local[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_local[v0, v1]


mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())