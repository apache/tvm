import tvm
from tvm.script import ir as I
from tvm.script import tir as T


@T.prim_func
def main(
    A: T.Buffer((1024, 1024), "float16"),
    B: T.Buffer((1024, 1024), "float16"),
    C: T.Buffer((1024, 1024), "float16"),
):
    T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
    # with T.block("root"):
    C_reindex_local = T.alloc_buffer((1, 1024, 1024), "float16", scope="local")
    A_reindex_shared = T.alloc_buffer((1, 1024, 1024), "float16", scope="shared")
    A_reindex_shared_local = T.alloc_buffer((1, 1024, 1024), "float16", scope="local")
    B_reindex_shared = T.alloc_buffer((1, 1024, 1024), "float16", scope="shared")
    B_reindex_shared_local = T.alloc_buffer((1, 1024, 1024), "float16", scope="local")
    for ax0_ax1_0_fused in T.thread_binding(32, thread="blockIdx.y"):
        for ax2_0 in T.thread_binding(32, thread="blockIdx.x"):
            for ax1_1 in T.thread_binding(1, thread="vthread.y"):
                for ax2_1 in T.thread_binding(2, thread="vthread.x"):
                    for ax1_2 in T.thread_binding(4, thread="threadIdx.y"):
                        for ax2_2 in T.thread_binding(4, thread="threadIdx.x"):
                            for ax1_3_init, ax2_3_init in T.grid(8, 4):
                                with T.block("B_init"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(
                                        1024,
                                        ax0_ax1_0_fused * 32 + ax1_1 * 32 + ax1_2 * 8 + ax1_3_init,
                                    )
                                    v2 = T.axis.spatial(
                                        1024, ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 4 + ax2_3_init
                                    )
                                    T.reads()
                                    T.writes(C_reindex_local[0, v1, v2])
                                    C_reindex_local[0, v1, v2] = T.float16(0)
                            for ax3_0 in range(16):
                                for ax0_ax1_ax2_fused_0 in range(16):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(
                                        4, thread="threadIdx.y"
                                    ):
                                        for ax0_ax1_ax2_fused_2 in T.thread_binding(
                                            4, thread="threadIdx.x"
                                        ):
                                            for ax0_ax1_ax2_fused_3 in T.serial(8):
                                                with T.block("A_reindex_shared"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial(
                                                        1024,
                                                        ax0_ax1_0_fused * 32
                                                        + (
                                                            ax0_ax1_ax2_fused_0 * 128
                                                            + ax0_ax1_ax2_fused_1 * 32
                                                            + ax0_ax1_ax2_fused_2 * 8
                                                            + ax0_ax1_ax2_fused_3
                                                        )
                                                        // 64,
                                                    )
                                                    v2 = T.axis.spatial(
                                                        1024,
                                                        ax3_0 * 64
                                                        + (
                                                            ax0_ax1_ax2_fused_0 * 128
                                                            + ax0_ax1_ax2_fused_1 * 32
                                                            + ax0_ax1_ax2_fused_2 * 8
                                                            + ax0_ax1_ax2_fused_3
                                                        )
                                                        % 64,
                                                    )
                                                    T.reads(A[v1, v2])
                                                    T.writes(A_reindex_shared[v0, v1, v2])
                                                    A_reindex_shared[v0, v1, v2] = A[v1, v2]
                                for ax0_ax1_ax2_fused_0 in range(16):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(
                                        4, thread="threadIdx.y"
                                    ):
                                        for ax0_ax1_ax2_fused_2 in T.thread_binding(
                                            4, thread="threadIdx.x"
                                        ):
                                            for ax0_ax1_ax2_fused_3 in range(8):
                                                with T.block("B_reindex_shared"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial(
                                                        1024,
                                                        ax3_0 * 64
                                                        + (
                                                            ax0_ax1_ax2_fused_0 * 128
                                                            + ax0_ax1_ax2_fused_1 * 32
                                                            + ax0_ax1_ax2_fused_2 * 8
                                                            + ax0_ax1_ax2_fused_3
                                                        )
                                                        // 32,
                                                    )
                                                    v2 = T.axis.spatial(
                                                        1024,
                                                        ax2_0 * 32
                                                        + (
                                                            ax0_ax1_ax2_fused_0 * 128
                                                            + ax0_ax1_ax2_fused_1 * 32
                                                            + ax0_ax1_ax2_fused_2 * 8
                                                            + ax0_ax1_ax2_fused_3
                                                        )
                                                        % 32,
                                                    )
                                                    T.reads(B[v2, v1])
                                                    T.writes(B_reindex_shared[v0, v1, v2])
                                                    B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                for ax3_1 in range(64):
                                    for ax0, ax1_ax2_fused_0 in T.grid(1, 1):
                                        for ax1_ax2_fused_1 in T.serial(8):
                                            with T.block("A_reindex_shared_local"):
                                                v0 = T.axis.spatial(1, ax0)
                                                v1 = T.axis.spatial(
                                                    1024,
                                                    ax0_ax1_0_fused * 32
                                                    + ax1_2 * 8
                                                    + ax1_ax2_fused_0 * 8
                                                    + ax1_ax2_fused_1,
                                                )
                                                v2 = T.axis.spatial(1024, ax3_0 * 64 + ax3_1)
                                                T.reads(A_reindex_shared[v0, v1, v2])
                                                T.writes(A_reindex_shared_local[v0, v1, v2])
                                                A_reindex_shared_local[
                                                    v0, v1, v2
                                                ] = A_reindex_shared[v0, v1, v2]
                                    for ax0, ax1_ax2_fused_0 in T.grid(1, 1):
                                        for ax1_ax2_fused_1 in T.vectorized(4):
                                            with T.block("B_reindex_shared_local"):
                                                v0 = T.axis.spatial(1, ax0)
                                                v1 = T.axis.spatial(1024, ax3_0 * 64 + ax3_1)
                                                v2 = T.axis.spatial(
                                                    1024,
                                                    ax2_0 * 32
                                                    + ax2_1 * 16
                                                    + ax2_2 * 4
                                                    + ax1_ax2_fused_0 * 4
                                                    + ax1_ax2_fused_1,
                                                )
                                                T.reads(B_reindex_shared[v0, v1, v2])
                                                T.writes(B_reindex_shared_local[v0, v1, v2])
                                                B_reindex_shared_local[
                                                    v0, v1, v2
                                                ] = B_reindex_shared[v0, v1, v2]
                                    for ax1_3, ax2_3 in T.grid(8, 4):
                                        with T.block("B_update"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial(
                                                1024,
                                                ax0_ax1_0_fused * 32
                                                + ax1_1 * 32
                                                + ax1_2 * 8
                                                + ax1_3,
                                            )
                                            v2 = T.axis.spatial(
                                                1024,
                                                ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 4 + ax2_3,
                                            )
                                            v3 = T.axis.reduce(1024, ax3_0 * 64 + ax3_1)
                                            T.reads(
                                                C_reindex_local[0, v1, v2],
                                                A_reindex_shared_local[0, v1, v3],
                                                B_reindex_shared_local[0, v3, v2],
                                            )
                                            T.writes(C_reindex_local[0, v1, v2])
                                            C_reindex_local[0, v1, v2] = (
                                                C_reindex_local[0, v1, v2]
                                                + A_reindex_shared_local[0, v1, v3]
                                                * B_reindex_shared_local[0, v3, v2]
                                            )
                            for ax0, ax1_ax2_fused_0 in T.grid(1, 4):
                                for ax1_ax2_fused_1 in T.serial(8):
                                    with T.block("C_reindex_local"):
                                        v0 = T.axis.spatial(1, ax0)
                                        v1 = T.axis.spatial(
                                            1024,
                                            ax0_ax1_0_fused * 32
                                            + ax1_2 * 8
                                            + (ax1_ax2_fused_0 * 8 + ax1_ax2_fused_1) // 4,
                                        )
                                        v2 = T.axis.spatial(
                                            1024,
                                            ax2_0 * 32
                                            + ax2_1 * 16
                                            + ax2_2 * 4
                                            + (ax1_ax2_fused_0 * 8 + ax1_ax2_fused_1) % 4,
                                        )
                                        T.reads(C_reindex_local[v0, v1, v2])
                                        T.writes(C[v1, v2])
                                        C[v1, v2] = C_reindex_local[v0, v1, v2]


mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
