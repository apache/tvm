import tvm
import numpy as np
from tvm.script import tir as T


@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer[(1012, 1014), "float32"],
        B: T.Buffer[(1014, 1017), "float32"],
        Y: T.Buffer[(1012, 1017), "float32"],
    ):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        Y_reindex_local = T.alloc_buffer([1024, 1024], dtype="float32", scope="local")
        A_reindex_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
        B_reindex_shared = T.alloc_buffer([1024, 1024], dtype="float32", scope="shared")
        A_reindex_shared_local = T.alloc_buffer([1024, 1024], dtype="float32", scope="local")
        B_reindex_shared_local = T.alloc_buffer([1024, 1024], dtype="float32", scope="local")
        for ax0_0_ax1_0_fused in T.thread_binding(
            128,
            thread="blockIdx.x",
            annotations={"pragma_auto_unroll_max_step": 1024, "pragma_unroll_explicit": 1},
        ):
            for ax0_1_ax1_1_fused in T.thread_binding(4, thread="vthread.x"):
                for ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused in T.thread_binding(
                    64, thread="threadIdx.x"
                ):
                    for ax0_3_init, ax1_3_init, ax0_4_init, ax1_4_init in T.grid(4, 4, 2, 1):
                        with T.block("Y_init"):
                            v0 = T.axis.spatial(
                                1024,
                                ax0_0_ax1_0_fused // 8 * 64
                                + ax0_1_ax1_1_fused // 2 * 32
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused // 32 * 16
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 2 * 8
                                + ax0_3_init * 2
                                + ax0_4_init,
                            )
                            v1 = T.axis.spatial(
                                1024,
                                ax1_4_init
                                + ax0_0_ax1_0_fused % 8 * 128
                                + ax0_1_ax1_1_fused % 2 * 64
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 32 // 2 * 4
                                + ax1_3_init,
                            )
                            T.reads()
                            T.writes(Y_reindex_local[v0, v1])
                            T.block_attr(
                                {
                                    "meta_schedule.thread_extent_high_inclusive": 1024,
                                    "meta_schedule.thread_extent_low_inclusive": 32,
                                    "meta_schedule.tiling_structure": "SSSRRSRS",
                                }
                            )
                            Y_reindex_local[v0, v1] = T.float32(0)
                    for ax2_0_fused in T.serial(
                        256,
                        annotations={
                            "software_pipeline_async_stages": [0, 1],
                            "software_pipeline_order": [0, 1, 3, 2, 4],
                            "software_pipeline_stage": [0, 0, 2, 3, 3],
                        },
                    ):
                        for ax0_ax1_fused_0 in T.serial(4):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                with T.block("A_reindex_shared"):
                                    v0 = T.axis.spatial(
                                        1024,
                                        ax0_0_ax1_0_fused // 8 * 64
                                        + (ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1) // 4,
                                    )
                                    v1 = T.axis.spatial(
                                        1024,
                                        ax2_0_fused * 4
                                        + (ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1) % 4,
                                    )
                                    T.reads(A[v0, v1])
                                    T.writes(
                                        A_reindex_shared[
                                            v1,
                                            v0 // 32 * 32
                                            + v0 % 8 // 4 * 16
                                            + v0 % 32 // 8 * 4
                                            + v0 % 4,
                                        ]
                                    )
                                    A_reindex_shared[
                                        v1,
                                        v0 // 32 * 32
                                        + v0 % 8 // 4 * 16
                                        + v0 % 32 // 8 * 4
                                        + v0 % 4,
                                    ] = T.if_then_else(
                                        v0 < 1012 and v1 < 1014,
                                        A[v0, v1],
                                        T.float32(0),
                                        dtype="float32",
                                    )
                        for ax0_ax1_fused_0 in T.serial(8):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                with T.block("B_reindex_shared"):
                                    v0 = T.axis.spatial(
                                        1024,
                                        ax2_0_fused * 4
                                        + (ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1) // 128,
                                    )
                                    v1 = T.axis.spatial(
                                        1024,
                                        ax0_0_ax1_0_fused % 8 * 128
                                        + (ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1) % 128,
                                    )
                                    T.reads(B[v0, v1])
                                    T.writes(
                                        B_reindex_shared[
                                            v0,
                                            v1 // 64 * 64
                                            + v1 % 8 // 4 * 32
                                            + v1 % 64 // 8 * 4
                                            + v1 % 4,
                                        ]
                                    )
                                    B_reindex_shared[
                                        v0,
                                        v1 // 64 * 64
                                        + v1 % 8 // 4 * 32
                                        + v1 % 64 // 8 * 4
                                        + v1 % 4,
                                    ] = T.if_then_else(
                                        v0 < 1014 and v1 < 1017,
                                        B[v0, v1],
                                        T.float32(0),
                                        dtype="float32",
                                    )
                        for ax2_1_fused in T.unroll(
                            4,
                            annotations={
                                "software_pipeline_order": [0, 1, 2],
                                "software_pipeline_stage": [0, 0, 1],
                            },
                        ):
                            for ax0_ax1_fused_0 in T.unroll(2):
                                for ax0_ax1_fused_1 in T.vectorized(4):
                                    with T.block("A_reindex_shared_local"):
                                        v0 = T.axis.spatial(1024, ax2_0_fused * 4 + ax2_1_fused)
                                        v1 = T.axis.spatial(
                                            1024,
                                            ax0_0_ax1_0_fused // 8 * 64
                                            + ax0_1_ax1_1_fused // 2 * 32
                                            + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused
                                            // 32
                                            * 16
                                            + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 2 * 8
                                            + ax0_ax1_fused_0 * 4
                                            + ax0_ax1_fused_1,
                                        )
                                        T.reads(
                                            A_reindex_shared[
                                                v0,
                                                v1 // 32 * 32
                                                + v1 % 8 // 4 * 16
                                                + v1 % 32 // 8 * 4
                                                + v1 % 4,
                                            ]
                                        )
                                        T.writes(A_reindex_shared_local[v0, v1])
                                        A_reindex_shared_local[v0, v1] = A_reindex_shared[
                                            v0,
                                            v1 // 32 * 32
                                            + v1 % 8 // 4 * 16
                                            + v1 % 32 // 8 * 4
                                            + v1 % 4,
                                        ]
                            for ax0_ax1_fused_0 in T.unroll(2):
                                for ax0_ax1_fused_1 in T.vectorized(2):
                                    with T.block("B_reindex_shared_local"):
                                        v0 = T.axis.spatial(1024, ax2_0_fused * 4 + ax2_1_fused)
                                        v1 = T.axis.spatial(
                                            1024,
                                            ax0_0_ax1_0_fused % 8 * 128
                                            + ax0_1_ax1_1_fused % 2 * 64
                                            + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused
                                            % 32
                                            // 2
                                            * 4
                                            + ax0_ax1_fused_0 * 2
                                            + ax0_ax1_fused_1,
                                        )
                                        T.reads(
                                            B_reindex_shared[
                                                v0,
                                                v1 // 64 * 64
                                                + v1 % 8 // 4 * 32
                                                + v1 % 64 // 8 * 4
                                                + v1 % 4,
                                            ]
                                        )
                                        T.writes(B_reindex_shared_local[v0, v1])
                                        B_reindex_shared_local[v0, v1] = B_reindex_shared[
                                            v0,
                                            v1 // 64 * 64
                                            + v1 % 8 // 4 * 32
                                            + v1 % 64 // 8 * 4
                                            + v1 % 4,
                                        ]
                            for ax0_3, ax1_3, ax2_2, ax0_4, ax1_4 in T.grid(4, 4, 1, 2, 1):
                                with T.block("Y_update"):
                                    v0 = T.axis.spatial(
                                        1024,
                                        ax0_0_ax1_0_fused // 8 * 64
                                        + ax0_1_ax1_1_fused // 2 * 32
                                        + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused // 32 * 16
                                        + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 2 * 8
                                        + ax0_3 * 2
                                        + ax0_4,
                                    )
                                    v1 = T.axis.spatial(
                                        1024,
                                        ax1_4
                                        + ax0_0_ax1_0_fused % 8 * 128
                                        + ax0_1_ax1_1_fused % 2 * 64
                                        + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused
                                        % 32
                                        // 2
                                        * 4
                                        + ax1_3,
                                    )
                                    v2 = T.axis.reduce(1024, ax2_0_fused * 4 + ax2_1_fused + ax2_2)
                                    T.reads(
                                        Y_reindex_local[v0, v1],
                                        A_reindex_shared_local[v2, v0],
                                        B_reindex_shared_local[v2, v1],
                                    )
                                    T.writes(Y_reindex_local[v0, v1])
                                    T.block_attr(
                                        {
                                            "meta_schedule.thread_extent_high_inclusive": 1024,
                                            "meta_schedule.thread_extent_low_inclusive": 32,
                                            "meta_schedule.tiling_structure": "SSSRRSRS",
                                        }
                                    )
                                    Y_reindex_local[v0, v1] = (
                                        Y_reindex_local[v0, v1]
                                        + A_reindex_shared_local[v2, v0]
                                        * B_reindex_shared_local[v2, v1]
                                    )
                    for ax0, ax1 in T.grid(8, 4):
                        with T.block("Y_reindex_local"):
                            T.where(
                                ax0_0_ax1_0_fused // 8 * 64
                                + ax0_1_ax1_1_fused // 2 * 32
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused // 32 * 16
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 2 * 8
                                + ax0
                                < 1012
                                and ax0_0_ax1_0_fused % 8 * 128
                                + ax0_1_ax1_1_fused % 2 * 64
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 32 // 2 * 4
                                + ax1
                                < 1017
                            )
                            v0 = T.axis.spatial(
                                1024,
                                ax0_0_ax1_0_fused // 8 * 64
                                + ax0_1_ax1_1_fused // 2 * 32
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused // 32 * 16
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 2 * 8
                                + ax0,
                            )
                            v1 = T.axis.spatial(
                                1024,
                                ax0_0_ax1_0_fused % 8 * 128
                                + ax0_1_ax1_1_fused % 2 * 64
                                + ax1_2_0_ax0_2_0_ax0_2_1_ax1_2_1_ax0_2_2_fused % 32 // 2 * 4
                                + ax1,
                            )
                            T.reads(Y_reindex_local[v0, v1])
                            T.writes(Y[v0, v1])
                            Y[v0, v1] = Y_reindex_local[v0, v1]


def test_matmul():
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        rt_mod = tvm.build(Module, target="cuda")

    M, N, K = 1012, 1017, 1014
    a_tvm = tvm.nd.array(np.random.rand(M, K).astype("float32"), device=tvm.cuda(0))
    b_tvm = tvm.nd.array(np.random.rand(K, N).astype("float32"), device=tvm.cuda(0))
    c_tvm = tvm.nd.array(np.empty((M, N)).astype("float32"), device=tvm.cuda(0))
    rt_mod(a_tvm, b_tvm, c_tvm)

    time_f = rt_mod.time_evaluator(rt_mod.entry_name, dev=tvm.cuda(0), number=10)
    time = time_f(a_tvm, b_tvm, c_tvm).mean

    flop = (M * N * K + M * N) * 2
    print("GFLOPS: %.2f" % (flop / time / 1e9))


if __name__ == "__main__":
    test_matmul()
