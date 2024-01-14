import tvm
from tvm.script import ir as I
from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 10), "float16"), T_softmax_norm: T.Buffer((128, 10), "float16")):
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((128,))
    T_softmax_exp = T.alloc_buffer((128, 10))
    T_softmax_expsum_shared = T.alloc_buffer((128,), scope="shared")
    for ax0, ax1 in T.grid(128, 10):
        with T.block("T_softmax_maxelem"):
            v0, v1 = T.axis.remap("SR", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(T_softmax_maxelem[v0])
            with T.init():
                T_softmax_maxelem[v0] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v0] = T.max(T_softmax_maxelem[v0], T.Cast("float32", A[v0, v1]))
    for ax0, ax1 in T.grid(128, 10):
        with T.block("T_softmax_exp"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1], T_softmax_maxelem[v0])
            T.writes(T_softmax_exp[v0, v1])
            T_softmax_exp[v0, v1] = T.exp(T.Cast("float32", A[v0, v1]) - T_softmax_maxelem[v0])
    for ax0_0_fused in T.thread_binding(128, thread="blockIdx.x"):
        for ax0_1_fused in T.thread_binding(1, thread="threadIdx.y"):
            for ax1_1_1_fused in T.thread_binding(10, thread="threadIdx.x"):
                for ax1_0, ax1_1_0 in T.grid(1, 1):
                    with T.block("T_softmax_expsum"):
                        v0 = T.axis.spatial(128, ax0_0_fused)
                        v1 = T.axis.reduce(10, ax1_0 * 10 + ax1_1_0 * 10 + ax1_1_1_fused)
                        T.reads(T_softmax_exp[v0, v1])
                        T.writes(T_softmax_expsum_shared[v0])
                        with T.init():
                            T_softmax_expsum_shared[v0] = T.float32(0)
                        T_softmax_expsum_shared[v0] = T_softmax_expsum_shared[v0] + T_softmax_exp[v0, v1]
    for ax0, ax1 in T.grid(128, 10):
        with T.block("T_softmax_norm"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(T_softmax_exp[v0, v1], T_softmax_expsum_shared[v0])
            T.writes(T_softmax_norm[v0, v1])
            T_softmax_norm[v0, v1] = T.Cast("float16", T_softmax_exp[v0, v1] / T_softmax_expsum_shared[v0])

mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
maxelem_block = sch.get_block("T_softmax_maxelem")
exp_block = sch.get_block("T_softmax_exp")
output_block = sch.get_block("T_softmax_norm")
schedule_block = sch.get_block("T_softmax_expsum")
# sch.reverse_compute_at(output_block, sch.get_loops(schedule_block)[-5])
# sch.compute_at(maxelem_block, sch.get_loops(schedule_block)[-5])
sch.compute_at(exp_block, sch.get_loops(schedule_block)[-5])
