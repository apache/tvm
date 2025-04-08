import logging
import tempfile
import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.target import Target
import numpy as np
from tvm.tir.analysis import estimate_tir_flops
from tvm.ir import IRModule
logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [1024, 1024])
    B = T.match_buffer(b, [1024, 1024])
    C = T.match_buffer(c, [1024, 1024])
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm -mcpu=icelake-server -num-cores 28")
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=64,
            num_trials_per_iter=16,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            dev = tvm.device("cpu", 0)
            a_np = np.random.uniform(size=(1024, 1024)).astype("float32")
            b_np = np.random.uniform(size=(1024, 1024)).astype("float32")
            buff_a = tvm.nd.array(a_np, dev)
            buff_b = tvm.nd.array(b_np, dev)
            buff_c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
            myfunc = tvm.build(sch.mod, target=target)

            evaluator = myfunc.time_evaluator(  
                myfunc.entry_name, dev, repeat=1000, number=1  
            )  
            eval_time = evaluator(buff_a, buff_b, buff_c).mean
            flops = estimate_tir_flops(IRModule({"main": matmul}))
            print(f"Final flops: {flops / eval_time * 1e-9} GB/s")
            # 1346.7994215407598 GB/s

if __name__ == """__main__""":
    test_tune_matmul_cuda()
