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
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.meta_schedule.arg_info import ArgInfo


def test_tune_matmul_cuda(op_name, order=2):
    mod = create_te_workload(op_name, order)
    with tempfile.TemporaryDirectory() as work_dir:
        target = Target("llvm -mcpu=icelake-server -num-cores 28")
        database = ms.tir_integration.tune_tir(
            mod=mod,
            target=target,
            work_dir=work_dir,
            tuning_time=10000,
            max_trials_global=400,
            num_trials_per_iter=64,
        )
        sch = ms.tir_integration.compile_tir(database, mod, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            args = ArgInfo.from_prim_func(mod)
            dev = tvm.device("cpu", 0)
            myfunc = tvm.build(sch.mod, target=target, name=op_name)
            inputs = []
            for arg in args:
                shape = arg.shape
                dtypes = arg.dtype
                buffer = tvm.nd.array(np.zeros(shape, dtype=dtypes), dev)
                inputs.append(buffer)
            evaluator = myfunc.time_evaluator(
                myfunc.entry_name, dev, repeat=1, number=100
            )
            eval_time = evaluator(*inputs).mean * 1e3
            print(f"The time of {op_name} is {eval_time} ms")


if __name__ == "__main__":
    tensor_ir = ["C1D","C2D","C3D", "DEP", "DIL", "GRP", "T2D", "GMM", "SFM"]
    for order in range(4):
        print("===========================")
        for op_name in tensor_ir:
            test_tune_matmul_cuda(op_name, order)
