import tempfile
import time
import tvm
import numpy as np
import os
from tvm import meta_schedule as ms

def pass_filter(sch, data):
    return True

def filter_func(sch, data):  # original Schedule sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
    try:
        block_name = "depthwise_conv2d_nhwc_output"
        block = sch.get_block(block_name)
        loops = sch.get_loops(block=block)
        factors_1 = data[sch.get_sref(loops[1])]
        factors_2 = data[sch.get_sref(loops[2])]
        factors_3 = data[sch.get_sref(loops[3])]
        _, _, _, s7 = factors_1
        _, _, _, s11 = factors_2
        _, _, _, s15 = factors_3
        calc_1 = "int(s7.value) < 16"
        calc_2 = "int(s11.value) < 16"
        calc_3 = "int(s15.value) <= 32"
        val = eval(calc_1 and calc_2 and calc_3)
        return val
    except RuntimeError:
        return True

def initializer():
    from tvm.runtime import Module
    @tvm.register_func("meta_schedule.builder.ndk_export")
    def ndk_export(mod: Module) -> str:  # pylint: disable=unused-variable
        import tempfile
        from tvm.contrib import ndk  # pylint: disable=import-outside-toplevel

        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod.so")
        mod.export_library(artifact_path, ndk.create_shared)
        return artifact_path

rpc_tracker_host = "0.0.0.0"
rpc_tracker_port = 9190
rpc_tracker_key = "android"

rpc_config=ms.runner.RPCConfig(
            tracker_host=rpc_tracker_host,
            tracker_port=rpc_tracker_port,
            tracker_key=rpc_tracker_key,
            session_timeout_sec=6000,
        )

evaluator_config=ms.runner.EvaluatorConfig(
    number=3,
    repeat=1,
    min_repeat_ms=100,
    enable_cpu_cache_flush=False,
)

ms_rpc_runner = ms.runner.RPCRunner(rpc_config=rpc_config,
            evaluator_config=evaluator_config,
            alloc_repeat=1,
        )

    
target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -num-cores 8", host="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod")
ndk_builder = ms.builder.LocalBuilder(f_export="meta_schedule.builder.ndk_export", timeout_sec=60, initializer = initializer)

seed = 0
np.random.seed(seed)

dtype = "int32"
deep = 64

dshape = (1,32,32,deep)
data = np.random.randint(
    low=np.iinfo(dtype).min,
    high=np.iinfo(dtype).max,
    size=dshape,
    dtype=dtype,
)

wshape = (3,3,deep,1)
weight = np.random.randint(
    low=np.iinfo(dtype).min,
    high=np.iinfo(dtype).max,
    size=wshape,
    dtype=dtype,
)

weight1 = np.random.randint(
    low=np.iinfo(dtype).min,
    high=np.iinfo(dtype).max,
    size=wshape,
    dtype=dtype,
)

weight2 = np.random.randint(
    low=np.iinfo(dtype).min,
    high=np.iinfo(dtype).max,
    size=wshape,
    dtype=dtype,
)

A = tvm.relay.var("data", tvm.relay.TensorType(dshape, dtype))
B = tvm.relay.const(weight, dtype=dtype)
B1 = tvm.relay.const(weight1, dtype=dtype)
B2 = tvm.relay.const(weight2, dtype=dtype)


if deep > 1:
  kernel_layout="HWOI"
else:
  kernel_layout="HWIO"

bias_w = tvm.relay.const(np.array([1]*deep), dtype="int32")

D = tvm.relay.nn.conv2d(data=A, weight=B, kernel_size=(3,3), channels=deep, groups=deep, padding=1, data_layout="NHWC", kernel_layout=kernel_layout, out_layout="", out_dtype="int32")
D = tvm.relay.nn.bias_add(D, bias_w, axis=3)
D = tvm.relay.nn.conv2d(data=D, weight=B1, kernel_size=(3,3), channels=deep, groups=deep, padding=1, data_layout="NHWC", kernel_layout=kernel_layout, out_layout="", out_dtype="int32")
D = tvm.relay.nn.conv2d(data=D, weight=B2, kernel_size=(3,3), channels=deep, groups=deep, padding=1, data_layout="NHWC", kernel_layout=kernel_layout, out_layout="", out_dtype="int32")

model = tvm.IRModule.from_expr(D)
executor = tvm.relay.backend.Executor("graph", {"link-params": True})
model = model.with_attr("executor", executor)

mod = model
params = {"weight":weight, "weight1":weight1, "weight2":weight2}
module_equality = "anchor-block"

t1 = time.time()
with tempfile.TemporaryDirectory() as work_dir:
    database = ms.relay_integration.tune_relay(
        mod=mod,
        params={},
        target=target,
        work_dir=work_dir,
        max_trials_global=800,
        max_trials_per_task=800,
        num_trials_per_iter=8,
        strategy="replay-trace",
        builder=ndk_builder,
        runner=ms_rpc_runner,
        module_equality=module_equality,
        space=ms.space_generator.PostOrderApply(
            sch_rules=[
                ms.schedule_rule.MultiLevelTiling(
                    structure="SSRSRS",
                )
            ],
            postprocs=[
                ms.postproc.FilterLoopSplits(filter=pass_filter),
            ],
            mutator_probs={},
        ),
        seed=0,
    )
t2 = time.time()
print("Tuning time without filter:", t2 - t1, flush=True)

mod = model
params = {"weight":weight, "weight1":weight1, "weight2":weight2}

t1 = time.time()
with tempfile.TemporaryDirectory() as work_dir:
    database = ms.relay_integration.tune_relay(
        mod=mod,
        params={},
        target=target,
        work_dir=work_dir,
        max_trials_global=800,
        max_trials_per_task=800,
        num_trials_per_iter=8,
        strategy="replay-trace",
        builder=ndk_builder,
        runner=ms_rpc_runner,
        module_equality=module_equality,
        space=ms.space_generator.PostOrderApply(
            sch_rules=[
                ms.schedule_rule.MultiLevelTiling(
                    structure="SSRSRS",
                )
            ],
            postprocs=[
                ms.postproc.FilterLoopSplits(filter=filter_func),
            ],
            mutator_probs={},
        ),
        seed=0,
    )
t2 = time.time()
print("Tuning time with filter:", t2 - t1, flush=True)