import tvm
import numpy as np
import sys
from tvm.tir import Schedule
from tvm import rpc
import os
from tvm import relay

from tvm.relay.backend import Executor

from tvm.contrib import ndk

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


def initializer():
    from tvm.runtime import Module

    @tvm.register_func("meta_schedule.builder.ndk_export")
    def ndk_export(mod: Module) -> str:  # pylint: disable=unused-variable
        import tempfile
        from tvm.contrib import ndk  # pylint: disable=import-outside-toplevel

        artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod.so")
        mod.export_library(artifact_path, ndk.create_shared)
        return artifact_path


def tflite_mobilenet_v1_quant(_name, _target, _target_host):
    try:
        import tflite.Model
    except ImportError:
        print("missing tflite support")

    name = _name
    tflite_model_path = os.path.join(
        ("/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite"), name + ".tflite"
    )

    with open(tflite_model_path, "rb") as f:
        tflite_model_buffer = f.read()

    try:
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
    except AttributeError:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)

    shape_dict = {}
    dtype_dict = {}
    shape_dict["input"] = [1, 224, 224, 3]
    dtype_dict["input"] = "float32"
    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
    )

    print("Compile default:", name)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=_target, target_host=_target_host, params=params)

    export_path = os.path.join(
        "/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite", name + ".so"
    )
    lib.export_library(export_path, ndk.create_shared)

    json_path = tflite_model_path + ".graph.json"
    text_file = open(json_path, "w")
    text_file.write(lib.get_graph_json())
    text_file.close()
    return mod, params


def benchmark(name, extension):
    if len(extension):
        input_path = name + "." + extension + ".so"
        json_path = name + "." + extension + ".graph.json"
    else:
        input_path = name + ".so"
        json_path = name + ".graph.json"

    input_path = os.path.join(
        "/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite", input_path
    )
    json_path = os.path.join(
        "/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite", json_path
    )
    print("measuring " + input_path)

    tracker = rpc.connect_tracker("0.0.0.0", 9190)
    remote = tracker.request("android", priority=0, session_timeout=None)

    ctx = remote.cpu(0)

    remote.upload(input_path)

    text_file = open(json_path, "r")
    json = text_file.read()
    text_file.close()

    from tvm.contrib import graph_executor

    lib = remote.load_module(os.path.basename(input_path))
    m = graph_executor.create(json, lib, ctx)
    ftimer = m.module.time_evaluator("run", ctx, repeat=10, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
    )


seed = 0
np.random.seed(seed)

from tvm import meta_schedule as ms

target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod"
target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod"
target_llvm = tvm.target.Target(target + " -num-cores 8", host=target_host)

name = "mobilenet_v1_1.0_224"

mod, params = tflite_mobilenet_v1_quant(name, target, target_host)
module_equality = "structural"
# module_equality = "anchor-block"
print("module_equality", module_equality, flush=True)
extracted_tasks = ms.relay_integration.extract_tasks(
    mod,
    target_llvm,
    params,
    module_equality=module_equality,
)
for i in range(len(extracted_tasks)):
    print(
        "task name", extracted_tasks[i].task_name, "weight", extracted_tasks[i].weight, flush=True
    )


work_dir = "mobilenet_v1/full_no_filter"
import os, shutil

print("rm", os.path.join("/Users/admin/workspace/ms_filter_tvm/tvm", work_dir), flush=True)
if os.path.isdir(os.path.join("/Users/admin/workspace/ms_filter_tvm/tvm", work_dir)):
    shutil.rmtree(os.path.join("/Users/admin/workspace/ms_filter_tvm/tvm", work_dir))
else:
    print("nothing to remove")

TUNE = True

import time

t1 = time.time()
from tvm.tir.schedule import BlockRV, Instruction, InstructionKind, LoopRV, Trace
from typing import List

if TUNE:
    def f(sch, data):  # orig sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
        try:
            block_name = "depthwise_conv2d_nhwc_output"
            block = sch.get_block(block_name)
            print(f"BLOCK WITH NAME {block_name} is ", block)
            loops = sch.get_loops(block=block)
            ddata = data
            factors_1 = ddata[sch.get_sref(loops[1])]
            factors_2 = ddata[sch.get_sref(loops[2])]
            factors_3 = ddata[sch.get_sref(loops[3])]
            print("all factors", [ddata[sch.get_sref(l)] for l in loops])
            s4, s5, s6, s7 = factors_1
            s8, s9, s10, s11 = factors_2
            s12, s13, s14, s15 = factors_3
            calc_1 = "int(s5.value) > 4"
            calc_2 = "int(s6.value) < 16"
            calc_3 = "int(s7.value) < 8"
            calc_4 = "int(s15.value) >= 16"
            val = eval(calc_1 and calc_2 and calc_3 and calc_4)
            print(
                f"Evaluate logical expression for block with name {block_name}: ",
                calc_1,
                " and ",
                calc_2,
                " and ",
                calc_3,
                " and ",
                calc_4,
                " is ",
                val,
                "where s5",
                int(s5.value),
                "where s6",
                int(s6.value),
                "where s7",
                int(s7.value),
                "where s15",
                int(s15.value),
                flush=True,
            )
            return val
        except RuntimeError:
            return True

    def f(sch, data):  # orig sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
            try:
                block_name = "depthwise_conv2d_nhwc_output"
                block = sch.get_block(block_name)
                print(f"BLOCK WITH NAME {block_name} is ", block)
                loops = sch.get_loops(block=block)
                ddata = data
                factors_1 = ddata[sch.get_sref(loops[1])]
                factors_2 = ddata[sch.get_sref(loops[2])]
                factors_3 = ddata[sch.get_sref(loops[3])]
                print("all factors", [ddata[sch.get_sref(l)] for l in loops])
                s4, s5, s6, s7 = factors_1
                s8, s9, s10, s11 = factors_2
                s12, s13, s14, s15 = factors_3
                calc_1 = "int(s5.value) > 4"
                calc_2 = "int(s6.value) < 16"
                calc_3 = "int(s7.value) < 8"
                calc_4 = "int(s15.value) >= 16"
                val = eval(calc_1 and calc_2 and calc_3 and calc_4)
                print(
                    f"Evaluate logical expression for block with name {block_name}: ",
                    calc_1,
                    " and ",
                    calc_2,
                    " and ",
                    calc_3,
                    " and ",
                    calc_4,
                    " is ",
                    val,
                    "where s5",
                    int(s5.value),
                    "where s6",
                    int(s6.value),
                    "where s7",
                    int(s7.value),
                    "where s15",
                    int(s15.value),
                    flush=True,
                )
                return val
            except RuntimeError:
                return True

    ndk_builder = ms.builder.LocalBuilder(
        f_export="meta_schedule.builder.ndk_export", timeout_sec=60, initializer=initializer
    )

    rpc_config = ms.runner.RPCConfig(
        tracker_host="0.0.0.0",
        tracker_port=9190,
        tracker_key="android",
        session_timeout_sec=6000,
    )

    evaluator_config = ms.runner.EvaluatorConfig(
        number=3,
        repeat=1,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    )

    ms_rpc_runner = ms.runner.RPCRunner(
        rpc_config=rpc_config,
        evaluator_config=evaluator_config,
        alloc_repeat=1,
    )

    database = ms.relay_integration.tune_relay(
        mod=mod,
        params=params,
        target=target_llvm,
        work_dir=work_dir,
        max_trials_global=20000,
        max_trials_per_task=8 * 40,
        num_trials_per_iter=8,
        # strategy=ms.search_strategy.EvolutionarySearch(),
        # strategy=ms.search_strategy.ReplayFunc(),
        strategy=ms.search_strategy.ReplayTrace(),
        builder=ndk_builder,
        runner=ms_rpc_runner,
        module_equality=module_equality,
        space=ms.space_generator.PostOrderApply(
            sch_rules=[
                ms.schedule_rule.MultiLevelTiling(
                    structure="SSRSRS",
                    # tile_binds=None,
                    # max_innermost_factor=64,
                    # vector_load_lens=None,
                    # reuse_read=None,
                    # reuse_write=ms.schedule_rule.ReuseType(req="may", levels=[1,2], scope="global"),
                    # filter_out_fn=ff,
                )
            ],
            postprocs=[
                ms.postproc.FilterLoopSplits(filter=ff),
                ms.postproc.RewriteParallelVectorizeUnroll(),
                ms.postproc.RewriteReductionBlock(),
                ms.postproc.RewriteTensorize(vectorize_init_loop=True),
            ],
            mutator_probs={},
        ),
        seed=0,
    )
else:
    database = ms.database.JSONDatabase(
        "%s/database_workload.json" % work_dir,
        "%s/database_tuning_record.json" % work_dir,
        module_equality=module_equality,
    )

t2 = time.time()
print("time", t2 - t1, flush=True)

print("ICE get_all_tuning_records", database.get_all_tuning_records(), flush=True)

data = []
for r in database.get_all_tuning_records():
    if r.run_secs:
        data.append(["{0:.20f}".format(r.run_secs[0].value), r.trace, r])

sdata = sorted(data, key=lambda v: float(v[0]))


record = sdata[0][2]
print("orig\n", record.workload.mod, flush=True)
sch = Schedule(record.workload.mod)
print("orig sch\n", sch.trace, flush=True)
record.trace.apply_to_schedule(sch, remove_postproc=False)
print("best\n", sch.mod, flush=True)
print("best time\n", sdata[0][0], flush=True)
print("best trace\n", record.trace, flush=True)

record = sdata[-1][2]
sch = Schedule(record.workload.mod)
record.trace.apply_to_schedule(sch, remove_postproc=False)
print("worst\n", sch.mod, flush=True)
print("worst time\n", sdata[-1][0], flush=True)
print("worst trace\n", record.trace, flush=True)

print("Compile metaschedule:", name)

executor = Executor("graph", {"link-params": True})

lib = ms.relay_integration.compile_relay(
    database=database,
    mod=mod,
    target=target,
    params=params,
    executor=executor,
)

export_path = os.path.join(
    "/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite", name + ".ms.so"
)
lib.export_library(export_path, ndk.create_shared)

json_path = os.path.join(
    "/Users/admin/workspace/scripts/metascheduler/comp_IEs/TFLite", f"{name}.ms.graph.json"
)
text_file = open(json_path, "w")
text_file.write(lib.get_graph_json())
text_file.close()

print("Benchmarking default", name)
benchmark(name, "")
print("Benchmarking tuned", name)
benchmark(name, "ms")
