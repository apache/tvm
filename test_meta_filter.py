import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import tempfile
import numpy as np
import sys
from tvm.tir import Schedule
from tvm import tir
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

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

print("relay\n",model, flush=True)

with tvm.transform.PassContext(opt_level=3):
    tvm_model = tvm.relay.build_module.create_executor("graph", model, tvm.cpu(0), "llvm -mcpu=core-avx2", {}).evaluate()

from tvm import meta_schedule as ms
target = "llvm -mcpu=core-avx2 -num-cores=1"

mod = model
params = {"weight":weight, "weight1":weight1, "weight2":weight2}
# module_equality = "structural"
module_equality = "anchor-block"
print("module_equality", module_equality, flush=True)
extracted_tasks=ms.relay_integration.extract_tasks(
        mod,
        target,
        params,
        module_equality=module_equality,
    )
for i in range(len(extracted_tasks)):
    print("task name", extracted_tasks[i].task_name, "weight", extracted_tasks[i].weight, flush=True)
    print("task mod\n", extracted_tasks[i].dispatched[0].script(), flush=True)


work_dir = "my_dir/"
import os, shutil
print("rm", os.path.join("/git/tvm/", work_dir), flush=True)
if os.path.isdir(os.path.join("/git/tvm/", work_dir)):
    shutil.rmtree(os.path.join("/git/tvm/", work_dir))
else:
    print("nothing to remove")

TUNE=True

import time
t1 = time.time()
from tvm.tir.schedule import BlockRV, Instruction, InstructionKind, LoopRV, Trace
from typing import List

if TUNE:

    def f(sch, block_rv): # TODO sch or orig sch, and sch trace
        block = sch.get(block_rv)
        print("sch trace f", sch.trace, flush=True)
        print("block", block, block.name_hint)
        if block.name_hint == "DepthwiseConv2d":
            loops = sch.get_loops(block=block)
            l1_factors = sch.get_split_factors(loop=loops[0])
            s1,s2,s3,s4 = l1_factors
            calc = "int(s4.value) <= 2"
            val = eval(calc)
            print("ICERES ", calc, "is", val, "where s4", int(s4.value))
            return val
        print("another block")
        return True
# orig_sch 6 loops no trace
# sch 20 loops has trace
    def ff(sch, block_rv, data): # TODO orig sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
        block = sch.get(block_rv)
        print("sch trace ff", sch.trace, flush=True)
        print("data ff", data, flush=True)
        ddata = { sch.get_sref(loop_rv) : factors  for loop_rv, factors in data.items() }
        print("data convert ff", ddata, flush=True)
        print("block", block, block.name_hint)
        if block.name_hint == "DepthwiseConv2d":
            # loops = sch.get_loops(block=block)
            loops = sch.get_loops(block=block_rv)
            # print("l1 l1__", loops[0], loops__[0], loops[0] == loops__[0], sch.get_sref(loops[0]) == sch.get_sref(loops__[0]))
            print("l1", loops[0])
            print("py loops: ", [str(l) for l in loops])
            print("py loops sref: ", [str(sch.get_sref(l)) for l in loops])
            factors = ddata[sch.get_sref(loops[1])]
            print("l2 factors", factors)
            print("all factors", [ddata[sch.get_sref(l)] for l in loops])
            s1,s2,s3,s4 = factors
            calc = "int(s4.value) <= 2"
            val = eval(calc)
            print("ICERES ", calc, "is", val, "where s4", int(s4.value))
            return val
        print("another block")
        return True

    def fff(sch, data): # TODO orig sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
        block = sch.get_block("DepthwiseConv2d")
        print("sch trace fff", sch.trace, flush=True)
        print("data fff", data, flush=True)
        ddata = { sch.get_sref(loop_rv) : factors  for loop_rv, factors in data.items() }
        print("data convert fff", ddata, flush=True)
        loops = sch.get_loops(block=block)
        # loops = sch.get_loops(block=block_rv)
        print("l1", loops[0])
        print("py loops: ", [str(l) for l in loops])
        print("py loops sref: ", [str(sch.get_sref(l)) for l in loops])
        if len(loops) < 1: return True
        if ddata.get(sch.get_sref(loops[1]), None) is None: return True
        factors = ddata[sch.get_sref(loops[1])]
        print("l2 factors ", factors)
        print("all factors", [ddata[sch.get_sref(l)] for l in loops])
        # if factors:
        # s1,s2,s3,s4 = l1_splits[0]
        s1,s2,s3,s4 = factors
        calc = "int(s4.value) <= 2"
        val = eval(calc)
        print("ICERES ", calc, "is", val, "where s4", int(s4.value))
        return val
    
    def ffff(sch, data): # TODO orig sch, data is {loop : factors} map Dict{LoopRV:List[Int]}
        block = sch.get_block("DepthwiseConv2d")
        # print("sch trace ffff", sch.trace, flush=True)
        loops = sch.get_loops(block=block)
        # print("py loops: ", [str(l) for l in loops], flush=True)
        # print("py loops sref: ", [str(sch.get_sref(l)) for l in loops], flush=True)
        # print("data ffff", data, flush=True)
        # ddata = { sch.get_sref(loop_rv) : factors  for loop_rv, factors in data.items() }
        ddata = data
        # print("data convert ffff", ddata, flush=True)
        # loops = sch.get_loops(block=block_rv)
        # print("l1", loops[0])



        factors = ddata[sch.get_sref(loops[1])]
        # print("l2 factors ", factors)
        print("all factors", [ddata[sch.get_sref(l)] for l in loops])
        # if factors:
        # s1,s2,s3,s4 = l1_splits[0]
        s1,s2,s3,s4 = factors
        calc = "int(s4.value) <= 2"
        val = eval(calc)
        print("ICERES ", calc, "is", val, "where s4", int(s4.value), flush=True)
        return val

    database = ms.relay_integration.tune_relay(
        mod=mod,
        params={},
        target=target,
        work_dir=work_dir,
        # for faster tuning
        max_trials_global=20000,
        # max_trials_per_task=8,
        # max_trials_per_task=1,
        max_trials_per_task=8*10, 
        num_trials_per_iter=8,
        # strategy=ms.search_strategy.EvolutionarySearch(),
        strategy=ms.search_strategy.ReplayTrace(),
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
            # postprocs=[],
            # postprocs=[ms.postproc.FilterLoopSplits(filter=fff)],
            postprocs=[ms.postproc.FilterLoopSplits(filter=ffff)],
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

# raise

t2 = time.time()
print("time",t2-t1, flush=True)

print("ICE get_all_tuning_records", database.get_all_tuning_records(), flush=True)

data = []
for r in database.get_all_tuning_records():
    # print("rec", r, flush=True)
    # print("rec trace", r.trace, flush=True)
    if r.run_secs:
        # print("rec run_secs", ['{0:.20f}'.format(v.value) for v in r.run_secs], flush=True)
        data.append(['{0:.20f}'.format(r.run_secs[0].value), r.trace, r])

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
# for d in sdata:
#     record = d[2]
#     print("d\n", sch.mod, flush=True)
#     print("d time\n", d[0], flush=True)
#     print("d trace\n", record.trace, flush=True)