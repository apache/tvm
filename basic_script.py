"""
Some random script.
"""

import os
import json
import time

import tvm
from tvm.ir import load_json
from tvm import meta_schedule as ms
from tvm.target import Target

# get workload
with open('test_tasks/relay-resnet_18-4,3,224,224_extracted_tasks.json', 'rb') as f:
    lines = f.readlines()
    task_conv2d_add = lines[0]
    task_name, task_mod = json.loads(task_conv2d_add)
    task_mod = load_json(json.dumps(task_mod))
    workload = ms.database.Workload(task_mod)
# get one candidate
with open('test_cand/resnet_18-4,3,224,224/fused_nn_conv2d_add.json', 'rb') as f:
    lines = f.readlines()
    cand_str = lines[0]
    json_obj = json.loads(cand_str)
    tuning_record = ms.database.TuningRecord.from_json(json_obj, workload)

# the schedule (containing transformed ir module
sch = tuning_record.transform_ir_module()
mod = sch.mod

# get the prim func
prim_func = mod[mod.get_global_vars()[0]]

# the case where n = 1
# omit the database for now

# create builder / runner
builder = ms.builder.LocalBuilder()
rpc_config = ms.runner.RPCConfig(
    tracker_host="172.31.51.224",
    tracker_port=4445,
    tracker_key="p3.2xlarge",
    session_timeout_sec=100,
)
evaluator_config = ms.runner.EvaluatorConfig(
    number=1,
    repeat=1,
    min_repeat_ms=0,
    enable_cpu_cache_flush=False,
)
rpc_workers = 1 # increase in the future with more workers
runner = ms.runner.RPCRunner(
    rpc_config=rpc_config,
    evaluator_config=evaluator_config,
    max_workers=rpc_workers,
)

# omit batch_size for now, only testing for one example
get_measure = tvm.get_global_func(
    "meta_schedule.TuneContextGetMeasureCandidate"
)
set_measure = tvm.get_global_func(
    "meta_schedule.TuneContextSetMeasureCandidates"
)
send_to_builder = tvm.get_global_func(
    "meta_schedule.TuneContextSendToBuilder"
)
send_to_runner = tvm.get_global_func(
    "meta_schedule.TuneContextSendToRunner"
)
join = tvm.get_global_func(
    "meta_schedule.TuneContextJoin"
)
clear_measure_state = tvm.get_global_func(
    "meta_schedule.TuneContextClearMeasureState"
)

# initialize context
context = ms.TuneContext(
    target=Target("nvidia/geforce-rtx-3070")
)
# get the measure candidate
candidate = get_measure(context, sch, prim_func)
# set the measure candidate
set_measure(context, [candidate])

# send to builder
start_time = time.time()
send_to_builder(context, builder)
end_time = time.time()
print('finished building!')
print(f'artifect path: {context.builder_results[0].artifact_path}')
print(f'error message: {context.builder_results[0].error_msg}')
print(f'building time: {end_time - start_time}')

# send to runner
start_time = time.time()
send_to_runner(context, runner)
results = join(context)
clear_measure_state(context)
end_time = time.time()
print('finished running!')
print(f'running time: {end_time - start_time}')
