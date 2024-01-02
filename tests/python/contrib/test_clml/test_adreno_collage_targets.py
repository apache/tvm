# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Compares Collage with various other baselines."""

import tvm
import logging
import tempfile
import os
import shutil
import numpy as np
from tvm.relay import testing
from tvm import rpc
from tvm.contrib import utils, ndk
from tvm.relay.build_module import bind_params_by_name

# The following are necessary to force global functions or pattern tables to be registered
from tvm.relay.collage.collage import *
from tvm.relay.op.contrib import clml
import pytest

logging.basicConfig(level=logging.INFO)


########### Configuration ###########

###
### TVM Opencl AutoTvm log file name
###
TUNING_LOG = ""

###
### If true, run all models
###
ALL_MODELS = False

###
### If true, run all configurations
###
ALL_CONFIGS = False

###
### How aggressively to look for candidates?
###
TVM_MAX_DEPTH = 8
BYOC_MAX_DEPTH = 8

###
### AutoTVM tuning parameters.
###
AUTOTVM_NUM_TRIALS = 1024
AUTOTVM_EARLY_STOPPING = 600
TIMEOUT = 10
MEASURE_NUMBER = tvm.relay.collage.MEASURE_NUMBER
MEASURE_REPEAT = tvm.relay.collage.MEASURE_REPEAT
WARMUP_MIN_REPEAT_MS = tvm.relay.collage.WARMUP_MIN_REPEAT_MS

##
## RPC Build configuration
##
HOST = tvm.target.Target("llvm -mtriple=arm64-linux-android")
OPENCL = tvm.target.Target("opencl", HOST)
RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 9090))
RPC_KEY = os.getenv("RPC_DEVICE_KEY", "android")
NDK_CROSS_COMPILER = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")


########### AutoTVM tuning helpers ###########


def extract_autotvm_tasks(mod, target):
    """Returns TVM kernels to tune for mod and target."""
    return tvm.autotvm.task.extract_from_program(mod, target=target, params=None)


def optional_tuning_records(log_filename):
    """Returns existing tuning records, if any."""
    if log_filename == "" or not os.path.exists(log_filename):
        return tvm.autotvm.task.FallbackContext()
    else:
        return tvm.autotvm.task.ApplyHistoryBest(log_filename)


def is_already_tuned(task, log_filename):
    """Returns True if we already have a tuning record for task in turning logs in log_filename"""
    if not os.path.exists(log_filename):
        return False

    dispatch_context = tvm.autotvm.task.ApplyHistoryBest(log_filename)
    return dispatch_context._query_inside(task.target, task.workload)


def tune_autotvm_tasks(tasks, log_filename):
    """Appends to log filename the best strategies for tasks"""
    if len(tasks) == 0:
        return

    measure_option = tvm.autotvm.measure_option(
        builder=tvm.autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
        runner=tvm.autotvm.RPCRunner(
            RPC_KEY, host=RPC_TRACKER_HOST, port=RPC_TRACKER_PORT, number=100, timeout=15
        ),
    )

    logging.info(
        f"Using autotvm tuning for {len(tasks)} tasks with {AUTOTVM_NUM_TRIALS} trials, logging to {log_filename}"
    )

    # create tmp log file, starting with contents from existing log file
    tmp_log_filename = log_filename + ".tmp"
    if os.path.exists(tmp_log_filename):
        os.remove(tmp_log_filename)
    if os.path.exists(log_filename):
        logging.info(f"Copying existing log {log_filename} to {tmp_log_filename}")
        shutil.copy(log_filename, tmp_log_filename)

    for i, task in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        logging.info(f"Considering task {task.name} {prefix}")
        if is_already_tuned(task, tmp_log_filename):
            logging.info(f"Re-using existing record for {task.name}")
            continue

        logging.info(f"Using autotvm to tune {task.name}")
        tuner_obj = tvm.autotvm.tuner.XGBTuner(task, loss_type="reg")
        if os.path.exists(tmp_log_filename):
            tuner_obj.load_history(tvm.autotvm.record.load_from_file(tmp_log_filename))

        # do tuning
        n_trial = min(AUTOTVM_NUM_TRIALS, len(task.config_space))
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=AUTOTVM_EARLY_STOPPING,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.progress_bar(n_trial, prefix=prefix),
                tvm.autotvm.callback.log_to_file(tmp_log_filename),
            ],
        )

    # Pick best records and copy back to main log file
    tvm.autotvm.record.pick_best(tmp_log_filename, log_filename)
    os.remove(tmp_log_filename)

    logging.info("Done with autotvm tuning")


def autotvm_tune_module(mod, target, log_filename):
    if log_filename == "":
        logging.info("Not tuning with autotvm since disabled")
        return
    # Extract and tune any TVM kernels. BYOC partitions will have no tasks extracted.
    logging.info("Extracting tasks from overall module")
    tasks = extract_autotvm_tasks(mod, target)
    logging.info(f"Auto-tuning {len(tasks)} tasks from overall module")
    tune_autotvm_tasks(tasks, log_filename)


########### Drivers ###########


def compile_and_benchmark(label, model, targets, tmp_dir):
    """Compile model for target and run it with profiling."""
    logging.info(f"Compiling {model['name']} using {label} with {targets}...")
    mod = model["mod"]
    mod = clml.preprocess_for_clml(mod)
    exe = tvm.relay.vm.compile(mod, target=targets, params=model["params"])
    lib = exe.mod
    lib_path = os.path.join(tmp_dir, "lib.so")
    logging.info(f"Exporting library to {lib_path}...")
    lib.export_library(lib_path, cc=NDK_CROSS_COMPILER)
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    ctx = remote.cl(0)
    remote_path = "lib.so"
    remote.upload(lib_path, target=remote_path)
    lib = remote.load_module(remote_path)
    vm_factory = tvm.runtime.vm.VirtualMachine(lib, ctx)
    args = {v.name_hint: arg_for(v.checked_type, ctx) for v in mod["main"].params}
    logging.info(f"Benchmarking for {model['name']} generated by {label}...")
    profile = vm_factory.benchmark(
        ctx, repeat=MEASURE_REPEAT, number=MEASURE_NUMBER, min_repeat_ms=0, **args
    )
    logging.info(f"Benchmarked for {model['name']} generated by {label}: {profile}")
    logging.info(f"RESULT: {label} | {model['name']} | {profile.median * 1e3}ms")


# Custom cost function for Opencl RPC targets.
@register_func("tvm.relay.collage.opencl_cost_estimator")
def opencl_cost_estimator(mod, target):
    mod = clml.preprocess_for_clml(mod) if "clml" == target.kind.name else mod
    try:
        # Build the module.
        logging.info("Compiling module to estimate")
        exe = tvm.relay.vm.compile(mod, target)
    except RuntimeError as err:
        # A build failure indicates the partition is not supported.
        # eg trying to build an nn.batch_norm on GPU, which has no schedule since we assume it
        # is only ever used with a tuple projection which is rewritten away.
        logging.info("Assigning module infinite cost since unable to build: %s", err)
        return math.inf

    lib = exe.mod
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    temp = utils.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    ctx = remote.cl(0)
    lib.export_library(dso_binary_path, cc=NDK_CROSS_COMPILER)
    remote_path = dso_binary
    remote.upload(dso_binary_path, target=remote_path)
    lib = remote.load_module(remote_path)

    vm_factory = tvm.runtime.vm.VirtualMachine(lib, ctx)
    func_name = "main"
    main_args = {v.name_hint: arg_for(v.checked_type, ctx) for v in mod[func_name].params}
    cost = vm_factory.benchmark(
        ctx, repeat=5, number=20, min_repeat_ms=0, func_name=func_name, **main_args
    )
    return cost.mean


def collage(model):
    """Run the Collage partitioner for a set of Opencl Adreno related targets and profile the result"""
    logging.info(f"collage | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    autotvm_tune_module(model["mod"], OPENCL, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        targets = []
        targets.append(OPENCL)
        use_fp16 = model["main_dtype"] == "float16"
        tmp_dir = tempfile.mkdtemp()
        targets.append(tvm.target.Target("clml", HOST))

        # Register byoc fusion style for compiler with available
        # options [compiler.NoFusion | compiler.TVMFusion | compiler.MaxDepthFusion]
        config = {
            "relay.collage.tvm_max_depth": TVM_MAX_DEPTH,
            "relay.collage.byoc_max_depth": BYOC_MAX_DEPTH,
            "relay.collage.byoc_fusion_style": ["clml.NoFusion"],
        }
        logging.info(f"Using PassContext(config={config}")
        ctxt = tvm.transform.PassContext(config=config)
        config = tvm.target.make_compilation_config(ctxt, targets)
        with ctxt:
            mod = model["mod"]
            mod = tvm.relay.transform.CapturePostDfsIndexInSpans()(mod)
            logging.info("-------------- BEGIN INDEXED --------------")
            logging.info(mod)
            logging.info("-------------- END INDEXED ----------------")
            # Register python custom cost function for targets in
            # custom cost estimator module.
            cost_estimator = CustomCostEstimator(
                py_fn_estimator="tvm.relay.collage.opencl_cost_estimator"
            )
            mod = tvm.relay.transform.CollagePartition(config, cost_estimator=cost_estimator)(mod)
            partitioned_model = model.copy()
            partitioned_model["mod"] = mod
            logging.info("-------------- BEGIN PARTITIONED --------------")
            logging.info(partitioned_model["mod"])
            logging.info("-------------- END PARTITIONED ----------------")
            compile_and_benchmark("collage", partitioned_model, targets, tmp_dir)


def just_clml(model):
    """Run partition_for_clml, complete the compilation with TVM, and profile the result."""
    logging.info(f"just_clml | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        logging.info("Partitioning for CLML...")
        mod = tvm.relay.op.contrib.clml.partition_for_clml(model["mod"], model["params"])
        partitioned_model = model.copy()
        partitioned_model["mod"] = mod
        logging.info("-------------- BEGIN PARTITIONED --------------")
        logging.info(partitioned_model["mod"])
        logging.info("-------------- END PARTITIONED ----------------")
        targets = []
        targets.append(OPENCL)
        targets.append(tvm.target.Target("clml", HOST))
        compile_and_benchmark("just_clml", partitioned_model, targets, tmp_dir)


def just_tvm(model):
    """Compile and profile using vanilla TVM."""
    logging.info(f"just_tvm | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    tmp_dir = tempfile.mkdtemp()
    autotvm_tune_module(model["mod"], OPENCL, TUNING_LOG)
    with optional_tuning_records(TUNING_LOG):
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            compile_and_benchmark("just_tvm", model, OPENCL, tmp_dir)


def get_model(model_name, dtype):

    if "mobilenet" in model_name:
        mod, params = testing.mobilenet.get_workload(batch_size=1, dtype=dtype)
    elif "resnet" in model_name:
        mod, params = testing.resnet.get_workload(num_layers=50, batch_size=1, dtype=dtype)
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
        mod = tvm.relay.transform.FoldConstant()(mod)
    return {
        "name": model_name,
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": dtype},
        "mod": mod,
        "params": params,
        "main_dtype": dtype,
    }


########### Runners ###########
@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def run_resnet50(dtype):

    just_clml(get_model("resnet-50", dtype))
    just_tvm(get_model("resnet-50", dtype))
    """Run Collage for tvm and clml compiler target."""
    collage(get_model("resnet-50", dtype))


@pytest.mark.parametrize("dtype", ["float32"])
@tvm.testing.requires_openclml
def run_mobilenetv1(dtype):

    just_clml(get_model("mobilenet", dtype))
    just_tvm(get_model("mobilenet", dtype))
    """Run Collage for tvm and clml compiler target."""
    collage(get_model("mobilenet", dtype))
