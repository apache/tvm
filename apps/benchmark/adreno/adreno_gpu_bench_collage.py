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
import argparse
import tvm
from tvm import relay
import logging
import os
import sys
import numpy as np
from tvm.relay import testing
from tvm.contrib.utils import tempdir
from tvm import rpc
from tvm.relay.build_module import bind_params_by_name
from tvm import autotvm
from tvm.runtime.vm import VirtualMachine
import tvm.contrib.graph_executor as runtime
from tvm.contrib import utils, ndk
from tvm.relay.collage.collage import *
from tvm.relay.op.contrib import clml

logging.basicConfig(level=logging.INFO)


###
### How aggressively to look for candidates?
###
TVM_MAX_DEPTH = 8
BYOC_MAX_DEPTH = 8

##
## Default config definition
##
HOST = tvm.target.Target("llvm -mtriple=arm64-linux-android")
OPENCL = tvm.target.Target("opencl -device=adreno", HOST)
NDK_CC = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")


def print_progress(msg):
    """print progress message

    Parameters
    ----------
    msg: str
        The message to print
    """
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1024,
    early_stopping=None,
    log_filename="tuning.log",
):
    from tvm.autotvm.tuner import XGBTuner

    tmp_log_file = log_filename + ".tmp"

    for i, tsk in enumerate(reversed(tasks)):
        print("Task: ", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

        autotvm.record.pick_best(tmp_log_file, log_filename)


########### Collage Drivers ###########


def compile_and_run(label, model, targets, inputs):
    """Compile model for target and run it with profiling."""
    logging.info(f"Compiling {model['name']} using {label} with {targets}...")
    mod = model["mod"]
    exe = tvm.relay.vm.compile(mod, target=targets, params=model["params"])
    lib = exe.mod
    temp = utils.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    logging.info(f"Exporting library to {dso_binary_path}...")
    lib.export_library(dso_binary_path, cc=NDK_CC)
    tracker = rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(args.rpc_key, priority=0, session_timeout=600)
    ctx = remote.cl(0)
    remote.upload(dso_binary_path)
    rlib = remote.load_module(dso_binary)
    vm_factory = tvm.runtime.vm.VirtualMachine(rlib, ctx, "naive")
    func_name = "main"
    main_args = {v.name_hint: arg_for(v.checked_type, ctx) for v in mod[func_name].params}
    profile = vm_factory.benchmark(
        ctx, repeat=5, number=20, min_repeat_ms=0, func_name=func_name, **main_args
    )
    return profile.mean


def collage(model, input_data, tune_log=""):
    """Run the Collage partitioner for a set of Opencl Adreno related targets and profile the result"""
    logging.info(f"collage | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with autotvm.apply_history_best(tune_log):
        targets = []
        targets.append(OPENCL)
        use_fp16 = model["main_dtype"] == "float16"
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
            """Collage partition with tvm opencl and clml target on rpc device"""
            mod = tvm.relay.transform.CollagePartition(
                config,
                cost_estimator=CostEstimator(
                    host=args.host, port=args.port, rpc_key=args.rpc_key, ndk_cc=NDK_CC
                ),
            )(mod)
            partitioned_model = model.copy()
            partitioned_model["mod"] = mod
            logging.info("-------------- BEGIN PARTITIONED --------------")
            logging.info(partitioned_model["mod"])
            logging.info("-------------- END PARTITIONED ----------------")
            return compile_and_run("collage", partitioned_model, targets, input_data)


def just_clml(model, input_data, tune_log=""):
    """Run partition_for_clml, complete the compilation with TVM, and profile the result."""
    logging.info(f"just_clml | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3):
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
            return compile_and_run("just_clml", partitioned_model, OPENCL, input_data)


def just_tvm(model, input_data, tune_log=""):
    """Compile and profile using vanilla TVM."""
    logging.info(f"just_tvm | {model['name']}")
    logging.info("-------------- BEGIN ORIGINAL --------------")
    logging.info(model["mod"])
    logging.info("-------------- END ORIGINAL ----------------")
    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3):
            return compile_and_run("just_tvm", model, OPENCL, input_data)


def get_model(model_name, dtype):

    if "mobilenet" in model_name:
        mod, params = testing.mobilenet.get_workload(batch_size=1, dtype=dtype)
    elif "resnet" in model_name:
        n_layer = int(model_name.split("-")[1])
        mod, params = testing.resnet.get_workload(num_layers=n_layer, batch_size=1, dtype=dtype)
    elif model_name == "inception_v3":
        input_shape = (1, 3, 299, 299)
        mod, params = testing.inception_v3.get_workload(batch_size=1, dtype=dtype)
    elif "vgg" in model_name:
        n_layer = int(model_name.split("-")[1])
        mod, params = testing.vgg.get_workload(num_layers=n_layer, batch_size=1, dtype=dtype)
    elif "densenet" in model_name:
        n_layer = int(model_name.split("-")[1])
        mod, params = testing.densenet.get_workload(
            densenet_size=n_layer, batch_size=1, dtype=dtype
        )
    elif "squeezenet" in model_name:
        version = model_name.split("_v")[1]
        mod, params = testing.squeezenet.get_workload(batch_size=1, version=version, dtype=dtype)

    initializer = tvm.relay.testing.init.Xavier()
    for param_name in list(params.keys()):
        filter_data = np.zeros(params[param_name].shape).astype(params[param_name].dtype)
        if len(filter_data.shape) > 1:
            initializer("weight", filter_data)
        else:
            initializer("bias", filter_data)
        params[param_name] = tvm.nd.array(filter_data)

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
def evaluate_network(model_name, dtype):
    print("Network evaluating .. " + model_name + " " + dtype)
    np.random.seed(0)
    model = get_model(model_name, dtype)
    tune_log = "adreno_v0.01.log"
    if args.tune:
        # Auto Tuning
        tune_log = "adreno-" + model_name + "-" + dtype + ".log"
        tuning_options = {
            "log_filename": tune_log,
            "early_stopping": None,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
                runner=autotvm.RPCRunner(
                    args.rpc_key,
                    host=args.host,
                    port=args.port,
                    number=3,
                    timeout=600,
                ),
            ),
        }
        tasks = autotvm.task.extract_from_program(
            net, target=OPENCL, target_host=HOST, params=params
        )
        tune_tasks(tasks, **tuning_options)

    print_progress("%-20s building..." % network)
    input_data = {}
    for name, shape in model["input_shapes"].items():
        input_data[name] = np.random.uniform(-1.0, 1.0, shape).astype(model["input_dtypes"][name])
    clml_time = just_clml(model, input_data, tune_log)
    tvm_time = just_tvm(model, input_data, tune_log)

    """Run Collage for tvm and clml compiler target."""
    collage_time = collage(model, input_data, tune_log)
    return (tvm_time, clml_time, collage_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=[
            "resnet-18",
            "resnet-34",
            "resnet-50",
            "vgg-16",
            "vgg-19",
            "densenet-121",
            "inception_v3",
            "mobilenet",
            "squeezenet_v1.0",
            "squeezenet_v1.1",
        ],
        help="The name of neural network",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--rpc-key", type=str, default="android")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        help="The data type of neural network",
    )
    parser.add_argument("--tune", type=bool, default=False)
    args = parser.parse_args()

    if args.network is None:
        networks = [
            "resnet-18",
            "resnet-34",
            "resnet-50",
            # "vgg-16",
            # "vgg-19",
            "densenet-121",
            "inception_v3",
            "mobilenet",
            "squeezenet_v1.0",
            "squeezenet_v1.1",
        ]
    else:
        networks = [args.network]

    if args.dtype is None:
        dtypes = ["float32", "float16"]
    else:
        dtypes = [args.dtype]

    results = {}
    net_results = []
    for network in networks:
        for dtype in dtypes:
            ftime = evaluate_network(network, dtype)
            results[network + "-" + dtype] = ftime
            # net_results.append([network + "-" + dtype] + list(ftime))
            # np.savetxt("results.txt", np.array(net_results), fmt="%s")

    print("----------------------------------------------------------------------")
    print(
        "%-30s %-20s %-20s %-20s"
        % ("Network Name", "TVM Opencl Time", "CLML Time", "Collage - TVM/CLML Time")
    )
    print("----------------------------------------------------------------------")
    for key, val in results.items():
        print(
            "%-30s %-20s %-20s %-20s"
            % (key, "%.2f ms" % val[0], "%.2f ms" % val[1], "%.2f ms" % val[2])
        )
