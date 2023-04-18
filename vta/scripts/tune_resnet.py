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

"""Perform ResNet autoTVM tuning on VTA using Relay."""

import argparse, os, time
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.debugger import debug_executor
import vta
from vta.testing import simulator
from vta.top import graph_pack
from tvm.autotvm.task import extract_from_program


def parse_arguments():

    parser = argparse.ArgumentParser(description="Train a model for image classification.")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18_v1",
        choices=["resnet18_v1"],
        help="Input model name.",
    )
    parser.add_argument(
        "--start-name",
        type=str,
        default="nn.max_pool2d",
        help="The name of the node where packing starts",
    )
    parser.add_argument(
        "--stop-name",
        type=str,
        default="nn.global_avg_pool2d",
        help="The name of the node where packing stops",
    )
    parser.add_argument(
        "--debug-profile", action="store_true", help="Show layer-wise time cost profiling results"
    )
    parser.add_argument(
        "--device", default="vta", choices=["vta", "arm_cpu"], help="Select device target"
    )
    parser.add_argument(
        "--measurements", type=int, default=1, help="Number of measurements during AutoTVM search"
    )
    parser.add_argument("--tuner", type=str, default="random", help="AutoTVM search strategy")
    parser.add_argument(
        "--log-filename", type=str, default="resnet-18.log", help="AutoTVM log file name"
    )

    return parser.parse_args()


def register_vta_tuning_tasks():
    from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.task.register("topi_nn_conv2d", override=True)
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        args = deserialize_args(args)
        A, W = args[:2]

        with tvm.target.vta():
            res = topi.nn.conv2d(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = topi.generic.schedule_conv2d_nchw([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]

    @autotvm.task.register("topi_nn_dense", override=True)
    def _topi_nn_dense(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        args = deserialize_args(args)
        A, W = args[:2]

        with tvm.target.vta():
            res = topi.nn.dense(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = topi.generic.schedule_dense([res])
        else:
            s = te.create_schedule([res.op])

        return s, [A, W, res]


def compile_network(opt, env, target):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(opt.model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            relay_prog = relay.quantize.quantize(mod["main"], params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            relay_prog,
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=opt.start_name,
            stop_name=opt.stop_name,
        )

    return relay_prog, params


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
    try_winograd=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
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

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial_ = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial_,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial_, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


if __name__ == "__main__":

    opt = parse_arguments()

    # Make sure that TVM was compiled with RPC=1
    assert tvm.runtime.enabled("rpc")

    # Read in VTA environment
    env = vta.get_env()

    # Get remote from fleet node
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    # Get remote
    if env.TARGET != "sim":

        # Measure build start time
        reconfig_start = time.time()

        # Get remote from fleet node
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

        # Reconfigure the JIT runtime and FPGA.
        # You can program the FPGA with your own custom bitstream
        # by passing the path to the bitstream file instead of None.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)

        # Report on reconfiguration time
        reconfig_time = time.time() - reconfig_start
        print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

    # In simulation mode, host the RPC server locally.
    else:
        remote = rpc.LocalSession()

    # VTA target and execution context
    target = env.target if opt.device == "vta" else env.target_vta_cpu
    ctx = remote.ext_dev(0) if opt.device == "vta" else remote.cpu(0)

    # Compile Relay program
    print("Initial compile...")
    relay_prog, params = compile_network(opt, env, target)

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extracting tasks...")
    tasks = extract_from_program(
        func=relay_prog,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=tvm.target.Target(target, host=env.target_host),
    )

    # Perform Autotuning
    print("Tuning...")
    tuning_opt = {
        "log_filename": opt.log_filename,
        "tuner": opt.tuner,
        "n_trial": 1e9,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
            runner=autotvm.RPCRunner(
                env.TARGET,
                tracker_host,
                tracker_port,
                number=4,
                min_repeat_ms=150,
                repeat=opt.measurements,
                timeout=60,
                # check_correctness=True, # TODO: re-enable when check_correctness works again.
            ),
        ),
    }
    tune_tasks(tasks, **tuning_opt)

    # Compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[opt.log_filename]):

        # Compile network
        print("Compiling network with best tuning parameters...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                graph, lib, params = relay.build(
                    relay_prog,
                    target=tvm.target.Target(target, host=env.target_host),
                    params=params,
                )
        else:
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                graph, lib, params = relay.build(
                    relay_prog,
                    target=tvm.target.Target(target, host=env.target_host),
                    params=params,
                )

        # Export library
        temp = utils.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # If detailed runtime info is needed build with debug runtime
        if opt.debug_profile:
            m = debug_executor.create(graph, lib, ctx)
        else:
            m = graph_executor.create(graph, lib, ctx)

        # Set the network parameters and synthetic input
        image = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        m.set_input(**params)
        m.set_input("data", image)

        # Perform inference
        timer = m.module.time_evaluator("run", ctx, number=4, repeat=opt.measurements)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )

        # Display profile information
        if opt.debug_profile:
            m.run()
