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
"""
Auto-tuning a ALU fused op on VTA
---------------------------------
"""

import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm import record

import vta
from vta.testing import simulator
from vta.top import graph_pack
import copy


#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model
def compile_network(env, target, model, start_pack, stop_pack):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with relay.build_config(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
        )

    return relay_prog, params


###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations.
# Here we use an Pynq-Z1 board as an example.

# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "resnet50_v2"
start_pack = "nn.max_pool2d"
stop_pack = "nn.global_avg_pool2d"

# Tuning option
log_file = "%s.alu.%s.log" % (device, network)
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=1),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}


def log_to_file(file_out, protocol="json"):
    """Log the tuning records into file.
    The rows of the log are stored in the format of autotvm.record.encode.
    for lhs == rhs, we add an extra rhs = [] record

    Parameters
    ----------
    file_out : str
        The file to log to.
    protocol: str, optional
        The log protocol. Can be 'json' or 'pickle'

    Returns
    -------
    callback : callable
        Callback function to do the logging.
    """

    def _callback(_, inputs, results):
        with open(file_out, "a") as f:
            for inp, result in zip(inputs, results):
                f.write(record.encode(inp, result, protocol) + "\n")

                # we only consider task with same lhs and rhs
                if inp.task.args[0] == inp.task.args[1]:
                    args = list(inp.task.args)
                    args[1] = (args[0][0], (), args[0][2])
                    inp_copy = copy.deepcopy(inp)
                    inp_copy.task.args = tuple(args)
                    f.write(record.encode(inp_copy, result, protocol) + "\n")

    return _callback


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=10,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
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
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks
def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

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

    @autotvm.template("add.vta")
    def _topi_add(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.add_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_add_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]

    @autotvm.template("multiply.vta")
    def _topi_multiply(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.multiply_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_multiply_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_and_evaluate(tuning_opt):

    if env.TARGET != "intelfocl":
        print("ALU only op only available for intelfocl target")
        return

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(
            relay.op.get("add"),
            relay.op.get("multiply"),
        ),
        target=tvm.target.Target(target, host=env.target_host),
    )

    # filter out non-packed alu task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))
    # filter out float alu task
    tasks = list(filter(lambda t: t.args[0][2] != "float32", tasks))

    # We should have extracted 10 convolution tasks
    tasks_set = {}
    print("Extracted {} alu tasks:".format(len(tasks)))
    for tsk in tasks:
        print("tsk = ", tsk)

        if len(tsk.args[1][1]) == 0:
            args = list(tsk.args)
            args[1] = args[0]
            tsk.args = tuple(args)

        if (tsk.name, tsk.args) in tasks_set:
            print("task {} already exists".format(tsk))
        tasks_set[(tsk.name, tsk.args)] = tsk

    tasks = list(tasks_set.values())
    print("After merged, final #tasks={}, tasks = {}".format(len(tasks), tasks))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)


# Run the tuning and evaluate the results
tune_and_evaluate(tuning_option)
