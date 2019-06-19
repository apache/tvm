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

"""Perform inference on VTA using Relay."""

import argparse, os, time
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

import topi
import tvm
from tvm import rpc, autotvm, relay
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
import vta
from vta.testing import simulator
from vta.top import graph_pack
from tvm.autotvm.task import extract_from_program

def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--model', type=str, default='resnet18_v1', choices=['resnet18_v1'],
                        help='Input model name.')
    parser.add_argument('--start-name', type=str, default='nn.max_pool2d',
                        help='The name of the node where packing starts')
    parser.add_argument('--stop-name', type=str, default='nn.global_avg_pool2d',
                        help='The name of the node where packing stops')
    parser.add_argument('--debug-profile', action='store_true',
                        help='Show layer-wise time cost profiling results')
    parser.add_argument('--device', default='vta',  choices=['vta', 'arm_cpu'],
                        help='Select device target')
    parser.add_argument('--measurements', type=int, default=1,
                        help='Number of measurements during AutoTVM search')
    parser.add_argument('--tuner', type=str, default="random",
                        help='AutoTVM search strategy')
    parser.add_argument('--log-filename', type=str, default="resnet-18.log",
                        help='AutoTVM log file name')

    return parser.parse_args()


def register_vta_tuning_tasks():
    from tvm.autotvm.task.topi_integration import TaskExtractEnv, deserialize_args

    @tvm.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.const(a_min, x.dtype)
        const_max = tvm.const(a_max, x.dtype)
        x = tvm.compute(x.shape, lambda *i: tvm.min(x(*i), const_max), name="clipA")
        x = tvm.compute(x.shape, lambda *i: tvm.max(x(*i), const_min), name="clipB")
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

        if tvm.target.current_target().device_name == 'vta':
            s = topi.generic.schedule_conv2d_nchw([res])
        else:
            s = tvm.create_schedule([res.op])
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

        if tvm.target.current_target().device_name == 'vta':
            s = topi.generic.schedule_dense([res])
        else:
            s = tvm.create_schedule([res.op])

        return s, [A, W, res]


def compile_network(opt, env, target):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": 'float32'}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(opt.model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    with relay.quantize.qconfig(global_scale=8.0, skip_k_conv=1):
        relay_prog = relay.quantize.quantize(mod[mod.entry_func], params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            relay_prog,
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=opt.start_name,
            stop_name=opt.stop_name)
        relay_prog = relay.ir_pass.fold_constant(relay_prog)

    return relay_prog, params

if __name__ == '__main__':

    opt = parse_arguments()

    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")

    # Read in VTA environment
    env = vta.get_env()

    # Get remote from fleet node
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", None))
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    # Get remote
    if env.TARGET != "sim":

        # Measure build start time
        reconfig_start = time.time()

        # Get remote from fleet node
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, tracker_port, timeout=10000)

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
    
    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Compile Relay program
    relay_prog, params = compile_network(opt, env, target)

    # Perform task extraction on Relay program
    tasks = extract_from_program(func=relay_prog,
                                 params=params,
                                 ops=(tvm.relay.op.nn.conv2d,),
                                 target=target,
                                 target_host=env.target_host)
    
    # Check that we have extracted the right number of tasks
    assert opt.model == "resnet18_v1" and len(tasks) == 10

    print("Task extraction passed!")
