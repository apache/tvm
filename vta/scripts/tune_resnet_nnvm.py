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

"""Perform ResNet autoTVM tuning on VTA using NNVM."""

import argparse
import os
import time
import numpy as np

import tvm
from tvm import rpc, autotvm
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_runtime, util
from tvm.contrib.download import download

import topi
import nnvm.compiler
import vta
import vta.testing

env = vta.get_env()

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



def generate_graph(sym, params, target, target_host):
    # Populate the shape and data type dictionary
    shape_dict = {"data": (1, 3, 224, 224)}
    dtype_dict = {"data": 'float32'}
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Apply NNVM graph optimization passes
    sym = vta.graph.clean_cast(sym)
    sym = vta.graph.clean_conv_fuse(sym)
    assert env.BLOCK_IN == env.BLOCK_OUT
    sym = vta.graph.pack(sym, shape_dict, env.BATCH, env.BLOCK_OUT)

    # Compile NNVM graph
    with nnvm.compiler.build_config(opt_level=3):
        with vta.build_config():
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape_dict, dtype_dict,
                params=params, target_host=target_host)

    return graph, lib, params


def extract_tasks(sym, params, target, target_host):
    # Populate the shape and data type dictionary
    shape_dict = {"data": (1, 3, 224, 224)}
    dtype_dict = {"data": 'float32'}
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Apply NNVM graph optimization passes
    sym = vta.graph.clean_cast(sym)
    sym = vta.graph.clean_conv_fuse(sym)
    assert env.BLOCK_IN == env.BLOCK_OUT
    sym = vta.graph.pack(sym, shape_dict, env.BATCH, env.BLOCK_OUT)

    with vta.build_config():
        tasks = autotvm.task.extract_from_graph(graph=sym, shape=shape_dict, dtype=dtype_dict, target=target,
                                                params=params, symbols=(nnvm.sym.conv2d,), target_host=target_host)
    return tasks


def download_model():
    url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
    categ_fn = 'synset.txt'
    graph_fn = 'resnet18_qt8.json'
    params_fn = 'resnet18_qt8.params'
    data_dir = '_data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in [categ_fn, graph_fn, params_fn]:
        if not os.path.isfile(file):
            download(os.path.join(url, file), os.path.join(data_dir, file))

    sym = nnvm.graph.load_json(open(os.path.join(data_dir, graph_fn)).read())
    params = nnvm.compiler.load_param_dict(open(os.path.join(data_dir, params_fn), 'rb').read())

    return sym, params


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial_ = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial_,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial_, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

if __name__ == '__main__':

    # Get tracker info from env
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", None))
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    # Download model
    sym, params = download_model()

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Extract tasks
    print("Extracting tasks...")
    target = tvm.target.vta()
    target_host = env.target_host
    tasks = extract_tasks(sym, params, target, target_host)

    # Perform Autotuning
    print("Tuning...")
    tuning_opt = {
        'log_filename': 'resnet-18.log',

        'tuner': 'random',
        'n_trial': 1e9,
        'early_stopping': None,

        'measure_option':  autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
                runner=autotvm.RPCRunner(env.TARGET, tracker_host, tracker_port,
                    number=4, repeat=3, timeout=60,
                    check_correctness=True))
    }
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[tuning_opt['log_filename']]):

        # ResNet parameters
        input_shape = (1, 3, 224, 224)
        dtype = 'float32'\

        # Compile network
        print("Compiling network with best tuning parameters...")
        graph, lib, params = generate_graph(sym, params, target, target_host)
        input_shape = (1, 3, 224, 224)
        dtype = 'float32'

        # Export library
        tmp = util.tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # Upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, tracker_port, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # Upload parameters to device
        ctx = remote.context(str(target), 0)
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module = graph_runtime.create(graph, rlib, ctx)
        module.set_input('data', data_tvm)
        module.set_input(**rparams)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=3, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

