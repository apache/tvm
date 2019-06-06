"""Perform inference on VTA using Relay."""

import argparse, os
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
    parser.add_argument('--model', type=str, required=False, default='resnet18_v1',
                        help='Input model name.')
    parser.add_argument('--start-name', type=str, default='nn.max_pool2d',
                        help='The name of the node where packing starts')
    parser.add_argument('--stop-name', type=str, default='nn.global_avg_pool2d',
                        help='The name of the node where packing stops')
    parser.add_argument('--debug-profile', action='store_true',
                        help='Show layer-wise time cost profiling results')
    parser.add_argument('--device', default="vta",
                        help='Select device target, either "vta" or "vtacpu"')
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


def compile_network(opt, env, target):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": 'float32'}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(opt.model, pretrained=True)
    relay_prog, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    with relay.quantize.qconfig(global_scale=8.0, skip_k_conv=1):
        relay_prog = relay.quantize.quantize(relay_prog, params=params)

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

    opt = parse_arguments()

    # Make sure that TVM was compiled with RPC=1
    assert tvm.module.enabled("rpc")

    # Read in VTA environment
    env = vta.get_env()

    # VTA target
    target = tvm.target.vta()

    # Get tracker info from env
    tracket_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracket_port = int(os.environ.get("TVM_TRACKER_PORT", None))
    if not tracket_host or not tracket_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()
    
    # Compile Relay program
    print("Initial compile...")
    relay_prog, params = compile_network(opt, env, target)

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extracting tasks...")
    tasks = extract_from_program(func=relay_prog,
                                 params=params,
                                 ops=(tvm.relay.op.nn.conv2d,),
                                 target=target,
                                 target_host=env.target_host)

    # Perform Autotuning
    print("Tuning...")
    tuning_opt = {
        'log_filename': opt.log_filename,
        'tuner': opt.tuner,
        'n_trial': 1e9,
        'early_stopping': None,
        'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func=vta.vta_autotvm_build_func),
                runner=autotvm.RPCRunner(env.TARGET, tracket_host, tracket_port,
                    number=4, min_repeat_ms=150, repeat=opt.measurements, timeout=60,
                    check_correctness=True))
    }
    tune_tasks(tasks, **tuning_opt)

    # Compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[opt.log_filename]):

        # ResNet parameters
        input_shape = (1, 3, 224, 224)
        dtype = 'float32'

        # Compile network
        print("Compiling network with best tuning parameters...")
        relay_prog, params = compile_network(opt, env, target)
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=env.target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=env.target_host)

        # Export library
        tmp = util.tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # Upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(env.TARGET, tracket_host, tracket_port, timeout=10000)
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
        ftimer = module.module.time_evaluator("run", ctx, number=4, repeat=opt.measurements)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))
