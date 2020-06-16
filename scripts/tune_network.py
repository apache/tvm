"""Tune all workloads in a network"""
import argparse
import logging
import random
import os
import time
import numpy as np

import tvm
from tvm import _ffi, ansor, relay
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm.contrib import util, ndk
from tvm.relay import testing
from tvm.ansor.utils import request_remote
#from baseline.utils import log_line, BenchmarkRecord

from common import str2bool
from tune_test import create_tune_option

dtype = "float32"

def get_network(name, model_path, batch_size, layout):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    input_name = 'data'

    if name.startswith("resnet3d"):
        n_layer = int(name.split('-')[1])
        layout = "NDHWC"
        image_shape = (16, 112, 112, 3)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.resnet3d.get_workload(num_layers=n_layer, batch_size=batch_size, image_shape=image_shape, dtype=dtype, layout=layout)
    elif name.startswith("resnet"):
        n_layer = int(name.split('-')[1])
        image_shape = (224, 224, 3) if layout == 'NHWC' else (3, 224, 224)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, layout=layout, image_shape=image_shape, dtype=dtype)
    elif "lstm" in name:
        mod, params = relay.testing.lstm.get_workload(iterations=10, num_hidden=512, batch_size=batch_size, dtype=dtype)
    elif "mlp" in name:
        input_shape = (batch_size, 1, 28, 28)
        mod, params = relay.testing.mlp.get_workload(batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'dcgan':
        input_shape = (batch_size, 100)
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size)
    elif name == 'dqn':
        image_shape = (4, 84, 84)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.dqn.get_workload(batch_size=batch_size, image_shape=image_shape, dtype=dtype)
    elif name == 'mobilenet':
        image_shape = (224, 224, 3) if layout == 'NHWC' else (3, 224, 224)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, layout=layout, image_shape=image_shape, dtype=dtype)
    elif name == 'r3d_18':
        import torch
        import torchvision

        model = getattr(torchvision.models.video, name)(pretrained=False)
        model = model.eval()

        # We grab the TorchScripted model via tracing
        input_shape = [batch_size, 3, 16, 112, 112]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = 'input0'  # only one input, set it to this name
        shape_list = {input_name: input_shape}
        mod, params = relay.frontend.from_pytorch(scripted_model,
                                                  shape_list)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"input_name": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = relay.Module.from_expr(net)
    elif name == 'tflite-mobilenet-v2' or name == 'tflite-resnet-v2-50':
        try:
            import tflite.Model
        except ImportError:
            raise ImportError("The tflite package must be installed")
        input_name = "input"
        input_shape = (1, 224, 224, 3)
        output_shape = (1, 1001)
        input_dtype = "float32"
        tflite_model_buf = open(model_path, "rb").read()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: input_dtype})
    elif name == 'pytorch-mobilenet-v2':
        import torch

        model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=False)
        model.eval()

        input_shape = [batch_size, 3, 224, 224]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = 'input0'
        shape_list = {input_name: input_shape}
        mod, params = relay.frontend.from_pytorch(scripted_model,
                                                  shape_list)
    elif name == 'bert':
        import tensorflow as tf

        bert_pb = './baseline/tensorflow/tf_models/bert/bert-B%d.pb' % batch_size
        try:
            with tf.compat.v1.gfile.GFile(bert_pb, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        except:
            raise ValueError("Need to run ./baseline/tensorflow/bert/generate_bert_pb.py to get model first")

        input_shape = (batch_size, 128)
        input_name = ['input']
        shape_dict = {
            'input': input_shape
        }
        out_names = [
            'bert/pooler/dense/Tanh'
        ]

        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                    shape=shape_dict,
                                                    outputs=out_names)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, output_shape


def create_module(data_shape, graph, lib, target, input_name, params, debug_profile,
        local_measure, ndk_cc, device_key, host, port, run_timeout, num_threads, seed=43):
    # Upload parameters to device
    if local_measure:
        if target.target_name == "cuda":
            ctx = tvm.gpu()
        else:
            ctx = tvm.cpu()
            if num_threads:
                config_threadpool = _ffi.get_global_func('runtime.config_threadpool')
                config_threadpool(0, num_threads)
    else:
        print("=============== Request Remote ===============")
        if 'TVM_NDK_CC' not in os.environ:
            os.environ['TVM_NDK_CC'] = ndk_cc
        remote = request_remote(device_key, host, port, timeout=run_timeout)

        print("=============== Export ===============")
        ctx = remote.cpu()
        temp = util.tempdir()
        path_lib = temp.relpath("deploy_lib.so")
        lib.export_library(path_lib, ndk.create_shared)

        print("=============== Upload ===============")
        remote.upload(path_lib)

        print("=============== Load ===============")
        lib = remote.load_module("deploy_lib.so")
        if num_threads:
            config_threadpool = remote.get_function('runtime.config_threadpool')
            config_threadpool(0, num_threads)

    np.random.seed(seed)
    data_tvm = tvm.nd.array(100 * (np.random.uniform(size=data_shape)).astype(dtype), ctx=ctx)
    if debug_profile:
        module = debug_runtime.create(graph, lib, ctx)
    else:
        module = runtime.create(graph, lib, ctx)
    if type(input_name) == list:
        for name in input_name:
            module.set_input(name, data_tvm)
    else:
        module.set_input(input_name, data_tvm)
    for k, v in params.items():
        module.set_input(k, v)

    return module, ctx


def tune_and_evaluate(target, target_host, log_n_lines, search_policy, tune,
                      debug_profile, check_correctness, network_parameters,
                      task_scheduler_parameters, tune_parameters, module_parameters):
    # Extract workloads from relay program
    mod, params, input_name, data_shape, out_shape = get_network(**network_parameters)

    if tune:
        print("=============== Extracting workloads ===============")
        workloads, wkl_weights = ansor.extract_from_program(mod, target=target,
                params=params, ops=(relay.op.nn.dense, relay.op.nn.softmax,
                                    relay.op.nn.conv2d, relay.op.nn.conv2d_transpose,
                                    relay.op.nn.max_pool2d, relay.op.nn.avg_pool2d,
                                    relay.op.nn.global_max_pool2d, relay.op.nn.global_avg_pool2d,
                                    relay.op.nn.conv3d, relay.op.nn.adaptive_avg_pool3d,
                                    relay.op.nn.batch_matmul, relay.op.mean,
                                    ))
        print("Totally %d workload extracted." % (len(workloads)))

        # Tune workloads with auto scheduler
        print("=============== Tuning ===============")
        tasks = []
        for i, wkl_key in enumerate(workloads):
            dag = ansor.workload_key_to_dag(wkl_key)
            print("[========= Task %d =========]\n" % i, dag)
            tasks.append(ansor.SearchTask(dag, wkl_key, target, target_host))

        tuner = ansor.SimpleTaskScheduler(tasks,
                lambda costs: sum(c * w for c, w in zip(costs, wkl_weights)),
                **task_scheduler_parameters)
        tune_option, measure_ctx = create_tune_option(target, **tune_parameters)

        if tune_parameters['local_measure'] and target.target_name != 'cuda':
            os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
        tuner.tune(tune_option, search_policy)

        if measure_ctx:
            del measure_ctx

    kernel_layout_rewrite =  False

    # Compile graph with best states found by auto-scheduler
    print("=============== Compile ===============")
    with ansor.apply_history_best(tune_parameters['log_file'], log_n_lines):
        os.environ['TVM_AUTO_CACHE_FLUSH'] = "0"
        os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
        if kernel_layout_rewrite:
            ansor.prepare_layout_rewrite(mod, target=target,
                                         params=params,
                                         ops=(relay.op.nn.dense, relay.op.nn.conv2d, relay.op.nn.conv3d))
        else:
            # disable layout rewrite
            ansor.LayoutRewriteLevel.BOTH_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE
            ansor.LayoutRewriteLevel.COMPUTE_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE

        with relay.build_config(opt_level=3):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        ansor.finish_layout_rewrite()
        print("=============== Compile Finish ===============")

        module, ctx = create_module(data_shape, graph, lib, target, input_name,
                                    opt_params, debug_profile, **module_parameters)

        # Evaluate
        print("========== Evaluate ==========")
        ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=3)
        prof_res = np.array(ftimer().results)
        # display profile information
        if debug_profile or check_correctness:
            module.run()
            if check_correctness:
                actual_output = module.get_output(0).asnumpy()
                print(actual_output)

        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res) * 1000, np.std(prof_res) * 1000))
        #log_line(BenchmarkRecord(target.target_name, 'gpu' if target.target_name == 'cuda' else 'cpu', 'network',
        #                         "%s.B%d" % (network_name, batch_size), 'AutoSchedule', layout,
        #                         {"costs": prof_res}, time.time()), record_file)

    if check_correctness:
        print("========== Check Correctness ==========")
        # clean relay cache
        relay.backend.compile_engine.get().clear()

        # disable layout rewrite
        ansor.LayoutRewriteLevel.BOTH_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE
        ansor.LayoutRewriteLevel.COMPUTE_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE
        target = tvm.target.create('llvm')
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        module, _ = create_module(data_shape, graph, lib, target, input_name,
                                  opt_params, debug_profile, **module_parameters)
        module.run()

        expected_output = module.get_output(0).asnumpy()
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task related options
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--check-correctness", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug-profile", type=str2bool, nargs='?', const=True, default=False)

    # Strategy related options
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--policy", type=str, choices=['multi-stage', 'meta-rewrite'],
                        default='meta-rewrite')
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='gradient',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')

    # File related options
    parser.add_argument("--log-file", type=str, help="Write log of measurement results to this file")
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")
    parser.add_argument("--load-log", type=str, help="Load history log for pre-training the cost model")
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--log-n-lines", type=int)

    # Detailed control options
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--early-stopping", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--device-key", type=str, default=None)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)

    target = tvm.target.create(args.target)
    log_file =  args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                     target.target_name)
    load_log_file = args.load_log or log_file
    search_policy = "%s.%s" % (args.policy, args.model_type)
    if args.layout:
        layout = args.layout
    elif target.target_name == "cuda":
        layout = "NCHW"
    else:
        layout = "NHWC"

    network_parameters = {
        'name': args.network,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'layout': layout
    }

    task_scheduler_parameters = {
        'strategy': args.task_scheduler,
        'load_log_file': load_log_file,
        'load_model_file': args.load_model,
        'verbose': args.verbose,
    }

    control_parameters = {
        'local_measure': args.local_measure,
        'device_key': args.device_key,
        'host': args.host,
        'port': args.port,
        'ndk_cc': args.ndk_cc,
    }

    tune_parameters = {
        'log_file': log_file,
        'n_trials': args.n_trials,
        'num_measure_per_iter': args.num_measure_per_iter,
        'verbose': args.verbose,
        'n_parallel': args.n_parallel,
        'build_timeout': args.build_timeout,
        'run_timeout': args.run_timeout,
        'early_stopping': args.early_stopping,
        **control_parameters
    }

    module_parameters = {
        'run_timeout': args.run_timeout,
        'num_threads': args.num_threads,
        **control_parameters
    }

    os.environ["TOPHUB_LOCATION"] = "NONE"
    tune_and_evaluate(target, args.target_host, args.log_n_lines, search_policy,
                      args.tune, args.debug_profile, args.check_correctness,
                      network_parameters, task_scheduler_parameters, tune_parameters,
                      module_parameters)
