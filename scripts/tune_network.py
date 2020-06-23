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

"""Tune a whole neural network"""
import argparse
import logging
import random
import os
import numpy as np

import tvm
from tvm import ansor, relay
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm.contrib import util, ndk
from tvm.relay import testing
from tvm.ansor.utils import request_remote
#from baseline.utils import log_line, BenchmarkRecord

from common import str2bool
from tune_test import create_tune_option

dtype = "float32"

def get_network(name, network_path, batch_size, layout):
    """Get the relay module and random weights for a network"""
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
        layout = "NHWC"
        image_shape = (84, 84, 4)
        input_shape = (batch_size, *image_shape)
        mod, params = relay.testing.dqn.get_workload(batch_size=batch_size, image_shape=image_shape, dtype=dtype, layout=layout)
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
        tflite_model_buf = open(network_path, "rb").read()
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
        local_measure, ndk_cc, rpc_device_key, rpc_host, rpc_port, rpc_num_threads, seed=43):
    if local_measure:
        if target.target_name == "cuda":
            ctx = tvm.gpu()
        else:
            ctx = tvm.cpu()
    else:
        print("=============== Request Remote ===============")
        if 'TVM_NDK_CC' not in os.environ:
            os.environ['TVM_NDK_CC'] = ndk_cc
        remote = request_remote(rpc_device_key, rpc_host, rpc_port)

        print("=============== Export ===============")
        ctx = remote.cpu()
        temp = util.tempdir()
        path_lib = temp.relpath("deploy_lib.so")
        lib.export_library(path_lib, ndk.create_shared)

        print("=============== Upload ===============")
        remote.upload(path_lib)

        print("=============== Load ===============")
        lib = remote.load_module("deploy_lib.so")

        if rpc_num_threads:
            config_threadpool = remote.get_function('runtime.config_threadpool')
            config_threadpool(0, rpc_num_threads)

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


def tune_and_evaluate(network_arguments, target, target_host,
                      search_policy, task_scheduler_arguments, tune_option_arguments,
                      tune, debug_profile, check_correctness, log_n_lines):
    # Extract tasks from relay program
    mod, params, input_name, data_shape, out_shape = get_network(**network_arguments)

    # Tune all
    if tune:
        print("=============== Extract Workloads ===============")
        workloads, wkl_weights = ansor.extract_from_program(mod, target=target, params=params)
        print("Extract %d workloads in total" % (len(workloads)))

        # Tune workloads with auto scheduler
        print("=============== Tune ===============")
        tasks = []
        for i, wkl_key in enumerate(workloads):
            dag = ansor.workload_key_to_dag(wkl_key)
            print("[========= Task %d =========]\n" % i, dag)
            tasks.append(ansor.SearchTask(dag, wkl_key, target, target_host))

        tuner = ansor.SimpleTaskScheduler(tasks,
            lambda costs: sum(c * w for c, w in zip(costs, wkl_weights)),
            **task_scheduler_arguments)
        tune_option, measure_ctx = create_tune_option(target, **tune_option_arguments)

        if tune_option_arguments['local_measure'] and target.target_name != 'cuda':
            os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
        tuner.tune(tune_option, search_policy)

        if measure_ctx:
            del measure_ctx

    kernel_layout_rewrite = True

    # Compile graph with best states found by auto-scheduler
    print("=============== Compile ===============")
    with ansor.apply_history_best(tune_option_arguments['log_file'], log_n_lines):
        os.environ['TVM_AUTO_CACHE_FLUSH'] = "0"

        if kernel_layout_rewrite:
            ansor.prepare_layout_rewrite(mod, target=target, params=params)
        else:
            # disable layout rewrite
            ansor.LayoutRewriteLevel.BOTH_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE
            ansor.LayoutRewriteLevel.COMPUTE_REWRITE = ansor.LayoutRewriteLevel.NO_REWRITE

        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        ansor.finish_layout_rewrite()
        print("=============== Compile Finish ===============")

        module, ctx = create_module(data_shape, graph, lib, target, input_name,
                                    opt_params, debug_profile, **common_measure_parameters)

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
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        module, _ = create_module(data_shape, graph, lib, target, input_name,
                                  opt_params, debug_profile, **common_measure_parameters)
        module.run()

        expected_output = module.get_output(0).asnumpy()
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Search task related arguments
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--network-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layout", type=str, default='NHWC')
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--check-correctness", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug-profile", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)

    # Search strategy related arguments
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--policy", type=str, choices=['sketch'], default='sketch')
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='gradient',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--load-log", type=str, help="Load history log to resume the status of search")
    parser.add_argument("--log-n-lines", type=int, help="Only load the first n lines for history log")
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--early-stopping", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--rpc-device-key", type=str, default=None)
    parser.add_argument("--rpc-host", type=str, default='0.0.0.0')
    parser.add_argument("--rpc-port", type=int, default=9190)
    parser.add_argument("--rpc-num-threads", type=int, default=None)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)
    os.environ["TOPHUB_LOCATION"] = "NONE"  # disable autotvm

    target = tvm.target.create(args.target)
    log_file = args.log_file or "%s-B%d-%s.json" % (args.network, args.batch_size,
                                                    target.target_name)
    load_log_file = args.load_log or log_file
    search_policy = "%s.%s" % (args.policy, args.model_type)
    if args.layout:
        layout = args.layout
    elif target.target_name == "cuda":
        layout = "NCHW"
    else:
        layout = "NHWC"

    network_arguments = {
        'name': args.network,
        'network_path': args.network_path,
        'batch_size': args.batch_size,
        'layout': layout
    }

    task_scheduler_parameters = {
        'strategy': args.task_scheduler,
        'load_log_file': load_log_file,
        'load_model_file': args.load_model,
        'verbose': args.verbose,
    }

    common_measure_parameters = {
        'local_measure': args.local_measure,
        'rpc_device_key': args.rpc_device_key,
        'rpc_host': args.rpc_host,
        'rpc_port': args.rpc_port,
        'rpc_num_threads': args.rpc_num_threads,
        'ndk_cc': args.ndk_cc,
    }

    tune_option_arguments = {
        'log_file': log_file,
        'n_trials': args.n_trials,
        'num_measure_per_iter': args.num_measure_per_iter,
        'verbose': args.verbose,
        'n_parallel': args.n_parallel,
        'build_timeout': args.build_timeout,
        'run_timeout': args.run_timeout,
        'early_stopping': args.early_stopping,
        **common_measure_parameters
    }

    tune_and_evaluate(network_arguments, target, args.target_host,
                      search_policy, task_scheduler_parameters, tune_option_arguments,
                      args.tune, args.debug_profile, args.check_correctness,
                      args.log_n_lines)
