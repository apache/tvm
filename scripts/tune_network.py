"""Tune all workloads in a network"""
import argparse
import logging
import random
import os
import time
import numpy as np

import tvm
from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server
from tvm import ansor as auto_scheduler
from tvm import relay
from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server
from tvm.relay import testing
#from tvm._ffi.function import get_global_func
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger import debug_runtime
from tvm.contrib import util, ndk
from common import str2bool
from tvm.ansor import LocalRunner, LogToFile, TuneOption, SimpleTaskScheduler, \
    RPCRunner, LocalBuilder
from tvm.ansor.utils import request_remote
#from baseline.utils import log_line, BenchmarkRecord

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
        print(mod)
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
        mod, params = relay.testing.dcgan.get_workload(batch_size=batch_size, layout=layout)
    elif name == 'dqn':
        image_shape = (84, 84, 4) if layout == 'NHWC' else (4, 84, 84)
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
    elif name == 'tflite-textcnn':
        try:
            import tflite.Model
        except ImportError:
            raise ImportError("The tflite package must be installed")
        model_path = './baseline/tensorflow/fake_textcnn.tflite'
        input_name = "Placeholder"
        input_shape = (batch_size, 200, 128, 1)
        output_shape = (1, 1001)
        input_dtype = "float32"
        tflite_model_buf = open(model_path, "rb").read()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
        mod, params = relay.frontend.from_tflite(tflite_model,
                                                 shape_dict={input_name: input_shape},
                                                 dtype_dict={input_name: input_dtype})
        print(mod['main'])
    elif name == 'textcnn':
        import tensorflow as tf

        bert_pb = './baseline/tensorflow/fake_textcnn.pb'
        try:
            with tf.compat.v1.gfile.GFile(bert_pb, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        except:
            raise ValueError("Need to run ./baseline/tensorflow/bert/generate_bert_pb.py to get model first")

        input_shape = (batch_size, 200, 128, 1)
        input_name = ['Placeholder']
        shape_dict = {
            'Placeholder': input_shape
        }
        out_names = [
            'concat/concat_dim'
        ]

        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                    shape=shape_dict,
                                                    outputs=out_names)
        print(mod['main'])
    elif name == 'tdnn':
        import tensorflow as tf

        pb = './baseline/tensorflow/pruned_model_0407.pb'
        #pb = './baseline/tensorflow/tdnn_4001.pb'
        try:
            with tf.compat.v1.gfile.GFile(pb, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        except:
            raise ValueError("Need to run ./baseline/tensorflow/bert_convert.py to get model first")

        input_shape = (batch_size, 600, 64)
        input_name = ['tf_loss_fn/Placeholder']

        shape_dict = {
            'tf_loss_fn/Placeholder': input_shape,
        }
        out_names = [
            #"tf_loss_fn/ForwardPass/w2l_encoder/conv91/Conv2D"
            "tf_loss_fn/ForwardPass/Softmax"
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
                config_threadpool = get_global_func('runtime.config_threadpool')
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


def tune_and_evaluate(network_name, model_path, batch_size, target, target_host,
                      local_measure, device_key, host, port, n_parallel, ndk_cc,
                      build_timeout, run_timeout, num_threads, tune, check_correctness,
                      debug_profile, tuning_parameters, record_file, layout_set):
    joint_tuner, model_type, policy, log_file, load_log_file = (tuning_parameters['joint_tuner'],
            tuning_parameters['model_type'], tuning_parameters['policy'],
            tuning_parameters['log_file'], tuning_parameters['load_log_file'])

    if layout_set:
        layout = layout_set
    elif target.target_name == 'cuda':
        layout = 'NCHW'
    else:
        layout = "NHWC"

    # Extract workloads from relay program
    print("=============== Extract workloads ===============")
    mod, params, input_name, data_shape, out_shape = get_network(network_name, model_path, batch_size, layout)

    if tune:
        workloads, wkl_weights = auto_scheduler.extract_from_program(mod, target=target,
                params=params, ops=(relay.op.nn.dense, relay.op.nn.softmax,
                                    relay.op.nn.conv2d, relay.op.nn.conv2d_transpose,
                                    relay.op.nn.max_pool2d, relay.op.nn.avg_pool2d,
                                    relay.op.nn.global_max_pool2d, relay.op.nn.global_avg_pool2d,
                                    relay.op.nn.conv3d, relay.op.nn.adaptive_avg_pool3d,
                                    relay.op.nn.batch_matmul, relay.op.mean,
                                    ))
        print("Total workload number: %d" % (len(workloads)))
        #workloads = workloads[1:2]
        #wkl_weights = wkl_weights[1:2]
        #workloads = ['["2543426b0070d4a379a1f75a362a5f1b"]']


        # Tune workloads with auto scheduler
        print("=============== Tuning ===============")
        tasks = []
        for i, wkl_key in enumerate(workloads):
            dag = auto_scheduler.workload_key_to_dag(wkl_key)
            print("[========= Task %d =========]\n" % i, dag)
            tasks.append(auto_scheduler.SearchTask(dag, wkl_key, target, target_host))

        if joint_tuner != 'rl':
            tuner = SimpleTaskScheduler(tasks, load_log_file=load_log_file)
        elif joint_tuner == 'rl':
            # put import here to remove pytorch dependency
            from tvm.auto_scheduler.joint_tuner.rl_joint_tuner import RLJointTuner
            tuner = RLJointTuner(tasks, weights=wkl_weights, load_log_file=load_log_file)
        else:
            raise ValueError("Invalid joint tuner: " + joint_tuner)

        if local_measure:
            builder = LocalBuilder(timeout=build_timeout)
            if target.target_name == "cuda":
                ctx = tvm.context("cuda", 0)
                cuda_arch = "sm_" + "".join(ctx.compute_version.split('.'))
                tvm.autotvm.measure.measure_methods.set_cuda_target_arch(cuda_arch)

                tracker = Tracker('0.0.0.0', port=port, port_end=10000, silent=True)
                if device_key is None:
                    device_key = '$local$device$%d' % tracker.port
                server = Server('0.0.0.0', port=tracker.port, port_end=10000,
                                key=device_key, use_popen=True, silent=True,
                                tracker_addr=(tracker.host, tracker.port))
                runner = RPCRunner(device_key, host=host, port=tracker.port,
                                repeat=1, min_repeat_ms=400,
                                n_parallel=n_parallel)
            else:
                os.environ['TVM_AUTO_CACHE_FLUSH'] = "1"
                runner = LocalRunner(repeat=10, number=1, min_repeat_ms=0, timeout=run_timeout)
        else:
            os.environ['TVM_NDK_CC'] = ndk_cc
            builder = LocalBuilder(build_func='ndk', timeout=build_timeout)
            runner = RPCRunner(device_key, host=host, port=port,
                               repeat=1, min_repeat_ms=400,
                               n_parallel=n_parallel, timeout=run_timeout)

        search_policy = "%s.%s" % (policy, model_type)
        tune_option = TuneOption(n_trials=tuning_parameters['n_trials'],
                                 early_stopping=tuning_parameters['early_stopping'],
                                 num_measure_per_iter=tuning_parameters['num_measure_per_iter'],
                                 builder=builder,
                                 verbose=tuning_parameters['verbose'],
                                 runner=runner,
                                 measure_callbacks=[LogToFile(log_file)])
        if local_measure and target.target_name != 'cuda':
            os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
            tuner.tune(tune_option, search_policy)
        else:
            tuner.tune(tune_option, search_policy)

    kernel_layout_rewrite =  False

    # Compile graph with best states found by auto-scheduler
    print("=============== Compile ===============")
    with auto_scheduler.apply_history_best(log_file, args.log_n_lines):
    #if True:
    #with auto_scheduler.BlockingEmptyContext():
        os.environ['TVM_AUTO_CACHE_FLUSH'] = "0"
        os.environ['TVM_BIND_MASTER_CORE_0'] = "1"
        if kernel_layout_rewrite:
            auto_scheduler.prepare_layout_rewrite(mod, target=target,
                                                  params=params,
                                                  ops=(relay.op.nn.dense, relay.op.nn.conv2d, relay.op.nn.conv3d))
        else:
            # disable layout rewrite
            auto_scheduler.LayoutRewriteLevel.BOTH_REWRITE = auto_scheduler.LayoutRewriteLevel.NO_REWRITE
            auto_scheduler.LayoutRewriteLevel.COMPUTE_REWRITE = auto_scheduler.LayoutRewriteLevel.NO_REWRITE

        with relay.build_config(opt_level=3):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)
        '''
        from tvm.relay.backend import graph_runtime_codegen
        with relay.build_config(opt_level=3):
            opt_mod, _ = relay.optimize(mod, target, params)
            grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
            grc.codegen(opt_mod["main"])
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)
        '''
        auto_scheduler.finish_layout_rewrite()
        print("=============== Compile Finish ===============")

        module, ctx = create_module(data_shape, graph, lib, target, input_name, opt_params,
                                    debug_profile, local_measure, ndk_cc,
                                    device_key, host, port, run_timeout, num_threads)

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
        auto_scheduler.LayoutRewriteLevel.BOTH_REWRITE = auto_scheduler.LayoutRewriteLevel.NO_REWRITE
        auto_scheduler.LayoutRewriteLevel.COMPUTE_REWRITE = auto_scheduler.LayoutRewriteLevel.NO_REWRITE
        target = tvm.target.create('llvm')
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, opt_params = relay.build_module.build(
                mod, target=target, params=params)

        module, _ = create_module(data_shape, graph, lib, target, input_name, opt_params,
                                  debug_profile, local_measure, ndk_cc,
                                  device_key, host, port, run_timeout, num_threads)
        module.run()

        expected_output = module.get_output(0).asnumpy()
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None, help="The path of tflite model")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--policy", type=str, choices=['multi-stage', 'meta-rewrite'],
            default='meta-rewrite')
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--check-correctness", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--debug-profile", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--log-file", type=str, help="Write log of measurement results to this file")
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--load-model", action='store_true')
    parser.add_argument("--model-file", type=str, default='saved_model.xgb')
    parser.add_argument("--load-log", type=str, help="Load history log for pre-training the cost model")
    parser.add_argument("--out-file", type=str, default='results.tsv')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--joint-tuner", type=str, default='bottleneck-decay', help='The type of joint tuner',
                        choices=['no', 'uniform', 'weighted', 'bottleneck', 'bottleneck-decay', 'sequential', 'round-robin', 'rl'])
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--device-key", type=str, default=None)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--layout", type=str, default=None)
    parser.add_argument("--log-n-lines", type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.basicConfig()
    logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)

    target = tvm.target.create(args.target)

    tuning_parameters = {
        'n_trials': args.n_trials,
        'num_measure_per_iter': args.num_measure_per_iter,
        'log_file': args.log_file if args.log_file else "%s-B%d.json" % (args.network, args.batch_size),
        'model_type': args.model_type,
        'joint_tuner': args.joint_tuner,
        'policy': args.policy,
        'early_stopping': -1,
        'verbose': 1,
    }
    tuning_parameters['load_log_file'] = args.load_log or tuning_parameters['log_file']

    os.environ["TOPHUB_LOCATION"] = "NONE"
    tune_and_evaluate(args.network, args.model_path, args.batch_size, target, args.target_host,
                      args.local_measure, args.device_key, args.host,
                      args.port, args.n_parallel, args.ndk_cc, args.build_timeout,
                      args.run_timeout, args.num_threads, args.tune, args.check_correctness,
                      args.debug_profile, tuning_parameters, args.out_file, args.layout)
