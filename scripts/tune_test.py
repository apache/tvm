"""Use auto scheduler to tune workloads"""
import argparse
import logging
import os
import random

import numpy as np

import tvm
from tvm import ansor
from tvm.ansor.utils import request_remote

from common import get_workload_keys, get_workload_weights, measure_schedule, str2bool


def make_cost_model(model_type, load_model_file, load_log_file):
    if model_type == 'xgb':
        model = ansor.XGBModel()
        if load_model_file:
            print("Load pretrained model...")
            model.load(load_model_file)
        elif load_log_file:
            model.load_log_file(load_log_file)
    elif model_type == "random":
        model = ansor.RandomModel()
    else:
        raise ValueError("Invalid model: " + model_type)
    return model


def tune_workload(wkl_key, target, target_host, n_trials, num_measure_per_iter,
                  policy, log_file, verbose,
                  model_type, load_model_file, load_log_file,
                  build_timeout, local_measure=True, device_key=None, host="0.0.0.0",
                  port=9190, n_parallel=1, ndk_cc=None, remeasure=True):
    """Tune a workload"""

    if False:
        # Debug info. Print static analysis results from the access analyzer
        dag = auto_scheduler.workload_key_to_dag(wkl_key)
        print(dag.access_analyzer)
        exit()

    model = make_cost_model(model_type, load_model_file, load_log_file)

    if policy == 'meta-rewrite':
        policy = ansor.MetaTileRewritePolicy(program_cost_model=model)
    elif policy == 'beam-search':
        policy = ansor.MetaTileRewritePolicy(program_cost_model=model,
                                       params={'use_beam_search': 1})
    else:
        raise ValueError("Invalid search policy: " + policy)

    if local_measure:
        builder = ansor.LocalBuilder(build_timeout)
        if target.target_name == "cuda":
            measure_ctx = ansor.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400)
            runner = measure_ctx.runner
        else:
            runner = ansor.LocalRunner(repeat=1, min_repeat_ms=400)
    else:
        os.environ['TVM_NDK_CC'] = ndk_cc
        builder = ansor.LocalBuilder(build_timeout, build_func='ndk')
        runner = ansor.RPCRunner(device_key, host=host, port=port,
                                 repeat=1, min_repeat_ms=400,
                                  n_parallel=n_parallel)

    tune_option = ansor.TuneOption(n_trials=n_trials,
                                   num_measure_per_iter=num_measure_per_iter,
                                   verbose=verbose,
                                   builder=builder,
                                   runner=runner,
                                   callbacks=[ansor.LogToFile(log_file)])
    s, bufs = ansor.auto_schedule(wkl_key,
                                  target=target, target_host=target_host,
                                  search_policy=policy,
                                  tune_option=tune_option)

    if remeasure:
        print("Found schedule:")
        print(tvm.lower(s, bufs, simple_mode=True))
        print("Redo measurement for double check...")
        if local_measure:
            remote = None
        else:
            remote = request_remote(device_key, host, port, 1)
        cost = np.mean((measure_schedule(s, bufs, target, remote=remote, ndk_cc=ndk_cc)))
        print("Best schedule: %.2f GFLOPS\tcost: %.3f ms" %
              (ansor.ComputeDAG(bufs).flop_ct / cost / 1e9, cost * 1e3))


def tune_workloads_jointly(wkl_keys, weights, joint_tuner, target, target_host,
                           n_trials, num_measure_per_iter,
                           search_policy, log_file, verbose,
                           model_type, load_model_file, load_log_file,
                           build_timeout, local_measure=True, device_key=None,
                           host="0.0.0.0", port=9190, n_parallel=1, ndk_cc=None):
    """Tune for multiple workloads jointly"""
    if local_measure:
        builder = ansor.LocalBuilder(timeout=build_timeout)
        if target.target_name == "cuda":
            measure_ctx = ansor.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400)
            runner = measure_ctx.runner
        else:
            runner = ansor.LocalRunner(repeat=1, min_repeat_ms=400)
    else:
        os.environ['TVM_NDK_CC'] = ndk_cc
        builder = ansor.LocalBuilder(build_func='ndk', timeout=build_timeout)
        runner = ansor.RPCRunner(device_key, host=host, port=port,
                                 repeat=1, min_repeat_ms=400,
                                 n_parallel=n_parallel)

    tasks = []
    for wkl_key in wkl_keys:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target, target_host))

    def objective_func(costs):
        return sum(c * w for c, w in zip(costs, weights))

    tuner = ansor.SimpleTaskScheduler(tasks, objective_func, strategy=joint_tuner,
                                      load_log_file=load_log_file, load_model_file=load_model_file)

    search_policy = "%s.%s" % (search_policy, model_type)
    tune_option = ansor.TuneOption(n_trials=n_trials,
                                   num_measure_per_iter=num_measure_per_iter,
                                   builder=builder,
                                   verbose=verbose,
                                   runner=runner,
                                   callbacks=[ansor.LogToFile(log_file)])
    tuner.tune(tune_option, search_policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wkl", type=str, required=True)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--policy", type=str, choices=['meta-rewrite', 'beam-search'], default='meta-rewrite')
    parser.add_argument("--log-file", type=str, help="Write log of measurement results to this file")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--load-model", type=str)
    parser.add_argument("--load-log", type=str, help="Load history log for pre-training the cost model")
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--task-scheduler", type=str, default='no',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--device-key", type=str, default=None)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    logging.basicConfig()
    logging.getLogger('auto_scheduler').setLevel(logging.DEBUG)

    log_file = args.log_file or args.wkl + ".json"
    load_log_file = args.load_log or log_file

    target = tvm.target.create(args.target)
    wkl_keys = get_workload_keys(args.wkl)
    weights = get_workload_weights(args.wkl)
    if args.task_scheduler == 'no':
        # tune workloads one by one
        for wkl_key in wkl_keys:
            tune_workload(wkl_key, target, args.target_host, args.n_trials,
                          args.num_measure_per_iter,
                          args.policy, log_file, args.verbose,
                          args.model_type, args.load_model, load_log_file,
                          args.build_timeout,
                          args.local_measure, args.device_key, args.host,
                          args.port, args.n_parallel, args.ndk_cc,
                          remeasure=len(wkl_keys) == 1)
    else:
        # tune workloads jointly using JointTuner
        tune_workloads_jointly(wkl_keys, weights, args.joint_tuner,
                               target, args.target_host,
                               args.n_trials, args.num_measure_per_iter,
                               args.policy, log_file, args.verbose,
                               args.model_type, args.load_model, args.load_log,
                               args.build_timeout,
                               args.local_measure, args.device_key, args.host,
                               args.port, args.n_parallel, args.ndk_cc)

