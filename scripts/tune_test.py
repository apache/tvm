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

def create_tune_option(target, log_file, n_trials, num_measure_per_iter, verbose,
                      n_parallel, build_timeout, local_measure, device_key, host,
                      port, ndk_cc, early_stopping=-1, run_timeout=10):
    builder = runner = measure_ctx = None
    if local_measure:
        builder = ansor.LocalBuilder(timeout=build_timeout)
        if target.target_name == "cuda":
            measure_ctx = ansor.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400)
            runner = measure_ctx.runner
        else:
            runner = ansor.LocalRunner(repeat=1, min_repeat_ms=400)
    else:
        os.environ['TVM_NDK_CC'] = ndk_cc
        builder = ansor.LocalBuilder(timeout=build_timeout, build_func='ndk')
        runner = ansor.RPCRunner(key=device_key, host=host, port=port, timeout=run_timeout,
                                 n_parallel=n_parallel, repeat=1, min_repeat_ms=400)

    tune_option = ansor.TuneOption(n_trials=n_trials, early_stopping=early_stopping,
                                   num_measure_per_iter=num_measure_per_iter,
                                   verbose=verbose,
                                   builder=builder,
                                   runner=runner,
                                   measure_callbacks=[ansor.LogToFile(log_file)],
                                   pre_search_callbacks=[ansor.PreLoadMeasuredStates(log_file)])

    return tune_option, measure_ctx


def replay_workload(wkl_key, target, target_host, log_file,
                    local_measure=True, device_key=None, host="0.0.0.0",
                    port=9190, ndk_cc=None, show_lower_result=True):
    cost = gflops = None

    inp, res = ansor.best_measure_pair_in_file(log_file, wkl_key, target)
    if inp is None:
        print("Cannot find log for: %s" % (wkl_key))
    else:
        dag = ansor.workload_key_to_dag(inp.task.workload_key)
        print("Found schedule for: %s" % (wkl_key))

        s, bufs = dag.apply_steps_from_state(inp.state)
        if show_lower_result:
            print(tvm.lower(s, bufs, simple_mode=True))

        if local_measure:
            remote = None
        else:
            remote = request_remote(device_key, host, port, 1)

        cost = np.mean((measure_schedule(s, bufs, target, remote=remote, ndk_cc=ndk_cc)))
        gflops = ansor.ComputeDAG(bufs).flop_ct / cost / 1e9
        print("Best schedule: %.2f GFLOPS\tcost: %.3f ms" %
                (gflops, cost * 1e3))

    return cost, gflops


def tune_workload(wkl_key, target, target_host, policy, model_type, load_model_file,
                  load_log_file, tune_option):
    """Tune a workload"""

    if False:
        # Debug info. Print static analysis results from the access analyzer
        dag = ansor.workload_key_to_dag(wkl_key)
        print(dag.access_analyzer)
        exit()

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

    if policy == 'meta-rewrite':
        policy = ansor.MetaTileRewritePolicy(program_cost_model=model)
    elif policy == 'beam-search':
        policy = ansor.MetaTileRewritePolicy(program_cost_model=model,
                                             params={'use_beam_search': 1})
    else:
        raise ValueError("Invalid search policy: " + policy)

    s, bufs = ansor.auto_schedule(wkl_key,
                                  target=target, target_host=target_host,
                                  search_policy=policy,
                                  tune_option=tune_option)


def tune_workloads_jointly(wkl_keys, weights, task_scheduler, target, target_host,
                           search_policy, model_type, load_model_file, load_log_file,
                           tune_option):
    """Tune for multiple workloads jointly"""

    tasks = []
    for wkl_key in wkl_keys:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target, target_host))

    def objective_func(costs):
        return sum(c * w for c, w in zip(costs, weights))

    tuner = ansor.SimpleTaskScheduler(tasks, objective_func, strategy=task_scheduler,
                                      load_log_file=load_log_file, load_model_file=load_model_file)
    search_policy = "%s.%s" % (search_policy, model_type)
    tuner.tune(tune_option, search_policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Task related options
    parser.add_argument("--wkl", type=str, required=True)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)

    # Strategy related options
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--policy", type=str, choices=['meta-rewrite', 'beam-search'], default='meta-rewrite')
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='no',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')

    # File related options
    parser.add_argument("--log-file", type=str, help="Write log of measurement results to this file")
    parser.add_argument("--load-model", type=str, help="Load pre trained cost model file")
    parser.add_argument("--load-log", type=str, help="Load history log for pre-training the cost model")

    # Detailed control options
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--local-measure", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--device-key", type=str, default=None)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--n-parallel", type=int, default=1)
    parser.add_argument("--ndk-cc", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    logging.basicConfig()
    logging.getLogger('ansor').setLevel(logging.DEBUG)

    wkl_keys = get_workload_keys(args.wkl)
    target = tvm.target.create(args.target)
    log_file = args.log_file or args.wkl + ".json"

    if args.tune:
        load_log_file = args.load_log or log_file
        weights = get_workload_weights(args.wkl)

        tune_option, measure_ctx = create_tune_option(target, log_file,
                args.n_trials, args.num_measure_per_iter, args.verbose,
                args.n_parallel, args.build_timeout, args.local_measure,
                args.device_key, args.host, args.port, args.ndk_cc)

        if args.task_scheduler == 'no':
            # tune workloads one by one
            for wkl_key in wkl_keys:
                tune_workload(wkl_key, target, args.target_host, args.policy,
                              args.model_type, args.load_model, load_log_file,
                              tune_option)
        else:
            # tune workloads jointly using JointTuner
            tune_workloads_jointly(wkl_keys, weights, args.task_scheduler,
                                   target, args.target_host, args.policy,
                                   args.model_type, args.load_model, load_log_file,
                                   tune_option)
        if measure_ctx:
            del measure_ctx

    if not args.tune or len(wkl_keys) == 1:
        for wkl_key in wkl_keys:
            replay_workload(wkl_key, target, args.target_host, log_file,
                            args.local_measure, args.device_key, args.host,
                            args.port, args.ndk_cc)
