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
                       n_parallel, build_timeout, local_measure, rpc_device_key, rpc_host,
                       rpc_port, rpc_num_threads, ndk_cc, early_stopping=-1, run_timeout=10):
    builder = runner = measure_ctx = None
    if local_measure:
        builder = ansor.LocalBuilder(timeout=build_timeout)
        if target.target_name == "cuda":
            measure_ctx = ansor.LocalRPCMeasureContext(repeat=1, min_repeat_ms=400)
            runner = measure_ctx.runner
        else:
            os.environ['TVM_AUTO_CACHE_FLUSH'] = "1"
            runner = ansor.LocalRunner(repeat=10, number=1, min_repeat_ms=0, timeout=run_timeout)
    else:
        os.environ['TVM_NDK_CC'] = ndk_cc
        builder = ansor.LocalBuilder(timeout=build_timeout, build_func='ndk')
        runner = ansor.RPCRunner(key=rpc_device_key, host=rpc_host, port=rpc_port,
                                 timeout=run_timeout, n_parallel=n_parallel,
                                 repeat=1, min_repeat_ms=200)
        remote = request_remote(rpc_device_key, rpc_host, rpc_port)
        if rpc_num_threads:
            config_threadpool = remote.get_function('runtime.config_threadpool')
            config_threadpool(0, rpc_num_threads)

    tune_option = ansor.TuneOption(n_trials=n_trials, early_stopping=early_stopping,
                                   num_measure_per_iter=num_measure_per_iter,
                                   verbose=verbose,
                                   builder=builder,
                                   runner=runner,
                                   measure_callbacks=[ansor.LogToFile(log_file)],
                                   pre_search_callbacks=[ansor.PreloadMeasuredStates(log_file)])

    return tune_option, measure_ctx


def replay_workload(wkl_key, target, target_host, log_file,
                    local_measure=True, rpc_device_key=None, rpc_host="0.0.0.0",
                    rpc_port=9190, rpc_num_threads=None, ndk_cc=None,
                    show_lower_result=True):
    cost = gflops = None

    inp, res = ansor.best_measure_pair_in_file(log_file, wkl_key, target)
    if inp is None:
        print("Cannot find log for: %s" % wkl_key)
    else:
        dag = ansor.workload_key_to_dag(inp.task.workload_key)
        print("Found schedule for: %s" % wkl_key)

        s, bufs = dag.apply_steps_from_state(inp.state)
        if show_lower_result:
            print(tvm.lower(s, bufs, simple_mode=True))

        if local_measure:
            remote = None
        else:
            remote = request_remote(rpc_device_key, rpc_host, rpc_port)
            if rpc_num_threads:
                config_threadpool = remote.get_function('runtime.config_threadpool')
                config_threadpool(0, rpc_num_threads)

        cost = np.mean((measure_schedule(s, bufs, target, target_host,
                                         remote=remote, ndk_cc=ndk_cc)))
        gflops = ansor.ComputeDAG(bufs).flop_ct / cost / 1e9
        print("Best schedule: %.2f GFLOPS\tcost: %.3f ms" % (gflops, cost * 1e3))

    return cost, gflops


def tune_workload(wkl_key, target, target_host, policy, model_type,
                  load_model_file, load_log_file, tune_option):
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

    if policy == 'sketch':
        policy = ansor.SketchSearchPolicy(program_cost_model=model)
    elif policy == 'beam-search':
        policy = ansor.SketchSearchPolicy(program_cost_model=model,
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
    """Tune for multiple workloads together with TaksScheduler"""
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
    # Search task related arguments
    parser.add_argument("--wkl", type=str, required=True)
    parser.add_argument("--target", type=str, default='llvm -mcpu=core-avx2')
    parser.add_argument("--target-host", type=str, default=None)
    parser.add_argument("--tune", type=str2bool, nargs='?', const=True, default=True)

    # Search strategy related arguments
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--policy", type=str, choices=['sketch', 'beam-search'], default='sketch')
    parser.add_argument("--model-type", type=str, choices=['xgb', 'random', 'no-share'], default='xgb')
    parser.add_argument("--task-scheduler", type=str, default='no',
                        choices=['no', 'gradient', 'round-robin'],
                        help='The strategy of task scheduler')
    parser.add_argument("--seed", type=int, default=0, help='random seed')

    # Log file related arguments
    parser.add_argument("--log-file", type=str, help="Write measurement records to this log file")
    parser.add_argument("--load-log", type=str, help="Load history log to resume the status of search")
    parser.add_argument("--load-model", type=str, help="Load pre-trained cost model from this file")

    # Measurement related and other arguments
    parser.add_argument("--num-measure-per-iter", type=int, default=48,
                        help="The number of programs to be measured at each iteration")
    parser.add_argument("--build-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=60)
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

    wkl_keys = get_workload_keys(args.wkl)
    target = tvm.target.create(args.target)
    log_file = args.log_file or args.wkl + ".json"

    # Tune workloads
    if args.tune:
        load_log_file = args.load_log or log_file
        weights = get_workload_weights(args.wkl)

        tune_option, measure_ctx = create_tune_option(target, log_file,
            args.n_trials, args.num_measure_per_iter, args.verbose,
            args.n_parallel, args.build_timeout, args.local_measure,
            args.rpc_device_key, args.rpc_host, args.rpc_port, args.rpc_num_threads,
            args.ndk_cc)

        if args.task_scheduler == 'no':
            # tune workloads one by one
            for wkl_key in wkl_keys:
                tune_workload(wkl_key, target, args.target_host, args.policy,
                              args.model_type, args.load_model, load_log_file,
                              tune_option)
        else:
            # tune workloads jointly with TaskScheduler
            tune_workloads_jointly(wkl_keys, weights, args.task_scheduler,
                                   target, args.target_host, args.policy,
                                   args.model_type, args.load_model, load_log_file,
                                   tune_option)
        if measure_ctx:
            del measure_ctx

    # Replay the best found schedule
    if len(wkl_keys) == 1 or not args.tune:
        for wkl_key in wkl_keys:
            replay_workload(wkl_key, target, args.target_host, log_file,
                            args.local_measure, args.rpc_device_key, args.rpc_host,
                            args.rpc_port, args.rpc_num_threads, args.ndk_cc)
