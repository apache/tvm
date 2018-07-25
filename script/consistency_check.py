"""Extract tunable operators from nnvm graph and tune them"""

import argparse
import logging
import time
import json

import numpy as np

import nnvm.testing
import nnvm.compiler
import tvm
from tvm.contrib import util
from tvm import autotvm
import tvm.contrib.graph_runtime as runtime

from tune_nnvm import get_network, get_target, get_tuning_option, tune_tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default='resnet-18')
    parser.add_argument("--target", type=str, default='rpi3b-cpu')
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--n-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0x93)
    parser.add_argument("--cache-file", type=str)
    parser.add_argument("--mode", type=str, default='infer')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dtype = 'float32'

    args.cache_file = args.cache_file or args.network + "." + args.target + ".tsv"

    # device related
    device_key, target, target_host = get_target(args.target)
    tuning_option, n_times = get_tuning_option(device_key, args)

    # network
    net, params, shape, out_shape = get_network(args.network, batch_size=1)
    tasks = autotvm.task.extract_from_graph(net, shape=shape, dtype=dtype,
                                            symbols=tuning_option['tuning_symbols'],
                                            target=target, target_host=target_host)

    task = tasks[0]
    task = autotvm.task.create(task.name, task.args, task.target, task.target_host, 'vanilla')

    measure_option = autotvm.measure_option(mode='local' if tuning_option['device_key'] == 'local' else 'rpc',
                                            number=tuning_option['number'],
                                            repeat=3,
                                            rpc_device_key=tuning_option['device_key'],
                                            parallel_num=tuning_option['parallel_num'],
                                            timeout=tuning_option['timeout'],
                                            rpc_timeout=tuning_option['rpc_timeout'],
                                            use_ndk=tuning_option.get('use_ndk', False))

    if args.mode == 'tune':
        print(task.config_space)
        print(task.workload)
        tuner = autotvm.tuner.XGBTuner(task)
        tuner.tune(n_trial=1000,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file('cache.tsv')])

    dispatch_context = autotvm.apply_history_best("cache.tsv")
    config = dispatch_context.query(task.target, task.workload)

    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)

    gflops = []
    n_trial = 1000000
    tmp = []
    ct = {}
    for i in range(n_trial // tuning_option['parallel_num']):
        inputs = [autotvm.MeasureInput(task.target, task, config)] * tuning_option['parallel_num']
        results = measure_batch(inputs)

        for res in results:
            if res.error_no != 0:
                pass
            else:
                gflops.append(task.flop / np.mean(res.costs) / 1e9)
                tmp.append(gflops[-1])

#            url = res.timestamp
#            if url not in ct:
#                ct[url] = 0
#            ct[url] += res.all_cost

        if len(tmp) >= 5:
            print("[" + " ".join(["%.2f" % x for x in tmp]) + "]")
            print("var: %.4f min: %.2f max: %.2f\n" % (np.std(gflops) / np.mean(gflops), np.min(gflops), np.max(gflops)))
            tmp = []


