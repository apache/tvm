"""Extract tunable operators from nnvm graph and tune them"""

import argparse
import logging
import time
import os

import numpy as np

import nnvm.testing
import nnvm.compiler
import tvm
from tvm import autotvm
from tvm.contrib import util
import tvm.contrib.graph_runtime as runtime

from util import save_curve, VisLogger

def tune_tasks(tasks, tuning_option):
    for i, task in enumerate(tasks):
        print("========== Task %d/%d ==========" % (i + 1, len(tasks)))
        print("Workload: ", task.workload)
        print("GFLOP:", task.flop)
        print(task.config_space)
        with open("status", "a") as fout:
            fout.write("\t".join([tuning_option['log_filename'],
                                 str(i+1), str(len(tasks)), time.asctime()]) +'\n')

        if tuning_option['device_key'] =='local':
            mode ='local'
        else:
            mode ='rpc'

        measure_option = autotvm.measure_option(mode=mode,
                                                repeat=3,
                                                number=tuning_option['number'],
                                                rpc_device_key=tuning_option['device_key'],
                                                parallel_num=tuning_option['parallel_num'],
                                                timeout=tuning_option['timeout'],
                                                rpc_timeout=tuning_option['rpc_timeout'],
                                                use_ndk=tuning_option.get('use_ndk', False))

        monitor = autotvm.callback.Monitor()
        visloger = VisLogger(task, args.target, tuning_option['tuner'],
                            'vanilla', tuning_option['n_trial'], "vis/" + tuning_option['device_key'] + "/vis.tsv")

        # tuning
        if tuning_option['tuner'] =='xgb-rank':
            tuner = autotvm.tuner.XGBTuner(task, loss_type='rank')
        elif tuning_option['tuner'] =='ga':
            tuner = autotvm.tuner.GATuner(task, pop_size=50)
        else:
            raise RuntimeError("Invalid tuner")

        if tuning_option['transfer_learning']:
            if os.path.isfile(tuning_option['log_filename']):
                tuner.load_history(autotvm.record.load_from_file(tuning_option['log_filename']))

        tuner.tune(n_trial=min(tuning_option['n_trial'], len(task.config_space)),
                   early_stopping=tuning_option['early_stopping'],
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(tuning_option['log_filename']),
                              monitor, visloger])

        # write log
        device = tuning_option['device_key']
        backend = str(task.target).split(" ")[0]
        workload = task.workload
        tuner = tuning_option['tuner']
        template_key = task.config_space.template_key

        save_curve(args.target, backend,'op', workload, tuner, template_key,
                  {'flops': [float(x) for x in monitor.trial_scores()],
                  'timestamp': [float(x) for x in monitor.trial_timestamps()]})

def get_target(target):
    # target device
    target_table = {
       'local':      ('local','llvm -model=rpi3b -device=arm_cpu','llvm'),
       'rk3399-cpu': ('rk3399',
                      tvm.target.arm_cpu('rk3399'), None),
       'rk3399-gpu': ('rk3399',
                      'opencl -model=rk3399 -device=mali', tvm.target.arm_cpu('rk3399')),

       'rpi3b-cpu':  ('rpi3b',
                      tvm.target.arm_cpu('rasp3b'), None),
       'pynq-cpu':   ('pynq',
                      tvm.target.arm_cpu('pynq'), None),

       'hikey960-cpu':  ('hikey960',
                         'llvm -model=hikey960 -device=arm_cpu -mtriple=aarch64-linux-gnu',
                          None),
       'hikey960-gpu':  ('hikey960',
                         'opencl -model=hikey960 -device=mali',
                         'llvm -mtriple=aarch64-linux-gnu'),

       'mate10pro-cpu': ('mate10pro',
                         tvm.target.arm_cpu('mate10pro'), None),
       'mate10pro-gpu': ('mate10pro',
                         'opencl -model=mate10pro -device=mali',
                         tvm.target.arm_cpu('mate10pro')),
       'p20pro-cpu':    ('p20pro',
                         tvm.target.arm_cpu('p20pro'), None),
       'p20pro-gpu':    ('p20pro',
                         'opencl -model=p20pro -device=mali', tvm.target.arm_cpu('p20pro')),
       'pixel2-cpu':    ('pixel2',
                         tvm.target.arm_cpu('pixel2'), None),
       'pixel2-gpu':    ('pixel2',
                         'opencl -model=pixel2 -device=mali', tvm.target.arm_cpu('pixel2')),

       'mi6-cpu':       ('mi6',
                         'llvm -model=mi6 -device=arm_cpu -mtriple=arm64-linux-android', None),
       'mi6-gpu':       ('mi6',
                         'opencl -model=mi6 -device=mali', 'llvm -target=arm64-linux-android'),
    }

    device_key, target, target_host = target_table[target]
    target = tvm.target.create(target)

    return device_key, target, target_host

def get_tuning_option(device_key, args):
    # extract tasks and tuning
    tuning_option = {
       'log_filename': args.cache_file,

       'device_key': device_key,

       'tuner':'xgb-rank',
       'n_trial': args.n_trial,
       'early_stopping': 300,

       'tuning_symbols': (nnvm.sym.conv2d,),

       'transfer_learning': True,
    }

    table = {
       'local':        (2,  20,  8, 10, 10, False),
       'rk3399-cpu':   (2,  20,  8, 8, 10, False),
       'rk3399-gpu':   (2,  20,  8, 10, 50, False),
       'rpi3b-cpu':    (8,  20,  8,  4, 10, False),
       'pynq-cpu':     (2,  20,  8,  2, 10, False),
       'hikey960-cpu': (1,  20,  8, 10, 10, False),
       'hikey960-gpu': (1,  20,  8, 10, 50, False),

       'p20pro-cpu':    (2, 20,  8, 8, 10, True),
       'p20pro-gpu':    (2, 20,  8, 10, 50, True),
       'pixel2-cpu':    (2, 20,  8, 8, 10, True),
       'pixel2-gpu':    (2, 20,  8, 10, 50, True),

       'mi6-cpu':       (1, 200, 100, 6, 10, True),
       'mi6-gpu':       (1, 200, 100, 6, 10, True),
    }

    table['mate10pro-cpu'] = table['p20pro-cpu']
    table['mate10pro-gpu'] = table['p20pro-gpu']

    tuning_option['parallel_num'], tuning_option['timeout'], tuning_option['rpc_timeout'], \
            tuning_option['number'], n_times, tuning_option['use_ndk'] = table[args.target]

    return tuning_option, n_times


def get_network(name, batch_size):
    shape = {"data": (batch_size, 3, 224, 224)}
    output_shape = (batch_size, 1000)
    if name =='resnet-18':
        net, params = nnvm.testing.resnet.get_workload(num_layers=18,
                                                       batch_size=batch_size, image_shape=(3, 224, 224))
    elif name =='nature-dqn':
        shape = {"data": (batch_size, 4, 84, 84)}
        output_shape = (batch_size, 18)
        net, params = nnvm.testing.dqn.get_workload(batch_size=batch_size)
    elif name =='mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name =='squeezenet':
        net, params = nnvm.testing.squeezenet.get_workload(batch_size=batch_size,
                                                           version='1.1')
    elif name =='vgg-16':
        net, params = nnvm.testing.vgg.get_workload(batch_size=batch_size, num_layers=16)
    elif name =='test':
        from nnvm.testing import utils
        net = nnvm.sym.Variable('data')
        net = nnvm.sym.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        net = nnvm.sym.flatten(net)
        net = nnvm.sym.dense(net, units=1000)
        net, params = utils.create_workload(net, 1, (3, 224, 224))
    else:
        raise RuntimeError("Unsupported network")

    return net, params, shape, output_shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default='dqn')
    parser.add_argument("--target", type=str, default='rpi3b-cpu')
    parser.add_argument("--target-host", type=str)
    parser.add_argument("--n-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0x93)
    parser.add_argument("--cache-file", type=str)
    parser.add_argument("--mode", type=str, default='tune')
    parser.add_argument("--check", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    dtype ='float32'

    args.cache_file = args.cache_file or args.network + "." + args.target + ".log"

    # device related
    device_key, target, target_host = get_target(args.target)
    tuning_option, n_times = get_tuning_option(device_key, args)

    # network
    net, params, shape, out_shape = get_network(args.network, batch_size=1)
    if args.mode =='tune': 
        tasks = autotvm.task.extract_from_graph(net, shape=shape, dtype=dtype,
                                                symbols=tuning_option['tuning_symbols'],
                                                target=target, target_host=target_host)
        for i in range(len(tasks)):
            try: # try winograd template
                task = autotvm.task.create(tasks[i].name, tasks[i].args, tasks[i].target, tasks[i].target_host,
                                          'winograd')
                tasks.append(task)
                print("try winograd for ", i)
            except Exception as e:
                pass
        tune_tasks(tasks, tuning_option)
    elif args.mode =='infer':
        # compile kernels with history best records
        with autotvm.apply_history_best(args.cache_file):
            raw_params = params
            with nnvm.compiler.build_config(opt_level=2, add_pass=['AlterOpLayout']):
                graph, lib, params = nnvm.compiler.build(
                    net, target=target, target_host=target_host,
                    shape=shape, params=params, dtype="float32")

            tmp = util.tempdir()
            if tuning_option.get('use_ndk', False):
                from tvm.contrib import ndk
                filename = "net.so"
                path_name = tmp.relpath(filename)
                lib.export_library(path_name, ndk.create_shared)
            else:
                filename = "net.tar"
                path_name = tmp.relpath(filename)
                lib.export_library(path_name)

            if device_key =='local':
                ctx = tvm.context(str(target), 0)
                rlib = lib
            else:
                remote = autotvm.measure.request_remote(device_key, timeout=10000)
                remote.upload(path_name)
                ctx = remote.context(str(target), 0)
                rlib = remote.load_module(filename)

            rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}
            module = runtime.create(graph, rlib, ctx)
            data_tvm = tvm.nd.array((np.random.uniform(size=shape['data'])).astype(dtype))
            module.set_input('data', data_tvm)
            module.set_input(**rparams)
            module.run()
            module.run()
            output = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx, dtype=dtype)).asnumpy()

            if args.check:
                with nnvm.compiler.build_config():
                    graph, lib, params = nnvm.compiler.build(
                        net, target='llvm',
                        shape=shape, params=raw_params, dtype="float32")

                ref_ctx = tvm.cpu()
                ref_module = runtime.create(graph, lib, ref_ctx)
                ref_module.set_input('data', data_tvm)
                ref_module.set_input(**params)
                ref_module.run()
                out_reference = ref_module.get_output(0,
                        tvm.nd.empty(out_shape, ctx=ref_ctx, dtype=dtype)).asnumpy()
                np.testing.assert_allclose(out_reference, output, rtol=1e-2)

            # evaluate
            ftimer = module.module.time_evaluator("run", ctx, number=n_times, repeat=2)
            prof_res = ftimer()
            print("\n" + args.network + " " + args.target + " " + str(prof_res), "\n")
            save_curve(args.target, str(target).split()[0],'network', args.network,'tvm','vanilla',
                       {'cost': prof_res.results}, outfile='network.tsv')
    else:
        raise RuntimeError("Invalid mode: " + args.mode)

