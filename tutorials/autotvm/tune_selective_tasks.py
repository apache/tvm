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
"""
Fast Auto-Tuning with Task Selection
====================================
**Author**: `Cody Yu <https://github.com/comaniac>`_

This is a tutorial about how to tune a model rapidly by only
tuning a few representative tasks (layers) and apply their best
schedules to other similar tasks in the model. We refer this
# feature as "selective tuning" in this tutorial.
"""
import os
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_runtime

#################################################################
# Setup Configuration and Define Network
# --------------------------------------
# Since most parts of using selective tuning is same as the normal
# AutoTVM process, please refer to :ref:`tune_relay_cuda` for details.
# Note that the selective tuning now works better on GPU instead of CPU.
# The reason is that TVM optimizes conv2d schedule on CPU with NCHWc
# layout and it has more limitations to the tile size, so the config for
# other shapes may be inapplicable. Fortunately, the tuning space on CPU
# is much smaller than GPU, so the benefit of selective tuning on CPU is
# moderated.
target = "cuda"

batch_size = 1
dtype = "float32"
log_file = "history.log"

tuning_option = {
    'log_filename': log_file,
    'n_trial': 1000,
    'tuner': 'xgb',
    'early_stopping': None,
    'measure_option':
    autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000),
    ),
}


def get_mobilenet_and_tasks(batch_size, target, dtype):
    """Get the symbol definition and random weight of a network and extract AutoTVM tasks"""
    in_shape = (batch_size, 3, 224, 224)

    mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    tasks = autotvm.task.extract_from_program(mod["main"],
                                              target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d, ))
    return mod, params, tasks, in_shape


mod, params, tasks, in_shape = get_mobilenet_and_tasks(batch_size, target, dtype)

################################################################
# Note that in the normal tuning process we replace some conv2d tasks with winograd
# implementation in the `tune_tasks` function, but we need to do so before selecting
# representative tasks, since we will need all tasks we are going to tune for analysis.

try_winograd = False
if try_winograd:
    for i in range(len(tasks)):
        try:  # try winograd template
            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                      tasks[i].target, tasks[i].target_host, 'winograd')
            input_channel = tsk.workload[1][1]
            if input_channel >= 64:
                tasks[i] = tsk
        except Exception:
            pass


#################################################################
# Mark Representative Tasks
# -------------------------
# Different from normal AutoTVM process, we need to identify
# representative tasks and spend most of our tuning time on them.
# We use a provided API to analyze and mark representative tasks.
# Initially, all tasks depend on itself, meaning that all of them
# are representative tasks.

print(all([task.depend == task for task in tasks]))

#################################################################
# .. code-block:: bash
#
#   True

#################################################################
# After we use a provided API to analyze and mark representative tasks,
# we can see that only 7 / 18 conv2d tasks are selected.
autotvm.task.mark_depend(tasks)

for idx, task in enumerate(tasks):
    if task.depend == task:
        print('Task %2d -- selected' % idx)
    else:
        print('Task %2d' % idx)

#################################################################
# .. code-block:: bash
#
#   Task  0
#   Task  1
#   Task  2 -- selected
#   Task  3 -- selected
#   Task  4
#   Task  5
#   Task  6
#   Task  7
#   Task  8
#   Task  9
#   Task 10 -- selected
#   Task 11 -- selected
#   Task 12
#   Task 13
#   Task 14
#   Task 15
#   Task 16 -- selected
#   Task 17 -- selected
#   Task 18 -- selected

#################################################################
# Two-Phase Tuning
# ----------------
# Now we start the tuning. Different from the normal AutoTVM process,
# we have to tune the selected tasks first.

def tune_kernels(tasks,
                 measure_option,
                 n_trial=1000,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    # The first phase tunes the selected tasks
    selected_tasks = [task for task in tasks if task.depend == task]
    for i, task in enumerate(selected_tasks):
        prefix = "[Selected Task %2d/%2d] " % (i + 1, len(selected_tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning as noraml
        curr_trial = min(n_trial, len(task.config_space))
        tuner_obj.tune(n_trial=curr_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(curr_trial,
                                                         prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)
                       ])

    # The second phase tunes the rest tasks
    other_tasks = [task for task in tasks if task.depend != task]
    for i, task in enumerate(other_tasks):
        prefix = "[Other Task %2d/%2d] " % (i + 1, len(other_tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning with the best schedules from selected tasks
        tuner_obj.tune(n_trial=10,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       depend_mode='top10', # We will use top 10 schedules from the depend tasks
                       callbacks=[
                           autotvm.callback.progress_bar(10, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)
                       ])

########################################################################
# Finally, we compile the network and evaluate the end-to-end performance.


def tune_and_evaluate(tasks, mod, target, params):
    print('Tuning...')
    tune_kernels(tasks, **tuning_option)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(mod,
                                                        target=target,
                                                        params=params)

    ctx = tvm.gpu(0)
    runtime = graph_runtime.create(graph, lib, ctx)
    runtime.set_input('data', tvm.nd.array(np.random.uniform(size=in_shape).astype(dtype)))
    runtime.set_input(**params)

    print("Evaluate inference time cost...")
    ftimer = runtime.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))


########################################################################
# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(tasks, mod, target, params)

######################################################################
# Sample Output on Nvidia V100
# ----------------------------
#
# .. code-block:: bash
#
#    Tuning...
#    [Selected Task  1/ 7]  Current/Best: 1458.89/2361.52 GFLOPS | Progress: (1000/1000) | 3524.46 s Done.
#    [Selected Task  2/ 7]  Current/Best:  201.83/ 394.07 GFLOPS | Progress: (1000/1000) | 2818.05 s Done.
#    [Selected Task  3/ 7]  Current/Best: 3575.41/4672.52 GFLOPS | Progress: (1000/1000) | 5499.92 s Done.
#    [Selected Task  4/ 7]  Current/Best:  881.27/1280.70 GFLOPS | Progress: (1000/1000) | 3686.66 s Done.
#    [Selected Task  5/ 7]  Current/Best: 5568.21/6276.49 GFLOPS | Progress: (1000/1000) | 6021.81 s Done.
#    [Selected Task  6/ 7]  Current/Best: 1771.98/2147.40 GFLOPS | Progress: (1000/1000) | 4524.21 s Done.
#    [Selected Task  7/ 7]  Current/Best: 2652.19/3562.88 GFLOPS | Progress: (1000/1000) | 6589.38 s Done.
#    [Other Task  1/12]  Current/Best: 2221.08/2221.59 GFLOPS | Progress: (10/10) | 33.26 s Done.
#    [Other Task  2/12]  Current/Best:  375.52/ 471.92 GFLOPS | Progress: (10/10) | 27.40 s Done.
#    [Other Task  3/12]  Current/Best: 2166.48/2599.82 GFLOPS | Progress: (10/10) | 34.59 s Done.
#    [Other Task  4/12]  Current/Best:  463.61/ 661.24 GFLOPS | Progress: (10/10) | 29.30 s Done.
#    [Other Task  5/12]  Current/Best: 2091.40/2423.46 GFLOPS | Progress: (10/10) | 32.71 s Done.
#    [Other Task  6/12]  Current/Best:  546.60/ 610.05 GFLOPS | Progress: (10/10) | 28.88 s Done.
#    [Other Task  7/12]  Current/Best: 4524.27/5232.79 GFLOPS | Progress: (10/10) | 32.72 s Done.
#    [Other Task  8/12]  Current/Best:  884.15/ 936.79 GFLOPS | Progress: (10/10) | 29.56 s Done.
#    [Other Task  9/12]  Current/Best: 6136.12/6136.12 GFLOPS | Progress: (10/10) | 34.93 s Done.
#    [Other Task 10/12]  Current/Best: 1218.13/1400.08 GFLOPS | Progress: (10/10) | 30.18 s Done.
#    [Other Task 11/12]  Current/Best: 5058.95/5063.29 GFLOPS | Progress: (10/10) | 32.35 s Done.
#    [Other Task 12/12]  Current/Best: 1823.53/1885.52 GFLOPS | Progress: (10/10) | 28.78 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 0.60 ms (0.02 ms)


######################################################################
# We can see that for rest unselected tasks we only need about 30 seconds to apply top 10 schedules
# from the task they depend on. Moreover, if you do not satisfy with the performance of certain
# tasks, other task 2 (471 GFLOPS) for example, you can simply unmark its dependent task to make it
# representative, and only re-tune it with the normal process.

retune_task = [task for task in tasks if task.depend != task][1]
retune_task.depend = retune_task

#tune_and_evaluate([retune_task], mod, target, params)

######################################################################
# Sample Output on Nvidia V100
# ----------------------------
#
# .. code-block:: bash
#
#    Tuning...
#    [Selected Task  1/ 1]  Current/Best:  178.97/ 490.45 GFLOPS | Progress: (1000/1000) | 2635.52 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 0.58 ms (0.03 ms)

######################################################################
# We then improve the throughput of the task from 471 GFLOPS to 490 GFLOPS, and also improve
# the end-to-end inference time to 0.56 ms.

