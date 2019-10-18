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
Auto-tuning using a 'ConfigLibrary'
===============================================
**Author**: `Matt Barrett <https://github.com/mbarrett97>`

This tutorial describes how to use a ConfigLibrary when
auto-tuning a network.
"""
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.config_library import ConfigLibrary
from tvm.autotvm.tuner.tuning_job import TuningJob
import tvm.contrib.graph_runtime as runtime

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# In this tutorial, we choose resnet-18 as simple example.


def get_network():
    """Get the symbol definition and random weight of a resnet network"""
    input_shape = (1, 3, 224, 224)
    output_shape = (1, 1000)
    mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1, dtype="float32")
    return mod, params, input_shape, output_shape


#################################################################
# We also choose some generic CPU tuning options
target = "llvm"
tuning_option = {
    'log_filename': 'tuning.log',
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}

#################################################################
# Perform the auto-tuning using jobs
# ----------------------------------
# To make use of the config library during auto-tuning, we should
# use a TuningJob. A job consists of a series of tuning tasks
# that have been tuned sequentially. We can annotate a tuning
# job with additional information such as the platform the tuning
# was performed on and when the job was run. If a config library
# is supplied to a TuningJob, the job will be saved in the
# library so that its results can be used during both compilation
# and subsequent tuning jobs.
#
# The following shows a simple tuning loop that utilises a
# TuningJob and ConfigLibrary. As the TuningJob is aware of a
# library, it can skip over tasks that have already been tuned
# during a previous job.


def tune_kernels(tasks,
                 n_trial,
                 config_library,
                 measure_option,
                 log_filename='tuning.log'):

    # Create a tuning job and point it at a config library
    job = TuningJob(
        log_filename,
        target,
        config_library=config_library,
    )
    # Use the tuning job during the tuning loop
    with job:
        for i, tsk in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

            # Convert conv2d tasks to conv2d_NCHWc tasks
            task = autotvm.task.create("topi_x86_conv2d_NCHWc", args=tsk.args,
                                       target=target, template_key='direct')
            task.workload = tsk.workload

            # Create tuner
            tuner_obj = GridSearchTuner(task)

            # Do tuning - the tuner will skip tasks which have already been tuned
            # in the config library
            tuner_obj.tune(
                n_trial=n_trial,
                early_stopping=n_trial,
                measure_option=measure_option,
                callbacks=[autotvm.callback.progress_bar(n_trial, prefix=prefix)],
            )

########################################################################
# Once we have auto-tuned a network and saved that into a config
# library, we need to use that library during compilation. This can be
# done simply by using config_library.load(target) where 'target' is the
# same as the target used during tuning.
#
# If you run this example for a second time, you should see that the
# tuning completes almost immediately as all pre-tuned tasks are
# loaded from the library.


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network()
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # Initialise a ConfigLibrary with a given index file
    config_library = ConfigLibrary("./configs")
    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, 20, config_library, **tuning_opt)

    # To use the results in the config library, use load(target)
    with config_library.load(target):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype("float32"))
        module = runtime.create(graph, lib, ctx)
        module.set_input("data", data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))


# Uncomment this line to run the tutorial
# tune_and_evaluate(tuning_option)
