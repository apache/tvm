# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# _log2
#   http://0.apache.0 << org_log2/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-micro-keras-on-device:

Autotuning with micro TVM
=========================
**Author**: `Andrew Reusch <https://github.com/areusch>`_

This tutorial explains how to autotune a model using the C runtime.
"""

####################
# Defining the model
####################
#
# To begin with, define a model in Keras to be executed on-device. This shouldn't look any different
# from a usual Keras model definition. Let's define a relatively small model here for efficiency's
# sake.


import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(2, 3, input_shape=(16, 16, 3)))
model.build()

model.summary()


####################
# Importing into TVM
####################
# Now, use `from_keras <https://tvm.apache.org/docs/api/python/relay/frontend.html#tvm.relay.frontend.from_keras>`_ to import the Keras model into TVM.


import tvm
from tvm import relay
import numpy as np

inputs = {
    i.name.split(":", 2)[0]: [x if x is not None else 1 for x in i.shape.as_list()]
    for i in model.inputs
}
inputs = {k: [v[0], v[3], v[1], v[2]] for k, v in inputs.items()}
tvm_model, params = relay.frontend.from_keras(model, inputs, layout="NCHW")
print(tvm_model)


#######################
# Defining the target #
#######################
# Now we define the TVM target that describes the execution environment. Here, we will autotune on
# the This looks very similar
# to target definitions from other microTVM tutorials:
TARGET = tvm.target.target.micro("host")

# %%
# Autotuning on physical hardware
#  When running on physical hardware, choose a target and a board that
#  describe the hardware. The STM32F746 Nucleo target and board is chosen in
#  this commented code. Another option would be to choose the same target but
#  the STM32F746 Discovery board instead. The disco board has the same
#  microcontroller as the Nucleo board but a couple of wirings and configs
#  differ, so it's necessary to select the "stm32f746g_disco" board below.
#
#  .. code-block:: python
#
TARGET = tvm.target.target.micro("stm32f746xx")
BOARD = "nucleo_f746zg"  # or "stm32f746g_disco"


#########################
# Extracting tuning tasks
#########################
# Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
# a single implementation is defined; others don't make sense as tuning tasks. Using
# `extract_from_program`, you can produce a list of tunable tasks.
#
# Because task extraction involves running the compiler, we first configure the compiler's
# transformation passes; we'll apply the same configuration later on during autotuning.


import logging

logging.basicConfig(level=logging.INFO)
pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    # with tvm.transform.PassContext(opt_level=3):
    tasks = tvm.autotvm.task.extract_from_program(tvm_model["main"], {}, TARGET)
assert len(tasks) > 0


builder = tvm.autotvm.LocalBuilder()
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)


######################
# Configuring microTVM
######################
# Before autotuning, we need to configure a `tvm.micro.Compiler`, its associated
# `tvm.micro.Flasher`, and then use a `tvm.micro.AutoTvmAdatper` to build
# `tvm.autotvm.measure_options`. This teaches AutoTVM how to build and flash microTVM binaries onto
# the target of your choosing.
#
# In this tutorial, we'll just use the x86 host as an example runtime; however, you just need to
# replace the `Flasher` and `Compiler` instances here to run autotuning on a bare metal device.
# import os
# import tvm.micro
# workspace = tvm.micro.Workspace()
# compiler = tvm.micro.DefaultCompiler(target=TARGET)
# opts = tvm.micro.default_options(os.path.join(tvm.micro.CRT_ROOT_DIR, "host"))

# adapter = tvm.micro.AutoTvmAdapter(workspace, compiler, compiler.flasher_factory,
#                                    extra_libs=[
#                                      os.path.join(tvm.micro.build.CRT_ROOT_DIR, "memory"),
#                                    ], **opts)
# builder = tvm.autotvm.LocalBuilder(
#     build_func=adapter.StaticRuntime,
#     n_parallel=1,
#     build_kwargs={'build_option': {'tir.disable_vectorize': True}},
#     do_fork=False)  # do_fork=False needed to persist stateful builder.
# runner = tvm.autotvm.LocalRunner(
#   number=1, repeat=1, timeout=0, module_loader=adapter.CodeLoader)

# measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# %%
# Autotuning on physical hardware
#  When running on physical hardware, define a tvm.micro.Compiler that produces binaries that
#  can be loaded onto your device. Then, use AutoTvmAdapter to connect the Compiler and its
#  associated Flasher to the AutoTVM Tuner.
#
#  .. code-block:: python
#
import os
import subprocess
import tvm.micro
from tvm.micro.contrib import zephyr

workspace = tvm.micro.Workspace(debug=True)
repo_root = subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
).strip()
project_dir = f"{repo_root}/tests/micro/qemu/zephyr-runtime"
compiler = tvm.micro.CompilerFactory(
    zephyr.ZephyrCompiler,
    init_args=tuple(),
    init_kw=dict(
        project_dir=project_dir,
        board=BOARD if "stm32f746" in str(TARGET) else "qemu_x86",
        zephyr_toolchain_variant="zephyr",
    ),
)
opts = tvm.micro.default_options(os.path.join(project_dir, "crt"))

module_loader = tvm.micro.autotvm_module_loader(
    compiler,
    extra_libs=[tvm.micro.get_standalone_crt_lib("memory")],
    compiler_options=opts,
    workspace_kw={"debug": True},
)
builder = tvm.autotvm.LocalBuilder(
    #    build_func=adapter.StaticRuntime,
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=False,
)  # do_fork=False needed to persist stateful builder.
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)


################
# Run Autotuning
################
# Now we can run autotuning separately on each extracted task.
NUM_TRIALS = 10
for task in tasks:
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=NUM_TRIALS,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file("autotune.log"),
            tvm.autotvm.callback.progress_bar(NUM_TRIALS, si_prefix="M"),
        ],
        si_prefix="M",
    )


############################
# Timing the untuned program
############################
# For comparison, let's compile and run the graph without imposing any autotuning schedules. TVM
# will select a randomly-tuned implementation for each operator, which should not perform as well as
# the tuned operator.


# from tvm.contrib.debugger import debug_runtime
# with pass_context:
#   graph, lowered_mod, lowered_params = tvm.relay.build(tvm_model, target=TARGET, params=params)

# workspace = tvm.micro.Workspace(debug=True)
# micro_binary = tvm.micro.build_static_runtime(workspace, compiler, lowered_mod, extra_libs=[os.path.join(tvm.micro.build.CRT_ROOT_DIR, "memory")], **opts)
# with tvm.micro.Session(flasher=compiler.flasher_factory.instantiate(), binary=micro_binary) as sess:
#   debug_module = tvm.micro.session.create_local_debug_runtime(graph, sess._rpc.get_function('runtime.SystemLib')(), ctx=sess.context)
#   debug_module.set_input(**lowered_params)
#   debug_module.run()
#   del debug_module


##########################
# Timing the tuned program
##########################
# Once autotuning completes, you can time execution of the entire program using the Debug Runtime:


# with tvm.autotvm.apply_history_best('autotune.log'):
#   with pass_context:
#     graph, lowered_mod, lowered_params = tvm.relay.build(tvm_model, target=TARGET, params=params)

# workspace = tvm.micro.Workspace(debug=True)
# micro_binary = tvm.micro.build_static_runtime(workspace, compiler, lowered_mod, extra_libs=[os.path.join(tvm.micro.build.CRT_ROOT_DIR, "memory")], **opts)
# with tvm.micro.Session(flasher=compiler.flasher_factory.instantiate(), binary=micro_binary) as sess:
#   debug_module = tvm.micro.session.create_local_debug_runtime(graph, sess._rpc.get_function('runtime.SystemLib')(), ctx=sess.context)
#   debug_module.set_input(**lowered_params)
#   debug_module.run()
#   del debug_module
