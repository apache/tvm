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
.. _tutorial-micro-autotune:

Autotuning with micro TVM
=========================
**Author**: `Andrew Reusch <https://github.com/areusch>`_, `Mehrdad Hessar <https://github.com/mehrdadh>`

This tutorial explains how to autotune a model using the C runtime.
"""

import argparse
from tvm.contrib import utils


PLATFORMS = {
    "host": ("host", None),
    "qemu_x86": ("host", "qemu_x86"),
    "nrf5340dk": ("nrf5340dk", "nrf5340dk_nrf5340_cpuapp"),
    "stm32f746xx_disco": ("stm32f746xx", "stm32f746g_disco"),
    "stm32f746xx_nucleo": ("stm32f746xx", "nucleo_f746zg"),
    "stm32l4r5zi_nucleo": ("stm32l4r5zi", "nucleo_l4r5zi"),
}


def main(args):
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
    # Now we define the TVM target that describes the execution environment. This looks very similar
    # to target definitions from other microTVM tutorials.
    #
    # When running on physical hardware, choose a target and a board that
    # describe the hardware. There are multiple hardware targets that could be selected from
    # PLATFORM list in this tutorial. You can chose the platform by passing --platform argument when running
    # this tutorial.
    #
    TARGET = tvm.target.target.micro(PLATFORMS[args.platform][0])
    BOARD = PLATFORMS[args.platform][1]

    #########################
    # Extracting tuning tasks
    #########################
    # Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
    # a single implementation is defined; others don't make sense as tuning tasks. Using
    # `extract_from_program`, you can produce a list of tunable tasks.
    #
    # Because task extraction involves running the compiler, we first configure the compiler's
    # transformation passes; we'll apply the same configuration later on during autotuning.

    pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
    with pass_context:
        # with tvm.transform.PassContext(opt_level=3):
        tasks = tvm.autotvm.task.extract_from_program(tvm_model["main"], {}, TARGET)
    assert len(tasks) > 0

    ######################
    # Configuring microTVM
    ######################
    # Before autotuning, we need to define a module loader and then pass that to
    # a `tvm.autotvm.LocalBuilder`. Then we create a `tvm.autotvm.LocalRunner` and use
    # both builder and runner to generates multiple measurements for auto tunner.
    #
    # In this tutorial, we have the option to use x86 host as an example or use different targets
    # from Zephyr RTOS. If you choose pass `--platform=host` to this tutorial it will uses x86. You can
    # choose other options by choosing from `PLATFORM` list.
    #

    import subprocess
    import pathlib
    import tvm.micro

    repo_root = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
    )

    if args.platform == "host":
        module_loader = tvm.micro.autotvm_module_loader(
            template_project_dir=repo_root / "src" / "runtime" / "crt" / "host",
            project_options={},
        )
        builder = tvm.autotvm.LocalBuilder(
            n_parallel=1,
            build_kwargs={"build_option": {"tir.disable_vectorize": True}},
            do_fork=False,
            build_func=tvm.micro.autotvm_build_func,
        )  # do_fork=False needed to persist stateful builder.
        runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0, module_loader=module_loader)

        measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    else:
        module_loader = tvm.micro.autotvm_module_loader(
            template_project_dir=repo_root / "apps" / "microtvm" / "zephyr" / "template_project",
            project_options={
                "zephyr_board": BOARD,
                "west_cmd": "west",
                "verbose": 1,
                "project_type": "host_driven",
            },
        )
        builder = tvm.autotvm.LocalBuilder(
            n_parallel=1,
            build_kwargs={"build_option": {"tir.disable_vectorize": True}},
            do_fork=False,
            build_func=tvm.micro.autotvm_build_func,
        )  # do_fork=False needed to persist stateful builder.
        runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0, module_loader=module_loader)

        measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    ################
    # Run Autotuning
    ################
    # Now we can run autotuning separately on each extracted task.
    NUM_TRIALS = 1
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

    with pass_context:
        lowered = tvm.relay.build(tvm_model, target=TARGET, params=params)

    temp_dir = utils.tempdir()
    if args.platform == "host":
        project = tvm.micro.generate_project(
            str(repo_root / "src" / "runtime" / "crt" / "host"), lowered, temp_dir / "project"
        )

    else:
        project = tvm.micro.generate_project(
            str(repo_root / "apps" / "microtvm" / "zephyr" / "template_project"),
            lowered,
            temp_dir / "project",
            {
                "zephyr_board": BOARD,
                "west_cmd": "west",
                "verbose": 1,
                "project_type": "host_driven",
            },
        )

    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        debug_module = tvm.micro.create_local_debug_executor(
            lowered.get_graph_json(), session.get_system_lib(), session.device
        )
        debug_module.set_input(**lowered.get_params())
        print("########## Build without Autotuning ##########")
        debug_module.run()
        del debug_module

    ##########################
    # Timing the tuned program
    ##########################
    # Once autotuning completes, you can time execution of the entire program using the Debug Runtime:

    with tvm.autotvm.apply_history_best("autotune.log"):
        with pass_context:
            lowered_tuned = tvm.relay.build(tvm_model, target=TARGET, params=params)

    temp_dir = utils.tempdir()
    if args.platform == "host":
        project = tvm.micro.generate_project(
            str(repo_root / "src" / "runtime" / "crt" / "host"), lowered_tuned, temp_dir / "project"
        )

    else:
        project = tvm.micro.generate_project(
            str(repo_root / "apps" / "microtvm" / "zephyr" / "template_project"),
            lowered_tuned,
            temp_dir / "project",
            {
                "zephyr_board": BOARD,
                "west_cmd": "west",
                "verbose": 1,
                "project_type": "host_driven",
            },
        )

    project.build()
    project.flash()
    with tvm.micro.Session(project.transport()) as session:
        debug_module = tvm.micro.create_local_debug_executor(
            lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
        )
        debug_module.set_input(**lowered_tuned.get_params())
        print("########## Build with Autotuning ##########")
        debug_module.run()
        del debug_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", required=True, choices=PLATFORMS.keys(), help="MicroTVM target plarform."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
