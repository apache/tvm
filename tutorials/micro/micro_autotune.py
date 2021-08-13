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
import numpy as np
import subprocess
import pathlib

import tvm

# A mapping of a microTVM device to its target and board.
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
    # To begin with, define a model in Relay to be executed on-device. Then create an IRModule from relay model and
    # fill parameters with random numbers.
    #

    data_shape = (1, 3, 10, 10)
    weight_shape = (6, 3, 5, 5)

    data = tvm.relay.var("data", tvm.relay.TensorType(data_shape, "float32"))
    weight = tvm.relay.var("weight", tvm.relay.TensorType(weight_shape, "float32"))

    y = tvm.relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        kernel_size=(5, 5),
        kernel_layout="OIHW",
        out_dtype="float32",
    )
    f = tvm.relay.Function([data, weight], y)

    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = tvm.relay.transform.InferType()(relay_mod)

    weight_sample = np.random.rand(
        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
    ).astype("float32")
    params = {"weight": weight_sample}

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
    target = tvm.target.target.micro(PLATFORMS[args.platform][0])
    board = PLATFORMS[args.platform][1]

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
        tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, target)
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

    repo_root = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
    )

    if args.platform == "host":
        module_loader = tvm.micro.AutoTvmModuleLoader(
            template_project_dir=repo_root / "src" / "runtime" / "crt" / "host",
            project_options={},
        )
        builder = tvm.autotvm.LocalBuilder(
            n_parallel=1,
            build_kwargs={"build_option": {"tir.disable_vectorize": True}},
            do_fork=True,
            build_func=tvm.micro.autotvm_build_func,
        )
        runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0, module_loader=module_loader)

        measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    else:
        module_loader = tvm.micro.AutoTvmModuleLoader(
            template_project_dir=repo_root / "apps" / "microtvm" / "zephyr" / "template_project",
            project_options={
                "zephyr_board": board,
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
        )
        runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=0, module_loader=module_loader)

        measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    ################
    # Run Autotuning
    ################
    # Now we can run autotuning separately on each extracted task.

    num_trials = 10
    for task in tasks:
        tuner = tvm.autotvm.tuner.GATuner(task)
        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file("microtvm_autotune.log"),
                tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
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
        lowered = tvm.relay.build(relay_mod, target=target, params=params)

    temp_dir = tvm.contrib.utils.tempdir()
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
                "zephyr_board": board,
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

    with tvm.autotvm.apply_history_best("microtvm_autotune.log"):
        with pass_context:
            lowered_tuned = tvm.relay.build(relay_mod, target=target, params=params)

    temp_dir = tvm.contrib.utils.tempdir()
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
                "zephyr_board": board,
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
    parser = argparse.ArgumentParser(
        description="A tutorial explains how to autotune a model for microTVM targets."
    )
    parser.add_argument(
        "--platform", required=True, choices=PLATFORMS.keys(), help="MicroTVM target plarform."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
