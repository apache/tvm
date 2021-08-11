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

from os import path
import sys
import numpy as np
import pathlib
import subprocess

import tensorflow as tf
from tensorflow import keras

import pytest
import tvm
from tvm import relay

import conftest

PLATFORMS = conftest.PLATFORMS


def _get_conv2d_model():
    """Build a conv2d operator in Keras and returns an (IRModule, parameters)"""
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(2, 3, input_shape=(16, 16, 3)))
    model.build()

    inputs = {
        i.name.split(":", 2)[0]: [x if x is not None else 1 for x in i.shape.as_list()]
        for i in model.inputs
    }
    inputs = {k: [v[0], v[3], v[1], v[2]] for k, v in inputs.items()}
    mod, params = relay.frontend.from_keras(model, inputs, layout="NCHW")
    return mod, params


def test_conv2d_build(temp_dir, platform, west_cmd, skip_build, tvm_debug):
    model, zephyr_board = PLATFORMS[platform]

    tvm_model, params = _get_conv2d_model()

    target = tvm.target.target.micro(model)
    pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
    with pass_context:
        # with tvm.transform.PassContext(opt_level=3):
        tasks = tvm.autotvm.task.extract_from_program(tvm_model["main"], {}, target)
    assert len(tasks) > 0

    repo_root = pathlib.Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
    )
    module_loader = tvm.micro.autotvm_module_loader(
        template_project_dir=repo_root / "apps" / "microtvm" / "zephyr" / "template_project",
        project_options={
            "zephyr_board": zephyr_board,
            "west_cmd": west_cmd,
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

    log_path = pathlib.Path("zephyr_autotune.log")
    if log_path.exists():
        log_path.unlink()

    n_trial = 10
    for task in tasks:
        print(f"mehrdad: {task}")
        tuner = tvm.autotvm.tuner.GATuner(task)
        tuner.tune(
            n_trial=n_trial,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(str(log_path)),
                tvm.autotvm.callback.progress_bar(n_trial, si_prefix="M"),
            ],
            si_prefix="M",
        )

    with tvm.autotvm.apply_history_best(str(log_path)):
        with pass_context:
            lowered_tuned = tvm.relay.build(tvm_model, target=target, params=params)
    project = tvm.micro.generate_project(
        str(repo_root / "apps" / "microtvm" / "zephyr" / "template_project"),
        lowered_tuned,
        temp_dir / "project",
        {
            "zephyr_board": zephyr_board,
            "west_cmd": west_cmd,
            "verbose": 1,
            "project_type": "host_driven",
        },
    )
    project.build()
    project.flash()

    with tvm.micro.Session(project.transport()) as session:
        graph_mod = tvm.micro.create_local_graph_executor(
            lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
        )
        graph_mod.set_input(**lowered_tuned.get_params())
        graph_mod.run()
        output = graph_mod.get_output(0).numpy()
        del graph_mod

    assert output is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
