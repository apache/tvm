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

from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from typing import Union

import pytest

import tvm
import tvm.testing
from tvm import relay

from ..zephyr.test_utils import ZEPHYR_BOARDS
from ..arduino.test_utils import ARDUINO_BOARDS
from tvm.micro import get_microtvm_template_projects
from tvm.driver.tvmc.frontends import load_model

KWS_MODEL_LOCATION = Path(__file__).parents[1] / "testdata" / "kws" / "yes_no.tflite"


def _get_platform_model_and_options(board: str):
    if board in ZEPHYR_BOARDS.keys():
        platform = "zephyr"
        target_model = ZEPHYR_BOARDS[board]
        options = {"zephyr_board": board, "west_cmd": "west"}

    elif board in ARDUINO_BOARDS.keys():
        platform = "arduino"
        target_model = ARDUINO_BOARDS[board]
        options = {"arduino_board": board, "arduino_cli_cmd": "arduino-cli"}

    else:
        raise ValueError(f"Board {board} is not supported.")

    return platform, target_model, options


def tune_model(mod, params, target, module_loader, num_trials):
    """Tune a Relay module of a full model and return best result for each task"""

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, target)
    assert len(tasks) > 0

    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=tvm.relay.backend.Runtime("crt", {"system-lib": True}),
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)
    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

    results = StringIO()
    for task in tasks:
        tuner = tvm.autotvm.tuner.GATuner(task)

        tuner.tune(
            n_trial=num_trials,
            measure_option=measure_option,
            callbacks=[
                tvm.autotvm.callback.log_to_file(results),
                tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
            ],
            si_prefix="M",
        )
        assert tuner.best_flops > 1

    return results


@pytest.mark.requires_hardware
@tvm.testing.requires_micro
def test_kws_autotune_workflow(board):
    tvmc_model = load_model(KWS_MODEL_LOCATION, model_format="tflite")
    mod, params = tvmc_model.mod, tvmc_model.params

    platform, target_model, options = _get_platform_model_and_options(board)
    target = tvm.target.target.micro(target_model)

    # Run autotuning and get logs
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=get_microtvm_template_projects(platform),
        project_options={
            **options,
            "project_type": "host_driven",
        },
    )
    buf_logs = tune_model(mod, params, target, module_loader, 2)
    str_logs = buf_logs.getvalue().rstrip().split("\n")
    print(str_logs)
    logs = list(map(json.loads, str_logs))

    assert len(logs) == 4

    # Check we tested both operators
    op_names = list(map(lambda x: x["input"][1], logs))
    assert op_names[0] == op_names[1] == "dense_nopack.x86"
    assert op_names[2] == op_names[3] == "dense_pack.x86"

    # Make sure we tested different code. != does deep comparison in Python 3
    assert logs[0]["config"]["index"] != logs[1]["config"]["index"]
    assert logs[0]["config"]["entity"] != logs[1]["config"]["entity"]
    assert logs[2]["config"]["index"] != logs[3]["config"]["index"]
    assert logs[2]["config"]["entity"] != logs[3]["config"]["entity"]

    # Use logs to apply historical best and compile example project
    with tvm.autotvm.apply_history_best(buf_logs), tvm.transform.PassContext(
        opt_level=3, config={"tir.disable_vectorize": True}
    ):
        standalone_crt = tvm.relay.build(
            mod,
            target,
            runtime=tvm.relay.backend.Runtime("crt"),
            executor=tvm.relay.backend.Executor("aot", {"unpacked-api": True}),
            params=params,
        )

    # Make sure the project compiles correctly. There's no point to uploading,
    # as the example project won't give any output we can check.
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir) / "project"
        project = tvm.micro.generate_project(
            tvm.micro.get_microtvm_template_projects(platform),
            standalone_crt,
            work_dir,
            {
                **options,
                "project_type": "example_project",
            },
        )

        project.build()


if __name__ == "__main__":
    tvm.testing.main()
