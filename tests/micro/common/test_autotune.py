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

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.testing.micro
from tvm import relay
from tvm.relay.backend import Executor, Runtime

from ..zephyr.test_utils import ZEPHYR_BOARDS
from ..arduino.test_utils import ARDUINO_BOARDS
from tvm.micro import get_microtvm_template_projects
from tvm.driver.tvmc.frontends import load_model

KWS_MODEL_LOCATION = Path(__file__).parents[1] / "testdata" / "kws" / "yes_no.tflite"



@pytest.mark.requires_hardware
@tvm.testing.requires_micro
def test_kws_autotune_workflow(platform, board):
    assert platform == "arduino"
    tvmc_model = load_model(KWS_MODEL_LOCATION, model_format="tflite")
    mod, params = tvmc_model.mod, tvmc_model.params
    target = tvm.testing.micro.get_target(platform, board)

    '''# Run autotuning and get logs
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=get_microtvm_template_projects(platform),
        project_options={
            **options,
            "project_type": "host_driven",
        },
    )
    buf_logs = tvm.testing.micro.tune_model(platform, target, mod, params, tasks, 2)
    str_logs = buf_logs.getvalue().rstrip().split("\n")
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

    # Compile the best model with AOT and connect to it
    buf_logs.seek(0)'''
    with tvm.testing.micro.create_aot_session(
            platform, board, target, mod, params, tune_logs=None,
        ) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())


        samples = (np.random.randint(
            low=-127,
            high=128,
            size=(1, 1960),
            dtype=np.int8
        ) for x in range(3))

        labels = [0, 0, 0]

        # Validate perforance across random runs
        acc, time = tvm.testing.micro.evaluate_model_accuracy(samples, labels)
        print(acc)
        print(time)

    '''# Make sure the project compiles correctly. There's no point to uploading,
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

        project.build()'''


if __name__ == "__main__":
    tvm.testing.main()
