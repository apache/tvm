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
import tvm.micro.testing
from tvm.testing.utils import fetch_model_from_url

TUNING_RUNS_PER_OPERATOR = 2


@pytest.mark.requires_hardware
@tvm.testing.requires_micro
def test_kws_autotune_workflow(platform, board, tmp_path):
    mod, params = fetch_model_from_url(
        url="https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite",
        model_format="tflite",
        sha256="09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83",
    )
    target = tvm.micro.testing.get_target(platform, board)

    str_io_logs = tvm.micro.testing.tune_model(
        platform, board, target, mod, params, TUNING_RUNS_PER_OPERATOR
    )
    assert isinstance(str_io_logs, StringIO)

    str_logs = str_io_logs.getvalue().rstrip().split("\n")
    logs = list(map(json.loads, str_logs))
    assert len(logs) == 2 * TUNING_RUNS_PER_OPERATOR  # Two operators

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
    with tvm.micro.testing.create_aot_session(
        platform,
        board,
        target,
        mod,
        params,
        build_dir=tmp_path,
        tune_logs=str_io_logs,
    ) as session:
        aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())

        samples = (
            np.random.randint(low=-127, high=128, size=(1, 1960), dtype=np.int8) for x in range(3)
        )

        labels = [0, 0, 0]

        # Validate perforance across random runs
        time, acc = tvm.micro.testing.evaluate_model_accuracy(
            session, aot_executor, samples, labels, runs_per_sample=20
        )
        # `time` is the average time taken to execute model inference on the
        # device, measured in seconds. It does not include the time to upload
        # the input data via RPC. On slow boards like the Arduino Due, time
        # is around 0.12 (120 ms), so this gives us plenty of buffer.
        assert time < 1


if __name__ == "__main__":
    tvm.testing.main()
