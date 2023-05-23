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

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.micro.testing
from tvm.testing.utils import fetch_model_from_url

TUNING_RUNS_PER_OPERATOR = 2


@pytest.mark.requires_hardware
@tvm.testing.requires_micro
@pytest.mark.skip_boards(
    ["nucleo_l4r5zi", "", "nucleo_f746zg", "stm32f746g_disco", "nrf5340dk_nrf5340_cpuapp"]
)
def test_kws_autotune_workflow(platform, board, tmp_path):
    mod, params = fetch_model_from_url(
        url="https://github.com/tensorflow/tflite-micro/raw/a56087ffa2703b4d5632f024a8a4c899815c31bb/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite",
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

    # Some tuning tasks don't have any config space, and will only be run once
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        tasks = tvm.autotvm.task.extract_from_program(mod["main"], {}, target)
    assert len(tasks) <= len(logs) <= len(tasks) * TUNING_RUNS_PER_OPERATOR

    # Check we tested both operators
    op_names = list(map(lambda x: x["input"][1], logs))
    assert op_names[0] == op_names[1] == "conv2d_nhwc_spatial_pack.arm_cpu"

    # Make sure we tested different code. != does deep comparison in Python 3
    assert logs[0]["config"]["index"] != logs[1]["config"]["index"]
    assert logs[0]["config"]["entity"] != logs[1]["config"]["entity"]

    # Compile the best model with AOT and connect to it
    str_io_logs.seek(0)
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

        # Validate perforance across random runs
        runtimes = [
            runtime
            for _, runtime in tvm.micro.testing.predict_labels_aot(
                session, aot_executor, samples, runs_per_sample=20
            )
        ]
        # `time` is the average time taken to execute model inference on the
        # device, measured in seconds. It does not include the time to upload
        # the input data via RPC. On slow boards like the Arduino Due, time
        # is around 0.12 (120 ms), so this gives us plenty of buffer.
        assert np.median(runtimes) < 1


if __name__ == "__main__":
    tvm.testing.main()
