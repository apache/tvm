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

import contextlib
import os
import sys
import pytest
import numpy as np

import tvm
import tvm.testing
from tvm import tir, te, TVMError
from tvm.script import tir as T
from tvm import autotvm


@autotvm.template("demo_template")
def demo_template():
    M, N, K = [1024] * 3
    A = te.placeholder((M, K), dtype="float32")
    B = te.placeholder((N, K), dtype="float32")
    k = te.reduce_axis((0, 1024), name="k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[j, k], axis=[k]))

    s = te.create_schedule(C.op)
    cfg = autotvm.get_config()

    m_iter, n_iter = s[C].op.axis
    (k_iter,) = s[C].op.reduce_axis

    cfg.define_split("k_split", k_iter, num_outputs=2)
    ko, ki = cfg["k_split"].apply(s, C, k_iter)

    return s, [A, B, C]


class HexagonModuleLoader:
    def __init__(self, hexagon_session, pre_load_function=None) -> None:
        self.pre_load_function = pre_load_function
        self.hexagon_session = hexagon_session

    @contextlib.contextmanager
    def __call__(self, remote_kwargs, build_result):
        remote = self.hexagon_session._rpc
        if self.pre_load_function is not None:
            self.pre_load_function(remote, build_result)

        try:
            yield remote, self.hexagon_session.load_module(build_result)
        finally:
            pass


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=2048,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    from tvm.autotvm.tuner import XGBTuner
    from tvm.autotvm.tuner import GATuner

    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


@pytest.mark.skip(reason="AutoTVM tuning is not yet enabled on Hexagon")
@tvm.testing.requires_hexagon
def test_autotvm(hexagon_session):
    logfilename = "./hexagon.autotvm.log"

    options = {
        "log_filename": logfilename,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=15),
            runner=autotvm.RPCRunner(
                module_loader=HexagonModuleLoader(hexagon_session),
                key=hexagon_session._remote_kw["key"],
                host=hexagon_session._remote_kw["host"],
                port=hexagon_session._remote_kw["port"],
                number=3,
                timeout=15,
                min_repeat_ms=150,
                # cooldown_interval=150
            ),
        ),
    }
    target_hexagon = tvm.target.hexagon("v68")
    task = autotvm.task.create(
        "demo_template", args=[], target=target_hexagon, target_host=target_hexagon
    )
    tune_tasks([task], **options)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
