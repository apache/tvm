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
# pylint: disable=missing-docstring
import logging
import tempfile

import pytest
from tvm.meta_schedule import TuneConfig, tune_te
from tvm.meta_schedule.testing import te_workload
from tvm.target.target import Target
from tvm.tir import Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


def test_tune_matmul():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_te(
            tensors=te_workload.batch_matmul_nkkm(B=1, N=128, M=128, K=128),
            target=Target("llvm --num-cores=16"),
            config=TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=1,
                max_trials_per_task=1,
                max_trials_global=1,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


if __name__ == """__main__""":
    test_tune_matmul()
