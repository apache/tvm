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
import time
import numpy as np

import pytest
import tvm

from tvm import meta_schedule as ms
from tvm.meta_schedule import TuneConfig, tune_tir
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=no-member,invalid-name,unused-variable


@pytest.mark.skip("Integration test")
def test_tune_matmul_cpu():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul,
            target=Target("llvm --num-cores=16"),
            config=TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


@pytest.mark.skip("Integration test")
def test_tune_matmul_cuda():
    with tempfile.TemporaryDirectory() as work_dir:
        sch: Schedule = tune_tir(
            mod=matmul,
            target=Target("nvidia/geforce-rtx-3070"),
            config=TuneConfig(
                strategy="replay_trace",
                num_trials_per_iter=32,
                max_trials_per_task=32,
                max_trials_global=32,
            ),
            work_dir=work_dir,
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)


def test_tune_run_module_via_rpc():
    # pylint: disable=import-outside-toplevel

    tracker = Tracker(host="127.0.0.1", port=9000, port_end=10000, silent=True)
    device_key = "$local$device$%d" % tracker.port
    server = Server(
        port=tracker.port,
        port_end=10000,
        key=device_key,
        silent=True,
        tracker_addr=("127.0.0.1", tracker.port),
    )
    rpc_config = ms.runner.RPCConfig(
        tracker_host=tracker.host,
        tracker_port=tracker.port,
        tracker_key=device_key,
        session_timeout_sec=3600,
    )
    # Wait for the processes to start
    time.sleep(0.5)

    target = tvm.target.Target("llvm")
    rt_mod = tvm.build(matmul, target)

    # construct the input
    input_data = []
    input_shape = (128, 128)
    input_dtype = "float32"
    dev = tvm.cpu()
    a_np = np.random.uniform(size=input_shape).astype(input_dtype)
    b_np = np.random.uniform(size=input_shape).astype(input_dtype)
    c_np = np.zeros(input_shape).astype(input_dtype)
    for i in range(128):
        for j in range(128):
            for k in range(128):
                c_np[i, j] = c_np[i, j] + a_np[i, k] * b_np[j, k]
    input_data.append(tvm.nd.array(a_np, dev))
    input_data.append(tvm.nd.array(b_np, dev))
    input_data.append(tvm.nd.array(np.zeros(input_shape).astype(input_dtype), dev))

    def f_timer(rt_mod, dev, input_data):
        rt_mod(input_data[0], input_data[1], input_data[2])
        return input_data[2]

    result = run_module_via_rpc(
        rpc_config=rpc_config,
        lib=rt_mod,
        dev_type=target.kind.name,
        args=input_data,
        continuation=f_timer,
    )
    tvm.testing.assert_allclose(result.numpy(), c_np, rtol=1e-3)
    tracker.terminate()
    server.terminate()


if __name__ == """__main__""":
    test_tune_matmul_cpu()
    test_tune_matmul_cuda()
    test_tune_run_module_via_rpc()
