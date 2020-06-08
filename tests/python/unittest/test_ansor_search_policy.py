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

"""Test search policy"""

import random
import os
import numpy as np
import tempfile

import tvm
from tvm import ansor
from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server

from test_ansor_common import matmul_nkkm

def search_common(target="llvm", seed=random.randint(1, 1 << 30), runner='local'):
    print("Test %s schedule search with the default search policy" % (target))

    N = 128
    A, B, C = matmul_nkkm(N, N, N)
    dag = ansor.ComputeDAG([A, B, C])
    tgt = tvm.target.create(target)
    task = ansor.SearchTask(dag, "test", tgt)

    random.seed(seed)

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        cost_model = ansor.RandomModel()
        search_policy = ansor.MetaTileRewritePolicy(cost_model, seed=seed)
        tune_option = ansor.TuneOption(n_trials=2, runner=runner,
                                       callbacks=[ansor.LogToFile(log_file)])
        state = ansor.auto_schedule(task, search_policy,
                                    tune_option=tune_option)
        sch, args = dag.apply_steps_from_state(state)

        print("==== Get State ====")
        print(state)
        print("==== Get Python Code ====")
        print(dag.print_python_code_from_state(state))

        try:
            print("==== Get Lowered Stmt ====")
            print(tvm.lower(sch, args, simple_mode=True))
            mod = tvm.build(sch, args, tgt)

            ctx = tvm.context(target, 0)
            a = tvm.nd.array(np.random.uniform(size=(N, N)).astype(A.dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=(N, N)).astype(B.dtype), ctx)
            c = tvm.nd.array(np.zeros((N, N), dtype=C.dtype), ctx)
            mod(a, b, c)
            tvm.testing.assert_allclose(c.asnumpy(), np.dot(
                a.asnumpy(), b.asnumpy()), rtol=1e-5)
            print("==== Verification passed ====")
        except Exception:
            raise Exception("Error encountered with seed: %d" % (seed))

        inp, res = ansor.best_measure_pair_in_file(log_file)
        s0 = dag.infer_bound_from_state(state)
        s1 = dag.infer_bound_from_state(inp.state)
        assert s0 == s1
    print()


def test_search_basic():
    search_common(seed=944563397)


def test_search_opencl():
    if tvm.context("opencl", 0).exist:
        host = '0.0.0.0'
        tracker = Tracker(host, port=9000, port_end=10000, silent=True)
        device_key = '$local$device$%d' % tracker.port
        server = Server(host, port=tracker.port, port_end=10000,
                        key=device_key,
                        use_popen=True, silent=True,
                        tracker_addr=(tracker.host, tracker.port))
        rpc_runner = ansor.RPCRunner(device_key, host, tracker.port)

        search_common("opencl", 380344973, rpc_runner)

        tracker.terminate()
        server.terminate()
    else:
        print("OpenCL device not found, skip this test.")


def test_search_cuda():
    ctx = tvm.context("cuda", 0)
    if ctx.exist:
        cuda_arch = "sm_" + "".join(ctx.compute_version.split('.'))
        tvm.autotvm.measure.measure_methods.set_cuda_target_arch(cuda_arch)
        host = '0.0.0.0'
        tracker = Tracker(host, port=9000, port_end=10000, silent=True)
        device_key = '$local$device$%d' % tracker.port
        server = Server(host, port=tracker.port, port_end=10000,
                        key=device_key,
                        use_popen=True, silent=True,
                        tracker_addr=(tracker.host, tracker.port))
        rpc_runner = ansor.RPCRunner(device_key, host, tracker.port)

        search_common("cuda", 903667810, rpc_runner)

        tracker.terminate()
        server.terminate()
    else:
        print("CUDA device not found, skip this test.")


if __name__ == "__main__":
    test_search_basic()
    test_search_opencl()
    test_search_cuda()
