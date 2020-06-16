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
""" Test Relay Integration """

import tempfile
import numpy as np

import tvm
from tvm import ansor, relay
import tvm.contrib.graph_runtime as runtime

from test_ansor_common import get_tiled_matmul

def dense_graph(N, dtype="float32"):
    ori_data = relay.var("data", shape=(N, N), dtype=dtype)
    weight = relay.var("weight", shape=(N, N), dtype=dtype)
    data = relay.multiply(ori_data, relay.const(2, dtype=dtype))
    dense = relay.nn.dense(data, weight, out_dtype=dtype)
    dense = relay.add(dense, weight)
    dense = relay.nn.dense(dense, weight, out_dtype=dtype)
    return ori_data, weight, dense

def test_dense_integration():
    N = 128
    data, weight, dense = dense_graph(N)
    mod = relay.Function([data, weight], dense)
    mod = tvm.IRModule.from_expr(mod)

    ctx = tvm.context("llvm")
    target = tvm.target.create("llvm")
    d = tvm.nd.array(np.random.uniform(size=(N, N)).astype(data.type_annotation.dtype), ctx)
    w = tvm.nd.array(np.random.uniform(size=(N, N)).astype(weight.type_annotation.dtype), ctx)
    workloads, wkl_weights = ansor.extract_from_program(mod, {}, target=target)

    assert len(workloads) == 2
    assert len(wkl_weights) == 2

    tasks = []
    for wkl_key in workloads:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target))

    assert str(tasks[0].compute_dag) == "placeholder = PLACEHOLDER [128, 128]\n" + \
        "placeholder = PLACEHOLDER [128, 128]\n" + \
        "compute(z, y, x) += (placeholder[z, ((k*16) + x)]*placeholder[y, ((k*16) + x)])\n" + \
        "compute(y, x) += compute[y, x, kk]\n"

    assert str(tasks[1].compute_dag) == "placeholder = PLACEHOLDER [128, 128]\n" + \
        "placeholder = PLACEHOLDER [128, 128]\n" + \
        "compute(z, y, x) += (placeholder[z, ((k*16) + x)]*placeholder[y, ((k*16) + x)])\n" + \
        "compute(y, x) += compute[y, x, kk]\n" + \
        "T_add(ax0, ax1) = (compute[ax0, ax1] + placeholder[ax0, ax1])\n"

    tuner = ansor.SimpleTaskScheduler(tasks)
    measure_ctx = ansor.LocalRPCMeasureContext()
    with tempfile.NamedTemporaryFile() as fp:
        tuner.tune(ansor.TuneOption(n_trials=4, runner=measure_ctx.runner,
                                    measure_callbacks=[ansor.LogToFile(fp.name)]))
        with ansor.apply_history_best(fp.name):
            with relay.build_config(opt_level=3):
                graph, lib, opt_params = relay.build_module.build(
                    mod, target=target)

                m = runtime.create(graph, lib, ctx)
                m.set_input('data', d)
                m.set_input('weight', w)
                m.run()
                res = m.get_output(0)
    if measure_ctx:
        del measure_ctx

    d = d.asnumpy()
    d = d * 2
    w = w.asnumpy()
    d = np.dot(d, np.transpose(w))
    d = d + w
    d = np.dot(d, np.transpose(w))

    tvm.testing.assert_allclose(res.asnumpy(), d, rtol=1e-5)

if __name__ == "__main__":
    test_dense_integration()
