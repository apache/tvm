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
from tvm.relay.testing import dqn

def test_tune_dense_graph():
    def dense_graph(N, dtype="float32"):
        ori_data = relay.var("data", shape=(N, N), dtype=dtype)
        weight = relay.var("weight", shape=(N, N), dtype=dtype)
        data = relay.multiply(ori_data, relay.const(2, dtype=dtype))
        dense = relay.nn.dense(data, weight, out_dtype=dtype)
        dense = relay.add(dense, weight)
        dense = relay.nn.dense(dense, weight, out_dtype=dtype)
        return ori_data, weight, dense

    N = 128
    data, weight, dense = dense_graph(N)
    mod = relay.Function([data, weight], dense)
    mod = tvm.IRModule.from_expr(mod)

    ctx = tvm.context("llvm")
    target = tvm.target.create("llvm")
    d = tvm.nd.array(np.random.uniform(size=(N, N)).astype(data.type_annotation.dtype), ctx)
    w = tvm.nd.array(np.random.uniform(size=(N, N)).astype(weight.type_annotation.dtype), ctx)
    wkl_keys, wkl_weights = ansor.extract_from_program(mod, {}, target=target)

    assert len(wkl_keys) == 2
    assert len(wkl_weights) == 2

    tasks = []
    for wkl_key in wkl_keys:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target))

    tuner = ansor.SimpleTaskScheduler(tasks)
    measure_ctx = ansor.LocalRPCMeasureContext()
    with tempfile.NamedTemporaryFile() as fp:
        tuner.tune(ansor.TuneOption(n_trials=2, runner=measure_ctx.runner,
                                    measure_callbacks=[ansor.LogToFile(fp.name)]))
        with ansor.apply_history_best(fp.name):
            with tvm.transform.PassContext(opt_level=3,  disabled_pass={"AlterOpLayout"}):
                graph, lib, opt_params = relay.build_module.build(
                    mod, target=target)

                m = runtime.create(graph, lib, ctx)
                m.set_input('data', d)
                m.set_input('weight', w)
                m.run()
                res = m.get_output(0)

    del measure_ctx

    d = d.asnumpy()
    d = d * 2
    w = w.asnumpy()
    d = np.dot(d, np.transpose(w))
    d = d + w
    d = np.dot(d, np.transpose(w))

    tvm.testing.assert_allclose(res.asnumpy(), d, rtol=1e-5)


def test_tune_dqn():
    mod, params = dqn.get_workload(1, image_shape=(84, 84, 4), layout='NHWC')
    target = tvm.target.create('llvm')
    ctx = tvm.context("llvm")

    wkl_keys, wkl_weights = ansor.extract_from_program(mod, params, target)

    tasks = []
    for wkl_key in wkl_keys:
        dag = ansor.workload_key_to_dag(wkl_key)
        tasks.append(ansor.SearchTask(dag, wkl_key, target))

    assert len(tasks) == 5

    tuner = ansor.SimpleTaskScheduler(tasks)
    measure_ctx = ansor.LocalRPCMeasureContext()
    with tempfile.NamedTemporaryFile() as fp:
        tuner.tune(ansor.TuneOption(n_trials=len(tasks), runner=measure_ctx.runner,
                                    measure_callbacks=[ansor.LogToFile('tmp.json')]),
                   search_policy='meta-rewrite.random')
        with ansor.apply_history_best('tmp.json'):
            ansor.prepare_layout_rewrite(mod, params, target)
            with tvm.transform.PassContext(opt_level=3,  disabled_pass={"AlterOpLayout"}):
                graph, lib, opt_params = relay.build_module.build(mod, target=target)
            ansor.finish_layout_rewrite()

    del measure_ctx

if __name__ == "__main__":
    test_tune_dense_graph()
    test_tune_dqn()

