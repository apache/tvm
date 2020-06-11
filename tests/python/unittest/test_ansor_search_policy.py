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
import numpy as np
import tempfile

import tvm
from tvm import ansor

from test_ansor_common import matmul_ansor_test

def search_common(target="llvm", seed=random.randint(1, 1 << 30), runner='local',
                  cost_model=ansor.RandomModel(), n_trials=2, params=None,
                  pre_search_callbacks=None):
    print("Test %s schedule search with the default search policy" % (target))

    random.seed(seed)
    N = 128
    workload_key = ansor.make_workload_key_func(matmul_ansor_test, (N, N, N))
    dag = ansor.workload_key_to_dag(workload_key)
    target = tvm.target.create(target)
    task = ansor.SearchTask(dag, workload_key, target)

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        search_policy = ansor.MetaTileRewritePolicy(cost_model, params=params,
                                                    seed=seed)
        tune_option = ansor.TuneOption(n_trials=n_trials, runner=runner,
                                       measure_callbacks=[ansor.LogToFile(log_file)],
                                       pre_search_callbacks=pre_search_callbacks)
        sch, args = ansor.auto_schedule(task, search_policy=search_policy,
                                        tune_option=tune_option)
        inp, res = ansor.best_measure_pair_in_file(log_file, workload_key, target)

        print("==== Python Code ====")
        print(dag.print_python_code_from_state(inp.state))

        try:
            print("==== Lowered Stmt ====")
            print(tvm.lower(sch, args, simple_mode=True))
            mod = tvm.build(sch, args, target)

            ctx = tvm.context(str(target), 0)
            dtype = dag.tensors[0].dtype
            a = tvm.nd.array(np.random.uniform(size=(N, N)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=(N, N)).astype(dtype), ctx)
            c = tvm.nd.array(np.zeros((N, N), dtype=dtype), ctx)
            mod(a, b, c)
            tvm.testing.assert_allclose(c.asnumpy(), np.dot(
                a.asnumpy(), b.asnumpy()), rtol=1e-5)
            print("==== Verification passed ====")
        except Exception:
            raise Exception("Error encountered with seed: %d" % (seed))
    print()


def test_search_basic():
    search_common(seed=944563397)


def test_search_xgb_model_rpc_runner():
    measure_ctx = ansor.LocalRPCMeasureContext()
    search_common(seed=456787236, cost_model=ansor.XGBModel(),
                  runner=measure_ctx.runner)


def test_search_opencl():
    if tvm.context("opencl", 0).exist:
        measure_ctx = ansor.LocalRPCMeasureContext()
        search_common("opencl", 380344973, measure_ctx.runner)
    else:
        print("OpenCL device not found, skip this test.")


def test_search_cuda():
    if tvm.context("cuda", 0).exist:
        measure_ctx = ansor.LocalRPCMeasureContext()
        search_common("cuda", 903667810, measure_ctx.runner)
    else:
        print("CUDA device not found, skip this test.")


def test_search_custom_sketch_rule():
    def meet_condition_func(meta_policy, state, stage_id):
        # Apply and Skip the Rest if this function does not return
        pass

    # Expecting:
    # i.0
    #   i.1
    #     i.2
    #       j.0
    #         j.1
    #           ax0
    #             ax1
    #               B.global
    #           j.2
    #             k
    #               C
    def apply_func1(meta_policy, state, stage_id):
        # Stage by stage way
        ret = []
        if stage_id == 2:
            state = ansor.loop_state.State(state)
            state.split(2, state.stages[2].iters[0], [4, 4])
            state.split(2, state.stages[2].iters[3], [4, 4])
            ret.append([state.state_object, stage_id - 1])
        elif stage_id == 1:
            state = ansor.loop_state.State(state)
            state.cache_read(1, "global", [2], meta_policy.cur_task.compute_dag)
            state.compute_at(2, 3, state.stages[3].iters[4])
            ret.append([state.state_object, stage_id - 1])
        else:
            ret.append([state, stage_id - 1])
        return ret

    def apply_func2(meta_policy, state, stage_id):
        # More template like way
        ret = []
        state = ansor.loop_state.State(state)

        state.split(2, state.stages[2].iters[0], [4, 4])
        state.split(2, state.stages[2].iters[3], [4, 4])
        state.cache_read(1, "global", [2], meta_policy.cur_task.compute_dag)
        state.compute_at(2, 3, state.stages[3].iters[4])

        ret.append([state.state_object, -1])
        return ret

    measure_ctx = ansor.LocalRPCMeasureContext()
    search_common(seed=887823438, runner=measure_ctx.runner,
                  pre_search_callbacks=[ansor.PreAddCustomRule(meet_condition_func,
                                                               apply_func1)],
                  params={'disable_change_compute_location': 1})
    search_common(seed=887823438, runner=measure_ctx.runner,
                  pre_search_callbacks=[ansor.PreAddCustomRule(meet_condition_func,
                                                               apply_func2)],
                  params={'disable_change_compute_location': 1})


if __name__ == "__main__":
    test_search_basic()
    test_search_xgb_model_rpc_runner()
    test_search_opencl()
    test_search_cuda()
    test_search_custom_sketch_rule()
