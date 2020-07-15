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
from tvm import auto_scheduler

from test_auto_scheduler_common import matmul_auto_scheduler_test, PropagatingThread

def search_common(workload=matmul_auto_scheduler_test, target="llvm",
                  search_policy=auto_scheduler.EmptyPolicy(),
                  seed=random.randint(1, 1 << 30), runner='local', cost_model=None,
                  num_measure_trials=2, params=None, pre_search_callbacks=None):
    print("Test %s schedule search with the default search policy" % (target))

    random.seed(seed)
    N = 128
    workload_key = auto_scheduler.make_workload_key(workload, (N, N, N))
    dag = auto_scheduler.ComputeDAG(workload_key)
    target = tvm.target.create(target)
    task = auto_scheduler.SearchTask(dag, workload_key, target)

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        tuning_options = auto_scheduler.TuningOptions(num_measure_trials=num_measure_trials, runner=runner,
                                             verbose=0,
                                             measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                                             pre_search_callbacks=pre_search_callbacks)
        sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options)
        inp, res = auto_scheduler.load_best(log_file, workload_key, target)

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


def test_workload_registry_search_basic():
    if not tvm.runtime.enabled("llvm"):
        return
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(target=search_common, kwargs={'seed': 944563397})
    t.start()
    t.join()
    t = PropagatingThread(target=search_common,
                          kwargs={'seed': 944563397, 'workload': "matmul_auto_scheduler_test"})
    t.start()
    t.join()
    t = PropagatingThread(target=search_common,
                          kwargs={'seed': 944563397, 'workload': "matmul_auto_scheduler_test_rename_1"})
    t.start()
    t.join()

if __name__ == "__main__":
    test_workload_registry_search_basic()
