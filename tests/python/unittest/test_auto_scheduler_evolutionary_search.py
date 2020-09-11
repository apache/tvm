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
""" Test evolutionary search. """

import tvm
from test_auto_scheduler_common import matmul_auto_scheduler_test
from tvm import auto_scheduler, te
from tvm.auto_scheduler.cost_model.cost_model import PythonBasedModel


class MockCostModel(PythonBasedModel):
    """A mock cost model that rates 1 only for the states with tile_k=2."""

    def predict(self, task, states):
        scores = []
        found = False
        for state in states:
            for line in str(state).split("\n"):
                if line.find("k.1") != -1 and line.find("(0,2)") != -1:
                    found = True
                    break
            scores.append(1 if found else 0)
        return scores


def test_evo_search():
    """Test evolutionary search. Since we cannot mock random number generator,
    we mocked the cost model to manually guide the evo search. If evo search works
    as expected, it should find the target state after a sufficient number of iterations.
    This unit test has been tested with 1,000 runs with no failures, meaning that
    the failure rate is less than 0.1%.
    """
    workload_key = auto_scheduler.make_workload_key(matmul_auto_scheduler_test, (10, 10, 4))
    dag = auto_scheduler.ComputeDAG(workload_key)
    task = auto_scheduler.SearchTask(dag, workload_key, tvm.target.Target("llvm"))
    policy = auto_scheduler.SketchPolicy(task, schedule_cost_model=MockCostModel(), verbose=0)
    states = policy.sample_initial_population(50)
    pruned_states = []
    for state in states:
        found = False
        for line in str(state).split("\n"):
            # Remove all tile_k=2 states and expect evo search will fine them.
            if line.find("k.1") != -1 and line.find("(0,2)") != -1:
                found = True
                break
        if not found:
            pruned_states.append(state)

    new_states = policy.evolutionary_search(pruned_states, 50)
    found = False
    for state in new_states:
        for line in str(state).split("\n"):
            # Check if evo search found at least one state with tile_k=2.
            if line.find("k.1") != -1 and line.find("(0,2)") != -1:
                found = True
                break
        if found:
            break
    assert found


if __name__ == "__main__":
    test_evo_search()
