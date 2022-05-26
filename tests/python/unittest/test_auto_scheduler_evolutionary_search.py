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
import pytest
from tvm.testing.auto_scheduler import matmul_auto_scheduler_test
from tvm import auto_scheduler, te
from tvm.auto_scheduler.cost_model.cost_model import PythonBasedModel


def test_mutate_tile_size():
    """
    The test case initializes evo search with a batch of "bad" states and check whether
    the search algorithm can find "good" states by mutating the "bad" states.

    This unit test has been tested with 1,000 runs with no failures, meaning that
    the failure rate is less than 0.1%.
    """

    class MockCostModel(PythonBasedModel):
        """A mock cost model that rates 1 only for the states with tile_k=2."""

        @staticmethod
        def is_good_state(state):
            for line in str(state).split("\n"):
                if line.find("k.1") != -1 and line.find("(0,2)") != -1:
                    return True
            return False

        def predict(self, task, states):
            scores = []
            for state in states:
                scores.append(1 if self.is_good_state(state) else 0)
            return scores

    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(10, 10, 4), target=tvm.target.Target("llvm")
    )
    policy = auto_scheduler.SketchPolicy(task, program_cost_model=MockCostModel(), verbose=0)
    states = policy.sample_initial_population()[:50]

    bad_states = []
    for state in states:
        if not MockCostModel.is_good_state(state):
            bad_states.append(state)

    new_states = policy.evolutionary_search(bad_states, 50)
    found = False
    for state in new_states:
        if MockCostModel.is_good_state(state):
            found = True
            break
    assert found


@pytest.mark.skip(reason="See https://github.com/apache/tvm/issues/11440")
def test_mutate_parallel():
    """
    The test case initializes evo search with a batch of "bad" states and check whether
    the search algorithm can find "good" states by mutating the "bad" states.
    """

    class MockCostModel(PythonBasedModel):
        @staticmethod
        def is_good_state(state):
            for line in str(state).split("\n"):
                if (
                    line.find("parallel i.0@ (0") != -1
                    or line.find("parallel i.0@j.0@ (0") != -1
                    or line.find("parallel i.0@j.0@i.1@ (0") != -1
                ):
                    return True
            return False

        def predict(self, task, states):
            scores = []
            for state in states:
                scores.append(1 if self.is_good_state(state) else 0)
            return scores

    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(1024, 1024, 1024), target="llvm"
    )
    policy = auto_scheduler.SketchPolicy(task, program_cost_model=MockCostModel(), verbose=0)

    found = False
    retry_ct = 0
    while retry_ct < 10 and not found:
        states = policy.sample_initial_population()[:100]
        bad_states = []
        for state in states:
            if not MockCostModel.is_good_state(state):
                bad_states.append(state)

        new_states = policy.evolutionary_search(bad_states, 50)
        for state in new_states:
            if MockCostModel.is_good_state(state):
                found = True
                break
        retry_ct += 1

    assert found


if __name__ == "__main__":
    test_mutate_tile_size()
    test_mutate_parallel()
