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

"""Test ComputeDAG (replay, infer bound)"""

import tvm
from tvm import ansor, te

from test_ansor_common import get_tiled_matmul


def test_apply_steps():
    dag, s = get_tiled_matmul()
    dag.print_python_code_from_state(s)
    sch, tensors = dag.apply_steps_from_state(s)
    stmt = tvm.lower(sch, tensors, simple_mode=True)


def test_infer_bound():
    dag, s = get_tiled_matmul()
    s = dag.infer_bound_from_state(s)

    A_global, B_global, C_global = 1, 3, 4
    assert s.stages[B_global].iters[0].range.extent == 512
    assert s.stages[B_global].iters[1].range.extent == 16
    assert s.stages[A_global].iters[0].range.extent == 1
    assert s.stages[A_global].iters[1].range.extent == 16
    assert s.stages[C_global].iters[0].range.extent == 64


def test_estimate_flop():
    dag, s = get_tiled_matmul()

    assert abs(dag.flop_ct - 2 * 512 ** 3) < 0.5


def test_lower_legalize_invalid_attach():
    N, M = 10, 10

    A = te.compute((N, M), lambda i, j: 1.0, name='A')
    B = te.compute((N, M), lambda i, j: A[i][j], name='B')

    dag = ansor.ComputeDAG([A, B])
    s = dag.get_init_state()

    A, B = 0, 1
    s.compute_at(A, B, s.stages[B].iters[1])
    s.split(B, s.stages[B].iters[1], [2])

    sch, tensors = dag.apply_steps_from_state(s)
    stmt = tvm.lower(sch, tensors, simple_mode=True)


if __name__ == "__main__":
    test_apply_steps()
    test_infer_bound()
    test_estimate_flop()
    test_lower_legalize_invalid_attach()

