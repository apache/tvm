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
import json
import pickle

import tvm
from tvm import topi
from tvm import auto_scheduler, te

from tvm.testing.auto_scheduler import (
    get_tiled_matmul,
    invalid_compute_definition,
    matmul_auto_scheduler_test,
    parallel_matmul_auto_scheduler_test,
)


def test_apply_steps():
    dag, s = get_tiled_matmul()
    dag.print_python_code_from_state(s)
    sch, tensors = dag.apply_steps_from_state(s)
    tvm.lower(sch, tensors, simple_mode=True)


def test_infer_bound():
    dag, s = get_tiled_matmul()
    s = dag.infer_bound_from_state(s)


def test_estimate_flop():
    N = 512
    A, B, C = matmul_auto_scheduler_test(N, N, N)
    dag = auto_scheduler.ComputeDAG([A, B, C])
    assert abs(dag.flop_ct - 2 * N**3) < 0.5

    D = topi.nn.relu(C)
    dag = auto_scheduler.ComputeDAG([A, B, D])
    assert abs(dag.flop_ct - (2 * N**3 + N * N)) < 0.5

    # should not count the comparison operations in padding
    E = topi.nn.pad(C, [1, 1])
    dag = auto_scheduler.ComputeDAG([A, B, E])
    assert abs(dag.flop_ct - 2 * N**3) < 0.5

    F = te.compute((N, N), lambda i, j: E[i, j], name="F", attrs={"FLOP": 1234})
    dag = auto_scheduler.ComputeDAG([A, B, F])
    assert abs(dag.flop_ct - (2 * N**3 + 1234)) < 0.5

    A = te.placeholder((N, N), dtype="float32", name="A")
    F = te.compute((N, N), lambda i, j: te.if_then_else(A[i, j] > 0, A[i, j], 0))
    dag = auto_scheduler.ComputeDAG([A, F])
    assert abs(dag.flop_ct - N**2) < 0.5


def test_stage_order():
    """Test if the stage order is preserved when recovering a DAG."""
    N = 512
    A, B, C, D, E = parallel_matmul_auto_scheduler_test(N)
    sch = te.create_schedule([D.op, E.op])
    (D_local,) = sch.cache_write([D], "local")
    (E_local,) = sch.cache_write([E], "local")
    sch.cache_read(A, "shared", [D_local])
    sch.cache_read(B, "shared", [D_local])
    sch.cache_read(A, "shared", [E_local])
    sch.cache_read(C, "shared", [E_local])

    dag = auto_scheduler.ComputeDAG(sch)
    stage_ops_1 = dag.get_init_state().stage_ops

    # 3 placeholder, 4 x.shared, 2 {D,E}.local, 2 {D,E} compute
    assert len(stage_ops_1) == 11

    # Cache read stage should follow the source stage
    for idx, op in enumerate(stage_ops_1):
        if op.name == "A":
            assert (
                stage_ops_1[idx + 1].name == "A.d.shared"
                and stage_ops_1[idx + 2].name == "A.shared"
            )
        elif op.name in ["B", "C"]:
            assert stage_ops_1[idx + 1].name == "%s.shared" % op.name

    # Apply the same schedule to Ansor state and it should have the same stage order
    dag = auto_scheduler.ComputeDAG([A, B, C, D, E])
    state = dag.get_init_state()

    D_local = state.cache_write(D, "local")
    E_local = state.cache_write(E, "local")
    state.cache_read(A, "shared", [D_local])
    state.cache_read(B, "shared", [D_local])
    state.cache_read(A, "shared", [E_local])
    state.cache_read(C, "shared", [E_local])

    stage_ops_2 = state.stage_ops
    assert len(stage_ops_1) == len(stage_ops_2)

    # Cache read stage should follow the source stage
    for op1, op2 in zip(stage_ops_1, stage_ops_2):
        assert op1.name == op2.name

    # Serialize and deserialize the ComputeDAG constructed by a list of tensor ops.
    loaded_dag = pickle.loads(pickle.dumps(dag))
    assert str(loaded_dag.get_init_state()) == str(dag.get_init_state())
    assert len(loaded_dag.get_init_state().stage_ops) == len(dag.get_init_state().stage_ops)

    # Serialize and deserialize the search task. Note that we intentionally skip hardware_params
    # to test if the default one is serialized along with other attributes as well.
    task = auto_scheduler.SearchTask(
        compute_dag=dag, workload_key=json.dumps(("test-key",)), target=tvm.target.Target("llvm")
    )

    task2 = pickle.loads(pickle.dumps(task))
    assert '["test-key"]' in auto_scheduler.workload_registry.WORKLOAD_FUNC_REGISTRY
    assert str(task.compute_dag.get_init_state()) == str(task2.compute_dag.get_init_state())
    assert len(task.compute_dag.get_init_state().stage_ops) == len(
        task2.compute_dag.get_init_state().stage_ops
    )
    assert task.workload_key == task2.workload_key
    assert str(task.target) == str(task2.target)
    assert task.hardware_params.num_cores == task2.hardware_params.num_cores
    assert task.hardware_params.vector_unit_bytes == task2.hardware_params.vector_unit_bytes
    assert task.hardware_params.cache_line_bytes == task2.hardware_params.cache_line_bytes


def test_invalid_compute_dag():
    failed = False
    try:
        A, B = invalid_compute_definition()
        auto_scheduler.ComputeDAG([A, B])
    except tvm.TVMError:
        failed = True

    assert failed


if __name__ == "__main__":
    test_apply_steps()
    test_infer_bound()
    test_estimate_flop()
    test_stage_order()
    test_invalid_compute_dag()
