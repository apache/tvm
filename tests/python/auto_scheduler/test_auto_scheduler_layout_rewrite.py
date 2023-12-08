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

"""Test AutoScheduler Layout Rewrite"""
import tempfile
import numpy as np

import pytest

import tvm
import tvm.testing
from tvm import topi
from tvm import auto_scheduler, te

from tvm.testing.auto_scheduler import get_tiled_matmul, matmul_auto_scheduler_test


def test_apply_steps_with_layout_rewrite():
    dag, s = get_tiled_matmul()
    _, bufs = dag.apply_steps_from_state(s)
    assert bufs[1].shape[0] == 512
    assert bufs[1].shape[1] == 512
    _, bufs = dag.apply_steps_from_state(
        s, layout_rewrite=auto_scheduler.LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
    )
    assert bufs[1].shape[0] == 4
    assert bufs[1].shape[1] == 8
    assert bufs[1].shape[2] == 4
    assert bufs[1].shape[3] == 4
    assert bufs[1].shape[4] == 512
    _, bufs = dag.apply_steps_from_state(
        s, layout_rewrite=auto_scheduler.LayoutRewriteOption.INSERT_TRANSFORM_STAGE
    )
    assert bufs[1].shape[0] == 512
    assert bufs[1].shape[1] == 512


def test_apply_steps_with_layout_rewrite_corner_case():
    A, B, C = matmul_auto_scheduler_test(1, 1, 1)
    dag = auto_scheduler.ComputeDAG([A, B, C])

    s = dag.get_init_state()

    s.compute_root(C)
    i_j_fused = s.fuse(C, [s[C].iters[0], s[C].iters[1]])
    s.parallel(C, i_j_fused)

    _, bufs = dag.apply_steps_from_state(
        s, layout_rewrite=auto_scheduler.LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
    )


@tvm.testing.requires_llvm
def test_correctness_layout_rewrite_rewrite_for_preTransformed():
    N = 16
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(func=matmul_auto_scheduler_test, args=(N, N, N), target=target)
    dag = task.compute_dag

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        search_policy = auto_scheduler.SketchPolicy(task)

        measure_ctx = auto_scheduler.LocalRPCMeasureContext()
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=100,
            runner=measure_ctx.runner,
            verbose=2,
            early_stopping=1,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        task.tune(tuning_options, search_policy=search_policy)
        inp, _ = auto_scheduler.load_best_record(log_file, task.workload_key, target)
        s, bufs = dag.apply_steps_from_state(
            inp.state, layout_rewrite=auto_scheduler.LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
        )
        s_ref, bufs_ref = dag.apply_steps_from_state(inp.state)
        np_args = [np.random.randn(*topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]
        np_args_ref = [np.array(x) for x in np_args]

        weight = np_args_ref[1]
        # infer shape for the rewritten layout
        if len(weight.shape) >= 6:
            # For cpu tile structure SSRSRS
            base = len(weight.shape) - 6
            red_dim = weight.shape[2 + base] * weight.shape[4 + base]
            out_dim = weight.shape[3 + base] * weight.shape[5 + base]
            for i in range(base + 2):
                out_dim *= weight.shape[i]
            new_order = (
                [
                    2 + base,
                    4 + base,
                ]
                + list(range(base + 2))
                + [
                    3 + base,
                    5 + base,
                ]
            )
            np_args_ref[1] = np_args_ref[1].transpose(new_order)
            np_args_ref[1] = np_args_ref[1].reshape((red_dim, out_dim))

        func = tvm.build(s, bufs, target=target)
        func_ref = tvm.build(s_ref, bufs_ref, target=target)

        dev = tvm.device(str(target))
        dev_ref = tvm.cpu()

        args = [tvm.nd.array(x, device=dev) for x in np_args]
        args_ref = [tvm.nd.array(x, device=dev_ref) for x in np_args_ref]
        dev.sync()

        func(*args)
        func_ref(*args_ref)
        dev.sync()

        tvm.testing.assert_allclose(args[0].numpy(), args_ref[0].numpy(), atol=1e-3, rtol=1e-3)
        tvm.testing.assert_allclose(args[2].numpy(), args_ref[2].numpy(), atol=1e-3, rtol=1e-3)
        del measure_ctx


@tvm.testing.requires_llvm
def test_correctness_layout_rewrite_insert_transform_stage():
    N = 128
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(func=matmul_auto_scheduler_test, args=(N, N, N), target=target)
    dag = task.compute_dag

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        search_policy = auto_scheduler.SketchPolicy(task)

        measure_ctx = auto_scheduler.LocalRPCMeasureContext()
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=2,
            runner=measure_ctx.runner,
            verbose=1,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        task.tune(tuning_options, search_policy=search_policy)
        inp, _ = auto_scheduler.load_best_record(log_file, task.workload_key, target)
        s, bufs = dag.apply_steps_from_state(
            inp.state, layout_rewrite=auto_scheduler.LayoutRewriteOption.INSERT_TRANSFORM_STAGE
        )

        s_ref, bufs_ref = dag.apply_steps_from_state(inp.state)
        np_args = [np.random.randn(*topi.get_const_tuple(x.shape)).astype(x.dtype) for x in bufs]

        func = tvm.build(s, bufs, target=target)
        func_ref = tvm.build(s_ref, bufs_ref, target=target)

        dev = tvm.device(str(target))
        dev_ref = tvm.cpu()

        args = [tvm.nd.array(x, device=dev) for x in np_args]
        args_ref = [tvm.nd.array(x, device=dev_ref) for x in np_args]
        dev.sync()

        func(*args)
        func_ref(*args_ref)
        dev.sync()

        tvm.testing.assert_allclose(args[0].numpy(), args_ref[0].numpy(), atol=1e-3, rtol=1e-3)
        tvm.testing.assert_allclose(args[1].numpy(), args_ref[1].numpy(), atol=1e-3, rtol=1e-3)
        tvm.testing.assert_allclose(args[2].numpy(), args_ref[2].numpy(), atol=1e-3, rtol=1e-3)
        del measure_ctx


if __name__ == "__main__":
    test_apply_steps_with_layout_rewrite()
    test_apply_steps_with_layout_rewrite_corner_case()
    test_correctness_layout_rewrite_rewrite_for_preTransformed()
    test_correctness_layout_rewrite_insert_transform_stage()
