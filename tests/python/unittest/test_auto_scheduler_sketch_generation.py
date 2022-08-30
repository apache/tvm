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

""" Test sketch generation. """

import sys
import tvm
import tvm.testing

import pytest

from tvm import te, auto_scheduler
from tvm.auto_scheduler import _ffi_api
from tvm.auto_scheduler.loop_state import Stage

from tvm.testing.auto_scheduler import (
    matmul_auto_scheduler_test,
    double_matmul_auto_scheduler_test,
    conv2d_nchw_bn_relu_auto_scheduler_test,
    max_pool2d_auto_scheduler_test,
    min_nm_auto_scheduler_test,
    softmax_nm_auto_scheduler_test,
    softmax_abcd_auto_scheduler_test,
    conv2d_winograd_nhwc_auto_scheduler_test,
    zero_rank_reduce_auto_scheduler_test,
)


def generate_sketches(
    workload_func, args, target, print_for_debug=False, init_search_callbacks=None
):
    # NOTE: test_cpu_matmul_sketch and test_cpu_max_pool2d_sketch assume 4 cores to trigger all
    # possible sketch generations.
    task = auto_scheduler.SearchTask(
        func=workload_func,
        args=args,
        target=target,
        hardware_params=auto_scheduler.HardwareParams(num_cores=4, target=target),
    )
    policy = auto_scheduler.SketchPolicy(
        task, verbose=0, init_search_callbacks=init_search_callbacks
    )
    return policy.generate_sketches(print_for_debug)


def assert_compute_at_condition(stage, condition):
    assert stage.compute_at == Stage.COMPUTE_AT_TRANS_TABLE[condition]


def assert_is_tiled(stage):
    assert _ffi_api.SearchPolicyUtilsIsTiled(stage)


def assert_is_not_tiled(stage):
    assert not _ffi_api.SearchPolicyUtilsIsTiled(stage)


def assert_has_cache_write(state, stage_id):
    assert _ffi_api.SearchPolicyUtilsHasCacheWriteStage(state, stage_id)


def assert_has_cache_read(state, stage_id):
    assert _ffi_api.SearchPolicyUtilsHasCacheReadStage(state, stage_id)


def assert_has_rfactor(state, stage_id):
    assert _ffi_api.SearchPolicyUtilsHasRfactorStage(state, stage_id)


def assert_has_cross_thread_reduction(state, stage_id):
    assert _ffi_api.SearchPolicyUtilsHasCrossThreadReduction(state, stage_id)


def test_cpu_matmul_sketch():
    sketches = generate_sketches(matmul_auto_scheduler_test, (512, 512, 512), "llvm")
    """ 3 multi-level tiling sketches
        No.0 : Multi-level tiling
        No.1 : Multi-level tiling with cache write on position 0
        No.2 : Multi-level tiling with cache write on position 1
    """
    assert len(sketches) == 3
    # Sketch 0
    assert_is_tiled(sketches[0].stages[2])
    # Sketch 1
    assert_is_tiled(sketches[1].stages[2])
    assert_has_cache_write(sketches[1], 2)
    assert_compute_at_condition(sketches[1].stages[2], "iter")
    # Sketch 2
    assert_is_tiled(sketches[2].stages[2])
    assert_has_cache_write(sketches[2], 2)
    assert_compute_at_condition(sketches[2].stages[2], "iter")
    assert sketches[1] != sketches[2]

    sketches = generate_sketches(matmul_auto_scheduler_test, (8, 8, 512), "llvm")
    """ 2 rfactor sketches + 3 multi-level tiling sketches
        No.0 : Rfactor with factor position 0
        No.1 : Rfactor with factor position 1
        No.2 : Multi-level tiling
        No.3 : Multi-level tiling with cache write on position 0
        No.4 : Multi-level tiling with cache write on position 1
    """
    assert len(sketches) == 5
    # Sketch 0
    assert_has_rfactor(sketches[0], 2)
    # Sketch 1
    assert_has_rfactor(sketches[1], 2)
    assert sketches[0] != sketches[1]
    # Sketch 2
    assert_is_tiled(sketches[2].stages[2])
    # Sketch 3
    assert_is_tiled(sketches[3].stages[2])
    assert_has_cache_write(sketches[3], 2)
    assert_compute_at_condition(sketches[3].stages[2], "iter")
    # Sketch 4
    assert_is_tiled(sketches[4].stages[2])
    assert_has_cache_write(sketches[4], 2)
    assert_compute_at_condition(sketches[4].stages[2], "iter")
    assert sketches[3] != sketches[4]

    sketches = generate_sketches(double_matmul_auto_scheduler_test, (512,), "llvm")
    """ 3 multi-level tiling sketches for one matmul, so 3 * 3 = 9 sketches in total """
    assert len(sketches) == 9
    assert_is_tiled(sketches[8].stages[5])


def test_cpu_conv2d_bn_relu_sketch():
    sketches = generate_sketches(
        conv2d_nchw_bn_relu_auto_scheduler_test, (1, 56, 56, 512, 512, 3, 1, 1), "llvm"
    )
    """ 3 multi-level tiling sketches
        No.0 : Conv2d multi-level tiling with fusion on position 0
        No.1 : Conv2d multi-level tiling with fusion on position 1
        No.2 : Conv2d multi-level tiling without fusion
    """
    assert len(sketches) == 3
    # Sketch 0
    assert_is_not_tiled(sketches[0].stages[1])
    assert_is_tiled(sketches[0].stages[3])
    assert_compute_at_condition(sketches[0].stages[3], "iter")
    assert_compute_at_condition(sketches[0].stages[5], "inlined")
    assert_compute_at_condition(sketches[0].stages[7], "inlined")
    assert_compute_at_condition(sketches[0].stages[9], "inlined")
    assert_is_tiled(sketches[0].stages[10])
    # Sketch 1
    assert_is_not_tiled(sketches[1].stages[1])
    assert_is_tiled(sketches[1].stages[3])
    assert_compute_at_condition(sketches[1].stages[3], "iter")
    assert_compute_at_condition(sketches[1].stages[5], "inlined")
    assert_compute_at_condition(sketches[1].stages[7], "inlined")
    assert_compute_at_condition(sketches[1].stages[9], "inlined")
    assert_is_tiled(sketches[1].stages[10])
    # Sketch 2
    assert_is_not_tiled(sketches[2].stages[1])
    assert_is_tiled(sketches[2].stages[3])
    assert_compute_at_condition(sketches[2].stages[3], "root")
    assert_compute_at_condition(sketches[2].stages[5], "inlined")
    assert_compute_at_condition(sketches[2].stages[7], "inlined")
    assert_compute_at_condition(sketches[2].stages[9], "inlined")
    assert_is_not_tiled(sketches[2].stages[10])


def test_cpu_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_auto_scheduler_test, (1, 56, 56, 512, 1), "llvm")
    """ 1 default sketch """
    assert len(sketches) == 1
    # Sketch 0
    assert len(sketches[0].transform_steps) == 0


def test_cpu_min_sketch():
    sketches = generate_sketches(min_nm_auto_scheduler_test, (10, 1024), "llvm")
    """ 2 rfactor sketches + 1 default sketch
        No.0 : Rfactor with factor position 0
        No.1 : Rfactor with factor position 1
        No.2 : Default sketch
    """
    assert len(sketches) == 3
    # Sketch 0
    assert_has_rfactor(sketches[0], 1)
    # Sketch 1
    assert_has_rfactor(sketches[1], 1)
    assert sketches[0] != sketches[1]
    # Sketch 2
    assert len(sketches[2].transform_steps) == 0


def test_cpu_softmax_sketch():
    sketches = generate_sketches(softmax_nm_auto_scheduler_test, (1, 1024), "llvm")
    """ (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) """
    assert len(sketches) == (3 * 3)
    for i in range(0, 3):
        for j in range(0, 3):
            sketch = sketches[i * 3 + j]
            if j in [0, 1]:
                assert_has_rfactor(sketch, 1)
            if i in [0, 1]:
                assert_has_rfactor(sketch, 4 if j in [0, 1] else 3)
    assert len(sketches[8].transform_steps) == 0

    sketches = generate_sketches(softmax_abcd_auto_scheduler_test, (1, 12, 128, 128), "llvm")
    """ (2 rfactor sketches + 1 default sketch) * (2 rfactor sketches + 1 default sketch) """
    assert len(sketches) == (3 * 3)
    for i in range(0, 3):
        for j in range(0, 3):
            sketch = sketches[i * 3 + j]
            if j in [0, 1]:
                assert_has_rfactor(sketch, 1)
            if i in [0, 1]:
                assert_has_rfactor(sketch, 4 if j in [0, 1] else 3)
    assert len(sketches[8].transform_steps) == 0


def test_cpu_conv2d_winograd_sketch():
    sketches = generate_sketches(
        conv2d_winograd_nhwc_auto_scheduler_test, (1, 28, 28, 128, 128, 3, 1, 1), "llvm"
    )
    """ 3 multi-level tiling sketches
        No.0 : Bgemm multi-level tiling
        No.1 : Bgemm multi-level tiling with cache write on position 0
        No.2 : Bgemm multi-level tiling with cache write on position 1
    """
    assert len(sketches) == 3
    # Sketch 0
    assert_is_not_tiled(sketches[0].stages[1])
    assert_is_not_tiled(sketches[0].stages[2])
    assert_compute_at_condition(sketches[0].stages[3], "inlined")
    assert_is_tiled(sketches[0].stages[4])
    assert_is_tiled(sketches[0].stages[6])
    assert_compute_at_condition(sketches[0].stages[7], "inlined")
    assert_is_tiled(sketches[0].stages[8])
    assert_is_not_tiled(sketches[0].stages[9])
    # Sketch 1
    assert_is_not_tiled(sketches[1].stages[1])
    assert_is_not_tiled(sketches[1].stages[2])
    assert_compute_at_condition(sketches[1].stages[3], "inlined")
    assert_is_tiled(sketches[1].stages[4])
    assert_is_tiled(sketches[1].stages[6])
    assert_has_cache_write(sketches[1], 6)
    assert_compute_at_condition(sketches[1].stages[6], "iter")
    assert_compute_at_condition(sketches[1].stages[8], "inlined")
    assert_is_tiled(sketches[1].stages[9])
    assert_is_not_tiled(sketches[1].stages[10])
    # Sketch 2
    assert_is_not_tiled(sketches[2].stages[1])
    assert_is_not_tiled(sketches[2].stages[2])
    assert_compute_at_condition(sketches[2].stages[3], "inlined")
    assert_is_tiled(sketches[2].stages[4])
    assert_is_tiled(sketches[2].stages[6])
    assert_has_cache_write(sketches[2], 6)
    assert_compute_at_condition(sketches[2].stages[6], "iter")
    assert_compute_at_condition(sketches[2].stages[8], "inlined")
    assert_is_tiled(sketches[2].stages[9])
    assert_is_not_tiled(sketches[2].stages[10])
    assert sketches[1] != sketches[2]


def test_cpu_zero_rank_sketch():
    sketches = generate_sketches(zero_rank_reduce_auto_scheduler_test, (128,), "llvm")
    """ 2 rfactor sketches + 1 multi-level tiling sketches """
    assert len(sketches) == 3


def test_cpu_custom_sketch():
    def meet_condition_func(search_policy, state, stage_id):
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST

    def apply_func(search_policy, state, stage_id):
        ret = []
        state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
        C = state.stage_ops[2]

        ret.append([state.state_object, -1])

        s1 = state.copy()
        i, _, _ = s1[C].iters
        s1.split(C, i, [8, 2])
        ret.append([s1.state_object, -1])
        return ret

    sketches = generate_sketches(
        matmul_auto_scheduler_test,
        (512, 512, 512),
        "llvm",
        init_search_callbacks=[
            auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func)
        ],
    )
    assert len(sketches) == 2
    assert sketches[0].stages[2].iters[0].range.extent == 512
    assert sketches[0].stages[2].iters[1].range.extent == 512
    assert sketches[0].stages[2].iters[2].range.extent == 512
    assert sketches[1].stages[2].iters[0].range.extent == 32
    assert sketches[1].stages[2].iters[1].range.extent == 8
    assert sketches[1].stages[2].iters[2].range.extent == 2
    assert sketches[1].stages[2].iters[3].range.extent == 512
    assert sketches[1].stages[2].iters[4].range.extent == 512


@tvm.testing.requires_cuda
def test_cuda_matmul_sketch():
    sketches = generate_sketches(matmul_auto_scheduler_test, (512, 512, 512), "cuda")
    """ 1 multi-level tiling sketch """
    assert len(sketches) == 1
    assert_has_cache_read(sketches[0], 0)
    assert_compute_at_condition(sketches[0].stages[1], "iter")
    assert_has_cache_read(sketches[0], 2)
    assert_compute_at_condition(sketches[0].stages[3], "iter")
    assert_has_cache_write(sketches[0], 4)
    assert_is_tiled(sketches[0].stages[4])
    assert_compute_at_condition(sketches[0].stages[4], "iter")
    assert_is_tiled(sketches[0].stages[5])

    sketches = generate_sketches(matmul_auto_scheduler_test, (8, 8, 1024), "cuda")
    """ 1 cross thread reuction sketch + 1 multi-level tiling sketch """
    assert len(sketches) == 2
    # Sketch 0
    assert_has_cross_thread_reduction(sketches[0], 2)
    # Sketch 1
    assert_has_cache_read(sketches[1], 0)
    assert_compute_at_condition(sketches[1].stages[1], "iter")
    assert_has_cache_read(sketches[1], 2)
    assert_compute_at_condition(sketches[1].stages[3], "iter")
    assert_has_cache_write(sketches[1], 4)
    assert_is_tiled(sketches[1].stages[4])
    assert_compute_at_condition(sketches[1].stages[4], "iter")
    assert_is_tiled(sketches[1].stages[5])

    sketches = generate_sketches(double_matmul_auto_scheduler_test, (512,), "cuda")
    """ 1 multi-level tiling sketch for one matmul, so 1 x 1 = 1 sketch in total """
    assert len(sketches) == 1
    assert_compute_at_condition(sketches[0].stages[5], "root")
    assert_compute_at_condition(sketches[0].stages[6], "iter")


@tvm.testing.requires_cuda
def test_cuda_conv2d_bn_relu_sketch():
    sketches = generate_sketches(
        conv2d_nchw_bn_relu_auto_scheduler_test, (1, 56, 56, 512, 512, 3, 1, 1), "cuda"
    )
    """ 1 multi-level tiling sketch """
    assert len(sketches) == 1
    assert_has_cache_read(sketches[0], 1)
    assert_compute_at_condition(sketches[0].stages[1], "inlined")
    assert_compute_at_condition(sketches[0].stages[2], "iter")
    assert_has_cache_read(sketches[0], 3)
    assert_compute_at_condition(sketches[0].stages[4], "iter")
    assert_is_tiled(sketches[0].stages[5])
    assert_compute_at_condition(sketches[0].stages[5], "iter")
    assert_compute_at_condition(sketches[0].stages[7], "inlined")
    assert_compute_at_condition(sketches[0].stages[9], "inlined")
    assert_compute_at_condition(sketches[0].stages[11], "inlined")
    assert_is_tiled(sketches[0].stages[12])


@tvm.testing.requires_cuda
def test_cuda_max_pool2d_sketch():
    sketches = generate_sketches(max_pool2d_auto_scheduler_test, (1, 56, 56, 512, 0), "cuda")
    """ 1 default sketch """
    assert len(sketches) == 1
    assert len(sketches[0].transform_steps) == 0


@tvm.testing.requires_cuda
def test_cuda_min_sketch():
    sketches = generate_sketches(min_nm_auto_scheduler_test, (10, 1024), "cuda")
    """ 1 cross thread reuction sketch + 1 default sketch """
    assert len(sketches) == 2
    # Sketch 0
    assert_has_cross_thread_reduction(sketches[0], 1)
    # Sketch 1
    assert len(sketches[1].transform_steps) == 0


@tvm.testing.requires_cuda
def test_cuda_softmax_sketch():
    sketches = generate_sketches(softmax_nm_auto_scheduler_test, (2, 1024), "cuda")
    """ (1 cross thread reuction sketch + 1 default sketch) * (1 cross thread reuction sketch + 1 default sketch) """
    assert len(sketches) == (2 * 2)
    # Sketch 0
    assert_has_cross_thread_reduction(sketches[0], 1)
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    assert_has_cross_thread_reduction(sketches[0], 3)
    # Sketch 1
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    assert_has_cross_thread_reduction(sketches[1], 3)
    # Sketch 2
    assert_has_cross_thread_reduction(sketches[2], 1)
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    # Sketch 3
    assert_compute_at_condition(sketches[3].stages[2], "inlined")

    sketches = generate_sketches(softmax_abcd_auto_scheduler_test, (1, 12, 128, 128), "cuda")
    """ (1 cross thread reuction sketch + 1 default sketch) * (1 cross thread reuction sketch + 1 default sketch) """
    assert len(sketches) == (2 * 2)
    # Sketch 0
    assert_has_cross_thread_reduction(sketches[0], 1)
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    assert_has_cross_thread_reduction(sketches[0], 3)
    # Sketch 1
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    assert_has_cross_thread_reduction(sketches[1], 3)
    # Sketch 2
    assert_has_cross_thread_reduction(sketches[2], 1)
    assert_compute_at_condition(sketches[3].stages[2], "inlined")
    # Sketch 3
    assert_compute_at_condition(sketches[3].stages[2], "inlined")


@tvm.testing.requires_cuda
def test_cuda_conv2d_winograd_sketch():
    sketches = generate_sketches(
        conv2d_winograd_nhwc_auto_scheduler_test, (1, 28, 28, 128, 128, 3, 1, 1), "cuda"
    )
    """ 1 multi-level tiling sketch """
    assert len(sketches) == 1
    assert_compute_at_condition(sketches[0].stages[1], "inlined")
    assert_compute_at_condition(sketches[0].stages[2], "iter")
    assert_compute_at_condition(sketches[0].stages[3], "inlined")
    assert_is_tiled(sketches[0].stages[4])
    assert_has_cache_read(sketches[0], 4)
    assert_compute_at_condition(sketches[0].stages[5], "iter")
    assert_has_cache_read(sketches[0], 6)
    assert_compute_at_condition(sketches[0].stages[7], "iter")
    assert_is_tiled(sketches[0].stages[8])
    assert_compute_at_condition(sketches[0].stages[8], "iter")
    assert_has_cache_write(sketches[0], 8)
    assert_compute_at_condition(sketches[0].stages[9], "root")
    assert_is_tiled(sketches[0].stages[11])
    assert_is_not_tiled(sketches[0].stages[12])


@tvm.testing.requires_cuda
def test_cuda_zero_rank_sketch():
    sketches = generate_sketches(zero_rank_reduce_auto_scheduler_test, (128,), "cuda")
    """ 1 cross thread reuction sketch + 1 multi-level tiling sketch """
    assert len(sketches) == 2


if __name__ == "__main__":
    tvm.testing.main()
