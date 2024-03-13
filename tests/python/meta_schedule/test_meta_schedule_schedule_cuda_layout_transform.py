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

import itertools
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule, relay
from tvm.meta_schedule.schedule.cuda.layout_transform import (
    cuda_layout_transform_schedule_rule,
)
from tvm.relay.op import OpPattern
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.schedule import BlockRV

# fmt: off
# Small gpu parameters which should work for nearly every (modern-ish) gpu.
TARGET = tvm.target.Target(
    "cuda -max_threads_per_block=32 -max_num_threads=128 -thread_warp_size=32 -max_shared_memory_per_block=8192 -registers_per_block=1024"
)


class PatchCustomLayoutTransformScheduleRule:
    """Patch the custom layout transform schedule to test only specific tile sizes.

    If tile_sizes = [], then returns the default (non-tiled) schedule, otherwise
    returns only the schedule with the given tiles.
    """

    FUNC_NAME = "meta_schedule.cuda.layout_transform"

    def __init__(self, tile_sizes: List[int]) -> None:
        self.tile_sizes = tile_sizes
        self.old_func = None

    def __enter__(self, *args, **kwargs) -> None:
        self.old_func = tvm.get_global_func(self.FUNC_NAME)

        def new_layout_rule(
            sch: tvm.tir.Schedule,
            block: BlockRV,
            tile_sizes: Optional[List[int]] = self.tile_sizes,
        ) -> List[tvm.tir.Schedule]:
            return cuda_layout_transform_schedule_rule(sch, block, tile_sizes)

        tvm.register_func(self.FUNC_NAME, new_layout_rule, override=True)

    def __exit__(self, *args, **kwargs) -> None:
        tvm.register_func(self.FUNC_NAME, self.old_func, override=True)


# Create unary functions which apply ops with compatible fusion levels to layout transform
def get_random_axis(data: relay.Expr):
    rank = len(relay.transform.InferTypeLocal(data).shape)
    return random.randint(0, rank - 1)


def apply_elemwise_clip(data: relay.Expr, min=0, max=10):
    assert relay.op.get("clip").get_attr("TOpPattern") == OpPattern.ELEMWISE
    return relay.clip(data, min, max)


def apply_broadcast_add(data: relay.Expr, val_to_add=5):
    assert relay.op.get("add").get_attr("TOpPattern") == OpPattern.BROADCAST
    type_info = relay.transform.InferTypeLocal(data)
    return relay.add(data, relay.const(val_to_add, dtype=type_info.dtype))


def apply_injective_concatenate(data: relay.Expr, axis=None):
    if axis is None:
        axis = get_random_axis(data)
    assert relay.op.get("concatenate").get_attr("TOpPattern") == OpPattern.INJECTIVE
    return relay.concatenate([data, data], axis)


def apply_comm_reduce_max(data: relay.Expr, axis=None):
    if axis is None:
        axis = get_random_axis(data)
    assert relay.op.get("max").get_attr("TOpPattern") == OpPattern.COMM_REDUCE

    # Do this to maintain dimensions
    return relay.add(data, relay.max(data, axis, keepdims=True))


pattern_level_to_op = {
    OpPattern.ELEMWISE: apply_elemwise_clip,
    OpPattern.BROADCAST: apply_broadcast_add,
    OpPattern.INJECTIVE: apply_injective_concatenate,
    OpPattern.COMM_REDUCE: apply_comm_reduce_max,
}


def apply_layout_transform(data: relay.Expr, src_layout: str, dst_layout: str):
    assert relay.op.get("layout_transform").get_attr("TOpPattern") == OpPattern.INJECTIVE
    return relay.layout_transform(data, src_layout, dst_layout)


def create_relay_module(
    input_shape: List[int], dtype: str, ops: List[Union[OpPattern, Tuple[str, str]]]
) -> tvm.IRModule:
    """Create a relay module with the given string of ops.

    ops:
        Applies the associated operators in order. If an integer, refers to applying
        the unary operator from `extra_pattern_level_to_op` map. If a tuple, applies
        a layout transform with the given (src_layout, dst_layout)
    """
    input_data = relay.var("input", shape=input_shape, dtype=dtype)

    cur_data = input_data
    for op_info in ops:
        # Progressively build type info
        relay.transform.InferTypeLocal(cur_data)
        if isinstance(op_info, tuple):
            # layout transform case
            src_layout, dst_layout = op_info
            cur_data = apply_layout_transform(cur_data, src_layout, dst_layout)
        else:
            cur_data = pattern_level_to_op[op_info](cur_data)

    relay.transform.InferTypeLocal(cur_data)
    return tvm.IRModule.from_expr(cur_data)


def extract_layout_transform_task(
    mod: tvm.IRModule, target: tvm.target.Target
) -> meta_schedule.ExtractedTask:
    """Given a relay IRModule, return the PrimFunc IRModule with fused layout transform task."""
    extracted_tasks = meta_schedule.relay_integration.extract_tasks(
        mod,
        target,
        {},
        pass_config={"relay.backend.use_meta_schedule": True},
    )
    task_of_interest = None
    for task in extracted_tasks:
        if "layout_transform" in task.task_name:
            task_of_interest = task
            break
    assert task_of_interest is not None
    return task_of_interest


def run_primfunc(
    primfunc_mod: tvm.IRModule, target: tvm.target.Target, input_tensors: List[tvm.nd.NDArray]
):
    """Compile and run the primfunc with the given input tensors."""
    with tvm.transform.PassContext(
        config={"relay.backend.use_meta_schedule": True},
        opt_level=3,
    ):
        lib = tvm.build(primfunc_mod, target=target)
    lib(*input_tensors)


@pytest.mark.skip("Integration test")
class TestRandomRelayE2ECorrectness:
    """Tests E2E correctness of layout transform schedule.

    Randomly generates relay mod with layout transform and fusable ops. Checks the
    layout transform task for correctness by comparing against its unscheduled result.
    """

    @staticmethod
    def generate_test_case(
        input_shape: List[int],
        implicit_reshape_info: Optional[Tuple[int, int]],
        dtype: str,
        num_additional_ops: int,
    ) -> tvm.IRModule:
        """Creates a random layout transform module with up to num_additional_ops fused."""
        # Create layout transforms
        rank = len(input_shape)

        # src_layout is a string like ABCDEFG... with length as rank
        src_layout = "".join([chr(i + ord("A")) for i in range(rank)])

        # dst_layout is randomly shuffled src_layout, potentially after adding split axis
        dst_layout = list(src_layout)
        if implicit_reshape_info:
            axis_to_reshape, size_new_dim = implicit_reshape_info
            cur_dim = dst_layout[axis_to_reshape]
            dst_layout[axis_to_reshape] = f"{cur_dim}"
            dst_layout.append(f"{size_new_dim}{cur_dim.lower()}")

        random.shuffle(dst_layout)
        while "".join(dst_layout) == src_layout:
            random.shuffle(dst_layout)
        dst_layout = "".join(dst_layout)

        # Randomly sample a list of potentially fusable ops to layout transform
        op_order = random.choices(
            list(pattern_level_to_op.keys()),
            k=num_additional_ops,
        )

        # Append tuple, representing layout transfomr from src --> dst layout
        op_order.append((src_layout, dst_layout))

        random.shuffle(op_order)
        return create_relay_module(input_shape, dtype, op_order)

    @staticmethod
    def get_primfunc(extracted_task: meta_schedule.ExtractedTask, tile_size: Optional[int]):
        with PatchCustomLayoutTransformScheduleRule(
            tile_sizes=[] if tile_size is None else [tile_size]
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                (
                    tune_contexts,
                    _,
                ) = meta_schedule.relay_integration.extracted_tasks_to_tune_contexts(
                    [extracted_task],
                    tmpdir,
                )
                tune_contexts[0].pre_tuning(1)
                candidates = tune_contexts[0].generate_measure_candidates()
                primfunc = candidates[0].sch.mod["main"]
                return primfunc

    @staticmethod
    def verify_layout_transform_task(
        extracted_task: meta_schedule.ExtractedTask,
        target: tvm.target.Target,
        tile_sizes: List[int],
    ):
        """Given a layout transform task, tests the given tile_sizes and verifies output matches."""
        device = tvm.cuda(0)
        relay_mod = extracted_task.mod

        # Create and cache inputs
        func_type = relay.transform.InferTypeLocal(relay_mod[relay_mod.get_global_vars()[0]])
        input_tensors = []
        for input_type in func_type.arg_types:
            orig_input_np = np.random.uniform(0, 10, size=list(map(int, input_type.shape))).astype(
                input_type.dtype
            )
            orig_input_np = np.arange(0, orig_input_np.size, dtype=input_type.dtype).reshape(
                orig_input_np.shape
            )
            input_tensors.append(tvm.nd.array(orig_input_np, device))
        ret_type = func_type.ret_type

        def get_output_tensor() -> Tuple[tvm.nd.NDArray, tvm.nd.NDArray]:
            numpy_init = np.random.uniform(0, 1000, size=list(map(int, ret_type.shape))).astype(
                ret_type.dtype
            )
            return tvm.nd.array(numpy_init, device)

        def run_and_get_output(tile_size: Optional[int]) -> np.ndarray:
            returned_primfunc = TestRandomRelayE2ECorrectness.get_primfunc(
                extracted_task, tile_size
            )
            output_tensor = get_output_tensor()
            run_primfunc(returned_primfunc, target, [*input_tensors, output_tensor])
            return output_tensor.numpy()

        # Passing None, we basically do not apply the custom rule we have created
        # and instead use the old default schedule which is the ground truth.
        ground_truth_np = run_and_get_output(None)

        for tile_size in tile_sizes:
            experimental_np = run_and_get_output(tile_size)
            np.testing.assert_allclose(ground_truth_np, experimental_np)

    (
        input_shape,
        implicit_reshape_info,
        dtype,
        tile_sizes,
        num_additional_ops,
    ) = tvm.testing.parameters(
        *itertools.product(
            # input_shape: Each has ~10k elements, should take single microseconds on modern gpu
            [
                [12, 48, 18],
                [890, 14],
                [10, 12, 2, 5, 3, 3],
            ],
            # implicit_reshape_info: Implicit reshape conditions.
            # None is do no implicit reshape, (0, 2) means divide axis 0 in half, e.g. AB --> A2aB
            [None, (0, 2), (1, 2)],
            # dtype: dtypes to test, should not matter that much
            ["float16"],
            # tile_sizes: Tile sizes to try
            [[8, 7]],
            # num_additional_ops: number of non-layout transform ops to include and may be fused
            [5],
        )
    )

    @tvm.testing.requires_gpu
    def test_all_test_case(
        self,
        input_shape,
        implicit_reshape_info,
        dtype,
        tile_sizes,
        num_additional_ops,
    ):
        """Tests the product of all conditions `repeat_per_condition` times."""
        # Generate random module of fusable ops + layout transform and extract fused layout transform task
        full_mod = self.generate_test_case(
            input_shape, implicit_reshape_info, dtype, num_additional_ops
        )

        # Fused layout transform task
        extracted_task = extract_layout_transform_task(full_mod, TARGET)
        self.verify_layout_transform_task(extracted_task, TARGET, tile_sizes)


@tvm.testing.requires_gpu
class TestManualCases:
    def assert_extracted_equals_expected(
        self, relay_mod: tvm.IRModule, expected_mod: tvm.IRModule, tile_size: int
    ):
        extracted_task = extract_layout_transform_task(relay_mod, TARGET)
        dispatched_mod = extracted_task.dispatched[0]
        sch = tvm.tir.Schedule(dispatched_mod)
        block = sch.get_block("T_layout_trans")
        output_sch = cuda_layout_transform_schedule_rule(sch, block, [tile_size])[0]
        assert output_sch.mod.script() == expected_mod.script()

    def test_simple_tiling(self):
        mod = create_relay_module([1, 32, 32, 32], "float16", [("NCHW", "NHWC")])

        # Main things to notice:
        # - two blocks each with 16, 16 extents which write/read shared mem
        # - coalesced accesses in inner loop of global memory buffer for both
        # fmt: off
        @I.ir_module
        class ExpectedModule:
            @T.prim_func
            def main(p0: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16"), T_layout_trans: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16")):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                # with T.block("root"):
                p0_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16", scope="shared")
                for ax0_ax2_ax1_0_ax3_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
                    for ax3_1_fused_0_ax3_1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for ax1_1_fused_0_ax1_1_fused_1_fused in range(T.int64(16)):
                            with T.block("p0_shared"):
                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v1 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + ax1_1_fused_0_ax1_1_fused_1_fused)
                                v2 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused // T.int64(4))
                                v3 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused % T.int64(2) * T.int64(16) + ax3_1_fused_0_ax3_1_fused_1_fused)
                                T.reads(p0[v0, v1, v2, v3])
                                T.writes(p0_shared[v0, v1, v2, v3])
                                p0_shared[v0, v1, v2, v3] = p0[v0, v1, v2, v3]
                    for ax0_ax1_fused_0 in range(T.int64(16)):
                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            with T.block("T_layout_trans"):
                                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_ax1 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused // T.int64(4))
                                v_ax2 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused % T.int64(2) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) // T.int64(16))
                                v_ax3 = T.axis.spatial(T.int64(32), ax0_ax2_ax1_0_ax3_0_fused % T.int64(4) // T.int64(2) * T.int64(16) + (ax0_ax1_fused_0 * T.int64(16) + ax0_ax1_fused_1) % T.int64(16))
                                T.reads(p0_shared[v_ax0, v_ax3, v_ax1, v_ax2])
                                T.writes(T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3])
                                T.block_attr({"dst_layout": "NHWC", "input_shape": [1, 32, 32, 32], "schedule_rule": "layout_transform", "src_layout": "NCHW"})
                                T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(v_ax0 < T.int64(1) and v_ax3 < T.int64(32) and v_ax1 < T.int64(32) and v_ax2 < T.int64(32), p0_shared[v_ax0, v_ax3, v_ax1, v_ax2], T.float16(0))

        self.assert_extracted_equals_expected(mod, ExpectedModule, 16)

    def test_simple_implicit_reshape(self):
        mod = create_relay_module([1, 32, 32, 32], "float16", [("NCHW", "NCHW4c")])

        # Main things to notice:
        # - two blocks each with 16, 16 extents which write/read shared mem
        # - coalesced accesses in inner loop of global memory buffer for both
        # - an implicit reshape is done (see p0_shared)
        # fmt: off
        @I.ir_module
        class ExpectedModule:
            @T.prim_func
            def main(p0: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16"), T_layout_trans: T.Buffer((T.int64(1), T.int64(8), T.int64(32), T.int64(32), T.int64(4)), "float16")):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                # with T.block("root"):
                p0_shared = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(4), T.int64(32), T.int64(32)), "float16", scope="shared")
                for ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
                    for ax3_1_fused_0_ax3_1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused in range(T.int64(16)):
                            with T.block("p0_shared"):
                                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_ax1 = T.axis.spatial(T.int64(8), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused // T.int64(16))
                                v_ax2 = T.axis.spatial(T.int64(32), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused % T.int64(16) * T.int64(2) + ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused // T.int64(8))
                                v_ax3 = T.axis.spatial(T.int64(32), ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused % T.int64(8) // T.int64(4) * T.int64(16) + ax3_1_fused_0_ax3_1_fused_1_fused)
                                v_ax4 = T.axis.spatial(T.int64(4), ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused % T.int64(4))
                                T.reads(p0[v_ax0, v_ax1 * T.int64(4) + v_ax4, v_ax2, v_ax3])
                                T.writes(p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3])
                                p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3] = p0[v_ax0, v_ax1 * T.int64(4) + v_ax4, v_ax2, v_ax3]
                    for ax0_ax1_ax2_fused_0 in range(T.int64(16)):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            with T.block("T_layout_trans"):
                                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_ax1 = T.axis.spatial(T.int64(8), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused // T.int64(16))
                                v_ax2 = T.axis.spatial(T.int64(32), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused % T.int64(16) * T.int64(2) + (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) // T.int64(128))
                                v_ax3 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) % T.int64(128) // T.int64(4))
                                v_ax4 = T.axis.spatial(T.int64(4), (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) % T.int64(4))
                                T.reads(p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3])
                                T.writes(T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                                T.block_attr({"dst_layout": "NCHW4c", "input_shape": [1, 32, 32, 32], "schedule_rule": "layout_transform", "src_layout": "NCHW"})
                                T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.if_then_else(v_ax0 < T.int64(1) and v_ax1 * T.int64(4) + v_ax4 < T.int64(32) and v_ax2 < T.int64(32) and v_ax3 < T.int64(32), p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3], T.float16(0))
        self.assert_extracted_equals_expected(mod, ExpectedModule, 16)

    def test_expected_fusion_post(self):
        mod = create_relay_module(
            [1, 32, 32, 32], "float16", [("NCHW", "NCHW4c"), OpPattern.BROADCAST]
        )

        # Main things to notice:
        # - two blocks each with 16, 16 extents which write/read shared mem
        # - coalesced accesses in inner loop of global memory buffer for both
        # - an implicit reshape is done (see p0_shared)
        # - an addition is inlined in the final block (p1 input)
        # fmt: off
        @I.ir_module
        class ExpectedModule:
            @T.prim_func
            def main(p0: T.Buffer((T.int64(1), T.int64(32), T.int64(32), T.int64(32)), "float16"), p1: T.Buffer((), "float16"), T_add: T.Buffer((T.int64(1), T.int64(8), T.int64(32), T.int64(32), T.int64(4)), "float16")):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                # with T.block("root"):
                p0_shared = T.alloc_buffer((T.int64(1), T.int64(8), T.int64(4), T.int64(32), T.int64(32)), "float16", scope="shared")
                for ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
                    for ax3_1_fused_0_ax3_1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                        for ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused in range(T.int64(16)):
                            with T.block("p0_shared"):
                                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_ax1 = T.axis.spatial(T.int64(8), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused // T.int64(16))
                                v_ax2 = T.axis.spatial(T.int64(32), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused % T.int64(16) * T.int64(2) + ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused // T.int64(8))
                                v_ax3 = T.axis.spatial(T.int64(32), ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused % T.int64(8) // T.int64(4) * T.int64(16) + ax3_1_fused_0_ax3_1_fused_1_fused)
                                v_ax4 = T.axis.spatial(T.int64(4), ax2_1_ax3_0_1_ax4_1_fused_0_ax2_1_ax3_0_1_ax4_1_fused_1_fused % T.int64(4))
                                T.reads(p0[v_ax0, v_ax1 * T.int64(4) + v_ax4, v_ax2, v_ax3])
                                T.writes(p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3])
                                p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3] = p0[v_ax0, v_ax1 * T.int64(4) + v_ax4, v_ax2, v_ax3]
                    for ax0_ax1_ax2_fused_0 in range(T.int64(16)):
                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                            with T.block("T_layout_trans"):
                                v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                                v_ax1 = T.axis.spatial(T.int64(8), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused // T.int64(16))
                                v_ax2 = T.axis.spatial(T.int64(32), ax0_ax1_ax2_0_ax4_0_ax3_0_0_fused % T.int64(16) * T.int64(2) + (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) // T.int64(128))
                                v_ax3 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) % T.int64(128) // T.int64(4))
                                v_ax4 = T.axis.spatial(T.int64(4), (ax0_ax1_ax2_fused_0 * T.int64(16) + ax0_ax1_ax2_fused_1) % T.int64(4))
                                T.reads(p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3], p1[()])
                                T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                                T.block_attr({"dst_layout": "NCHW4c", "input_shape": [1, 32, 32, 32], "schedule_rule": "layout_transform", "src_layout": "NCHW"})
                                T_add[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.if_then_else(v_ax0 < T.int64(1) and v_ax1 * T.int64(4) + v_ax4 < T.int64(32) and v_ax2 < T.int64(32) and v_ax3 < T.int64(32), p0_shared[v_ax0, v_ax1, v_ax4, v_ax2, v_ax3], T.float16(0)) + p1[()]
        self.assert_extracted_equals_expected(mod, ExpectedModule, 16)


if __name__ == "__main__":
    tvm.testing.main()
