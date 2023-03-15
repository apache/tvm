# Edge Cases:
# 1. Fusion with ops
# 2. Fusion with ops

# Properties to test for
# 1. Compiling -- compiles well without crashing
# 2. Correctness when running
# 3. Autotuning ability


import itertools
import random
import tempfile
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import meta_schedule, relay
from tvm.meta_schedule.schedule.cuda.layout_transform import cuda_layout_transform_schedule_rule
from tvm.relay.op import OpPattern
from tvm.tir.schedule import BlockRV, ExprRV, LoopRV


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


class TestRandomRelayE2ECorrectness:
    """Tests E2E correctness of layout transform schedule.

    Randomly generates relay mod with layout transform and fusable ops. Checks the
    layout transform task for correctness by comparing against its unscheduled result.
    """

    # Create unary functions which apply ops with compatible fusion levels to layout transform
    @staticmethod
    def get_random_axis(data: relay.Expr):
        rank = len(relay.transform.InferTypeLocal(data).shape)
        return random.randint(0, rank - 1)

    @staticmethod
    def apply_elemwise_clip(data: relay.Expr, min=0, max=10):
        assert relay.op.get("clip").get_attr("TOpPattern") == OpPattern.ELEMWISE
        return relay.clip(data, min, max)

    @staticmethod
    def apply_broadcast_add(data: relay.Expr, val_to_add=5):
        assert relay.op.get("add").get_attr("TOpPattern") == OpPattern.BROADCAST
        type_info = relay.transform.InferTypeLocal(data)
        return relay.add(data, relay.const(val_to_add, dtype=type_info.dtype))

    @staticmethod
    def apply_injective_concatenate(data: relay.Expr, axis=None):
        if axis is None:
            axis = TestRandomRelayE2ECorrectness.get_random_axis(data)
        assert relay.op.get("concatenate").get_attr("TOpPattern") == OpPattern.INJECTIVE
        return relay.concatenate([data, data], axis)

    @staticmethod
    def apply_comm_reduce_max(data: relay.Expr, axis=None):
        if axis is None:
            axis = TestRandomRelayE2ECorrectness.get_random_axis(data)
        assert relay.op.get("max").get_attr("TOpPattern") == OpPattern.COMM_REDUCE

        # Do this to maintain dimensions
        return relay.add(data, relay.max(data, axis, keepdims=True))

    @staticmethod
    def get_map_pattern_level_to_op() -> Dict[OpPattern, Callable]:
        # These are the only levels of op which can possibly be fused with layout_transform (which injective)
        return {
            OpPattern.ELEMWISE: TestRandomRelayE2ECorrectness.apply_elemwise_clip,
            OpPattern.BROADCAST: TestRandomRelayE2ECorrectness.apply_broadcast_add,
            OpPattern.INJECTIVE: TestRandomRelayE2ECorrectness.apply_injective_concatenate,
            OpPattern.COMM_REDUCE: TestRandomRelayE2ECorrectness.apply_comm_reduce_max,
        }

    @staticmethod
    def apply_layout_transform(data: relay.Expr, src_layout: str, dst_layout: str):
        assert relay.op.get("layout_transform").get_attr("TOpPattern") == OpPattern.INJECTIVE
        return relay.layout_transform(data, src_layout, dst_layout)

    @staticmethod
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
                cur_data = TestRandomRelayE2ECorrectness.apply_layout_transform(
                    cur_data, src_layout, dst_layout
                )
            else:
                cur_data = TestRandomRelayE2ECorrectness.get_map_pattern_level_to_op()[op_info](
                    cur_data
                )

        relay.transform.InferTypeLocal(cur_data)
        return tvm.IRModule.from_expr(cur_data)

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
            list(TestRandomRelayE2ECorrectness.get_map_pattern_level_to_op().keys()),
            k=num_additional_ops,
        )

        # Append tuple, representing layout transfomr from src --> dst layout
        op_order.append((src_layout, dst_layout))

        random.shuffle(op_order)
        return TestRandomRelayE2ECorrectness.create_relay_module(input_shape, dtype, op_order)

    @staticmethod
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

    @staticmethod
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
            TestRandomRelayE2ECorrectness.run_primfunc(
                returned_primfunc, target, [*input_tensors, output_tensor]
            )
            return output_tensor.numpy()

        # Passing None, we basically do not apply the custom rule we have created
        # and instead use the old default schedule which is the ground truth.
        ground_truth_np = run_and_get_output(None)

        for tile_size in tile_sizes:
            experimental_np = run_and_get_output(tile_size)
            np.testing.assert_allclose(ground_truth_np, experimental_np)

    input_shape, implicit_reshape_info, dtype, tile_sizes = tvm.testing.parameters(
        *itertools.product(
            # InputShapes: Each has ~10k elements, should take single microseconds on modern gpu
            [
                [12, 48, 18],
                [890, 14],
                [10, 12, 2, 5, 3, 3],
            ],
            # Implicit reshape conditions.
            # None is do no implicit reshape, (0, 2) means divide axis 0 in half, e.g. AB --> A2aB
            [None, (0, 2), (1, 2)],
            # Dtypes to test, should not matter that much
            ["float16"],
            # Tile sizes to try
            [[8, 7]],
        )
    )

    @tvm.testing.requires_gpu
    def test_all_test_case(
        self,
        input_shape,
        implicit_reshape_info,
        dtype,
        tile_sizes,
        # number of non-layout transform ops to include and may be fused
        num_additional_ops: int = 5,
    ):
        """Tests the product of all conditions `repeat_per_condition` times."""
        # Small gpu parameters which should work for nearly every (modern-ish) gpu.
        target = tvm.target.Target(
            "cuda -max_threads_per_block=32 -max_num_threads=128 -thread_warp_size=32 -max_shared_memory_per_block=8192 -registers_per_block=1024"
        )

        # Generate random module of fusable ops + layout transform and extract fused layout transform task
        full_mod = self.generate_test_case(
            input_shape, implicit_reshape_info, dtype, num_additional_ops
        )

        # Fused layout transform task
        extracted_task = self.extract_layout_transform_task(full_mod, target)
        print(full_mod)
        print(extracted_task.task_name)
        self.verify_layout_transform_task(extracted_task, target, tile_sizes)
        print("Done!")
        print()


if __name__ == "__main__":
    tvm.testing.main()
