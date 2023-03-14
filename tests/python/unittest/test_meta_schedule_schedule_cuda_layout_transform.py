# Edge Cases:
# 1. Fusion with ops
# 2. Fusion with ops

# Properties to test for
# 1. Compiling -- compiles well without crashing
# 2. Correctness when running
# 3. Autotuning ability


import itertools
import random
from typing import List, Optional, Tuple, Union

import numpy as np

import tvm
from tvm import meta_schedule, relay
from tvm.meta_schedule.schedule.cuda.layout_transform import cuda_layout_transform_schedule_rule
from tvm.relay.op import OpPattern
from tvm.tir.schedule import BlockRV, ExprRV, LoopRV


# Create unary functions which apply ops with compatible fusion levels
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


# Applying the actual layout transform will be different
def apply_layout_transform(data: relay.Expr, src_layout: str, dst_layout: str):
    assert relay.op.get("layout_transform").get_attr("TOpPattern") == OpPattern.INJECTIVE
    return relay.layout_transform(data, src_layout, dst_layout)


# These are the only levels of op which can possibly be fused with layout_transform (which injective)
extra_pattern_level_to_op = {
    OpPattern.ELEMWISE: apply_elemwise_clip,
    OpPattern.BROADCAST: apply_broadcast_add,
    OpPattern.INJECTIVE: apply_injective_concatenate,
    OpPattern.COMM_REDUCE: apply_comm_reduce_max,
}


class PatchCustomLayoutTransformScheduleRule:
    """Patch the custom layout transform schedule to test only specific tile sizes."""

    FUNC_NAME = "meta_schedule.cuda.layout_transform"

    def __init__(self, tile_sizes: List[int]) -> None:
        self.tile_sizes = tile_sizes
        self.old_func = None

    def __enter__(self, *args, **kwargs) -> None:
        self.old_func = tvm.get_global_func(self.FUNC_NAME)

        def new_layout_rule(
            sch: tvm.tir.Schedule, block: BlockRV, tile_sizes: Optional[List[int]] = self.tile_sizes
        ) -> List[tvm.tir.Schedule]:
            return cuda_layout_transform_schedule_rule(sch, block, tile_sizes)

        tvm.register_func(self.FUNC_NAME, new_layout_rule, override=True)

    def __exit__(self, *args, **kwargs) -> None:
        tvm.register_func(self.FUNC_NAME, self.old_func, override=True)


def create_relay_module(
    input_shape: List[int], dtype: str, ops: List[Union[int, Tuple[str, str]]]
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
            cur_data = extra_pattern_level_to_op[op_info](cur_data)

    relay.transform.InferTypeLocal(cur_data)
    return tvm.IRModule.from_expr(cur_data)


def generate_test_case(
    input_shape: List[int],
    implicit_reshape_info: Optional[Tuple[int, int]],
    dtype: str,
    num_additional_ops: int,
) -> tvm.IRModule:
    # Create layout transforms
    rank = len(input_shape)
    src_layout = "".join([chr(i + ord("A")) for i in range(rank)])

    dst_layout = list(src_layout)
    if implicit_reshape_info:
        axis_to_reshape, size_new_dim = implicit_reshape_info
        cur_dim = dst_layout[axis_to_reshape]
        dst_layout[axis_to_reshape] = f"{cur_dim}{size_new_dim}{cur_dim.lower()}"

    random.shuffle(dst_layout)
    while "".join(dst_layout) == src_layout:
        random.shuffle(dst_layout)
    dst_layout = "".join(dst_layout)

    op_choices = random.choices(list(extra_pattern_level_to_op.keys()), k=num_additional_ops)
    op_choices.append((src_layout, dst_layout))

    random.shuffle(op_choices)
    return create_relay_module(input_shape, dtype, op_choices)


def extract_layout_transform_task(
    mod: tvm.IRModule, target: tvm.target.Target
) -> Tuple[tvm.IRModule, tvm.IRModule]:
    """Given a relay IRModule, return the PrimFunc IRModule with fused layout transform task."""
    extracted_tasks = meta_schedule.relay_integration.extract_tasks(
        mod,
        target,
        {},
        pass_config={
            "relay.backend.use_meta_schedule": True,
            "relay.FuseOps.max_depth": 30,
            "relay.backend.tir_converter": "default",
        },
    )
    task_of_interest = None
    for task in extracted_tasks:
        if "layout_transform" in task.task_name:
            task_of_interest = task
            break
    assert task_of_interest is not None

    # Fused layout transform task
    relay_mod = task_of_interest.mod
    dispatched_mod = task_of_interest.dispatched[0]
    return relay_mod, dispatched_mod


def run_primfunc(
    primfunc_mod: tvm.IRModule, target: tvm.target.Target, input_tensors: List[tvm.nd.NDArray]
):
    with tvm.transform.PassContext(
        config={
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": False,
            "relay.FuseOps.max_depth": 30,
        },
        opt_level=3,
    ):
        lib = tvm.build(primfunc_mod, target=target)
    lib(*input_tensors)


def verify_layout_transform_task(
    relay_mod: tvm.IRModule,
    dispatched_mod: tvm.IRModule,
    target: tvm.target.Target,
    tile_sizes: List[int],
):
    """Given a layout transform primfunc, tests the given tile_sizes and verifies output matches."""
    space_generator = meta_schedule.space_generator.PostOrderApply(
        sch_rules=meta_schedule.schedule_rule.schedule_rule.create("cuda"),
        postprocs=meta_schedule.postproc.postproc.create("cuda"),
        mutator_probs=meta_schedule.mutator.mutator.create("cuda"),
    )
    device = tvm.cuda(0)

    func_type = relay.transform.InferTypeLocal(relay_mod[relay_mod.get_global_vars()[0]])
    input_tensors = []
    for input_type in func_type.arg_types:
        orig_input_np = np.random.uniform(0, 10, size=list(map(int, input_type.shape))).astype(
            input_type.dtype
        )
        input_tensors.append(tvm.nd.array(orig_input_np, device))
    ret_type = func_type.ret_type

    def get_output_tensor() -> Tuple[tvm.nd.NDArray, tvm.nd.NDArray]:
        numpy_init = np.random.uniform(0, 1000, size=list(map(int, ret_type.shape))).astype(
            ret_type.dtype
        )
        return tvm.nd.array(numpy_init, device)

    def run_and_get_output(tile_size: Optional[int]) -> np.ndarray:
        # By setting the tile_sizes to search to nothing, the layout transform rule just returns
        # the original schedule.
        tile_size_input = [] if tile_size is None else [tile_size]
        with PatchCustomLayoutTransformScheduleRule(tile_sizes=tile_size_input):
            tune_context = meta_schedule.TuneContext(
                mod=dispatched_mod,
                target=target,
                space_generator=space_generator,
                search_strategy=meta_schedule.search_strategy.create(),
            )
            tune_context.pre_tuning(32)
            returned_primfunc = tune_context.generate_measure_candidates()[0].sch.mod
            output_tensor = get_output_tensor()
            run_primfunc(returned_primfunc, target, [*input_tensors, output_tensor])
            # print(returned_primfunc)
            return output_tensor.numpy()

    # Passing None, we basically do not apply the custom rule we have created.
    ground_truth_np = run_and_get_output(None)
    for tile_size in tile_sizes:
        experimental_result_np = run_and_get_output(tile_size)

        np.testing.assert_allclose(ground_truth_np, experimental_result_np)


def generate_all_test_case(
    # Each has ~10k elements
    input_shapes: List[List[int]] = [
        [12, 48, 18],
        [890, 14],
        [10, 12, 2, 5, 3, 3],
    ],
    implicit_reshape_conditions: List[Optional[Tuple[int, int]]] = [None, (0, 2), (1, 2)],
    dtypes: List[str] = ["float32", "float16"],
    num_additional_ops: int = 0,
    tile_sizes: List[int] = [32, 20, 19],
    repeats_per_condition=10,
):
    # Small numbers which should work for nearly every (modern-ish) gpu.
    target = tvm.target.Target(
        "cuda -max_threads_per_block=32 -max_num_threads=128 -thread_warp_size=32 -max_shared_memory_per_block=8192 -registers_per_block=1024"
    )
    for _ in range(repeats_per_condition):
        for input_shape, implicit_reshape_info, dtype in itertools.product(
            input_shapes, implicit_reshape_conditions, dtypes
        ):
            # Generate random module of fusable ops + layout transform and extract fused layout transform task
            full_mod = generate_test_case(
                input_shape, implicit_reshape_info, dtype, num_additional_ops
            )

            # Fused layout transform task
            relay_mod, dispatched_mod = extract_layout_transform_task(full_mod, target)

            print(relay_mod)
            verify_layout_transform_task(relay_mod, dispatched_mod, target, tile_sizes)
            print("Verified!")
            print()


if __name__ == "__main__":
    # mod = create_relay_module([12, 48, 18], "float32", [("ABC", "B2bAC"), 2])
    # extracted_tasks = meta_schedule.relay_integration.extract_tasks(
    #     mod,
    #     tvm.target.Target("cuda"),
    #     {},
    #     pass_config={
    #         "relay.backend.use_meta_schedule": True,
    #         "relay.FuseOps.max_depth": 30,
    #         "relay.backend.tir_converter": "default",
    #     },
    # )
    # task_of_interest = None
    # for task in extracted_tasks:
    #     if "layout_transform" in task.task_name:
    #         task_of_interest = task
    #         break
    # assert task_of_interest is not None

    # # # Fused layout transform task
    # dispatched_mod = task_of_interest.dispatched[0]
    # base_schedule = tvm.tir.Schedule(dispatched_mod)
    # verify_schedule(base_schedule, [30, 20, 19])

    # exit()

    generate_all_test_case()
