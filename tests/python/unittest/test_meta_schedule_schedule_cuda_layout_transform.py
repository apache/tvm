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

import tvm
from tvm import meta_schedule, relay
from tvm.meta_schedule.schedule.cuda.layout_transform import cuda_layout_transform_schedule_rule
from tvm.relay.op import OpPattern


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
):
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


def verify_schedule(sch: tvm.tir.Schedule, tile_sizes: List[int]):
    block_layout_transform = sch.get_block("T_layout_trans")
    schedules = cuda_layout_transform_schedule_rule(sch, block_layout_transform, tile_sizes)

    assert len(schedules) == len(tile_sizes) + 1

    # This is the default schedule which does not apply the schedule rule
    schedule_baseline = schedules[0]

    # These are the tiled schedules we want to test
    schedule_end = schedules[1:]
    # TODO


def generate_all_test_case(
    # Each has ~10k elements
    input_shapes: List[List[int]] = [
        [12, 48, 18],
        [890, 14],
        [10, 12, 2, 5, 3, 3],
    ],
    implicit_reshape_conditions: List[Optional[Tuple[int, int]]] = [None, (0, 2), (1, 2)],
    dtypes: List[str] = ["float32", "float16"],
    num_additional_ops: int = 1,
    tile_sizes: List[int] = [32, 20, 19],
    repeats_per_condition=10,
):
    for _ in range(repeats_per_condition):
        for input_shape, implicit_reshape_info, dtype in itertools.product(
            input_shapes, implicit_reshape_conditions, dtypes
        ):
            # Generate random module of fusable ops + layout transform and extract fused layout transform task
            mod = generate_test_case(input_shape, implicit_reshape_info, dtype, num_additional_ops)
            extracted_tasks = meta_schedule.relay_integration.extract_tasks(
                mod,
                tvm.target.Target("cuda"),
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
            dispatched_mod = task_of_interest.dispatched[0]
            base_schedule = tvm.tir.Schedule(dispatched_mod)
            print(mod)
            verify_schedule(base_schedule, tile_sizes)


if __name__ == "__main__":
    mod = create_relay_module([890, 14], "float32", [("AB", "BA"), 2])
    extracted_tasks = meta_schedule.relay_integration.extract_tasks(
        mod,
        tvm.target.Target("cuda"),
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

    # # Fused layout transform task
    # dispatched_mod = task_of_interest.dispatched[0]
    # base_schedule = tvm.tir.Schedule(dispatched_mod)
    # verify_schedule(base_schedule, [32, 20, 19])

    breakpoint()

    generate_all_test_case()
