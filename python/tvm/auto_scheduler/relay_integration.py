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
# pylint: disable=unused-variable,invalid-name

"""
Integrate auto_scheduler into relay. It implements the following items:
1. Extract search tasks from a relay program
2. Provide auto-scheduling for all TOPI compute functions
"""

import json
import logging
import threading
import warnings

import tvm
from tvm import autotvm, transform
from tvm.ir.transform import PassContext
from tvm.runtime import convert_to_object
from tvm.target import Target
from tvm.te.tensor import ComputeOp, PlaceholderOp, Tensor
from tvm.tir import Reduce
from tvm.tir import expr as _expr

from . import _ffi_api
from .compute_dag import ComputeDAG, LayoutRewriteOption
from .dispatcher import DispatchContext
from .search_task import SearchTask
from .utils import get_const_tuple
from .workload_registry import register_workload_tensors

logger = logging.getLogger("auto_scheduler")


def call_all_topi_funcs(mod, params, target, opt_level=3):
    """Call all TOPI compute to extract auto_scheduler tasks in a Relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    # Turn off AutoTVM config not found warnings
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    with transform.PassContext(
        opt_level=opt_level,
        config={
            "relay.backend.use_auto_scheduler": True,
        },
        disabled_pass={"AutoSchedulerLayoutRewrite"},
    ):
        compiler = relay.vm.VMCompiler()
        if params:
            compiler.set_params(params)
        mod = tvm.IRModule.from_expr(mod) if isinstance(mod, relay.Function) else mod
        compiler.lower(mod, target)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent


def extract_tasks(
    mod,
    params,
    target,
    target_host=None,
    hardware_params=None,
    include_simple_tasks=False,
    dump_workload_to_dag_log=None,
    opt_level=3,
):
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod: tvm.IRModule or relay.function.Function
        The module or function to tune
    params: dict of str to numpy array
        The associated parameters of the program
    target: Union[tvm.target.Target, str]
        The compilation target
    target_host: Optional[Union[tvm.target.Target, str]]
        The host compilation target
    hardware_params : Optional[HardwareParams]
        Hardware parameters used for the search tasks
    include_simple_tasks: bool
        Whether to extract simple tasks that do not include complicated ops.
    dump_workload_to_dag_log: Optional[str]
        A file to dump an association between the workload keys and the actual DAG
    opt_level : Optional[int]
        The optimization level of the task extractions.

    Returns
    -------
    tasks: List[SearchTask]
        The tasks in this network
    weights: List[int]
        The weight (i.e. the number of appearance) of extracted tasks
    """
    # pylint: disable=import-outside-toplevel
    if target_host is not None:
        warnings.warn(
            "target_host parameter is going to be deprecated. "
            "Please pass in tvm.target.Target(target, host=target_host) instead."
        )

    target, target_host = Target.check_and_update_host_consist(target, target_host)

    # Run the compiler to collect all TOPI calls during compilation.
    env = TracingEnvironment(
        TracingMode.EXTRACT_TASK if include_simple_tasks else TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )

    dispatch_ctx = DispatchContext.current
    old_verbose = dispatch_ctx.verbose
    dispatch_ctx.verbose = 0
    with env:
        # Wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(
            target=call_all_topi_funcs, args=(mod, params, target, opt_level)
        )
        build_thread.start()
        build_thread.join()
    dispatch_ctx.verbose = old_verbose

    # create search tasks
    tasks = []
    weights = []
    for wkl_key, (weight, func_names) in env.wkl_key_to_weight.items():
        tasks.append(
            SearchTask(
                workload_key=wkl_key,
                target=target,
                hardware_params=hardware_params,
                # When auto scheduler is used in end to end network, try to apply layout rewrite
                # to improve the overall performance
                layout_rewrite_option=LayoutRewriteOption.get_target_default(target, True),
                task_inputs=(
                    env.wkl_key_to_input_names[wkl_key]
                    if wkl_key in env.wkl_key_to_input_names
                    else None
                ),
                task_inputs_save_to_file=True,
                desc=",".join(func_names),
            )
        )
        weights.append(int(weight))

    if dump_workload_to_dag_log is not None:
        with open(dump_workload_to_dag_log, "w") as f:
            json.dump({task.workload_key: str(task.compute_dag) for task in tasks}, f)

    return tasks, weights


class TracingMode:
    """Two modes for tracing"""

    EXTRACT_TASK = 0  # trace all topi calls to extract tasks
    # same as EXTRACT_TASK but ignore the task without complex ops
    EXTRACT_COMPLEX_TASK_ONLY = 1
    PREPARE_LAYOUT_REWRITE = 2  # trace topi calls to prepare layout rewrite


class TracingEnvironment:
    """Global environment for tracing all topi function calls"""

    current = None

    def __init__(self, tracing_mode):
        self.tracing_mode = tracing_mode
        self.relay_disable_build_cache = "false"
        self.func_name_to_wkl_key = {}
        self.wkl_key_to_weight = {}
        self.wkl_key_to_input_names = {}

    def __enter__(self):
        TracingEnvironment.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TracingEnvironment.current = None

    def add_workload_key(self, func_name, workload_key):
        """Add the workload key of a search task.

        Parameters
        ----------
        func_name: str
            The function name of the task.

        workload_key: str
            The workload key of a task.
        """
        self.func_name_to_wkl_key[func_name] = workload_key
        if workload_key not in self.wkl_key_to_weight:
            self.wkl_key_to_weight[workload_key] = (0, set())
        weight, func_names = self.wkl_key_to_weight[workload_key]
        func_names.add(func_name)
        self.wkl_key_to_weight[workload_key] = (weight + 1, func_names)

    def add_workload_input_names(self, workload_key, input_names):
        """Add special task inputs to this workload.

        Parameters
        ----------
        workload_key : str
            The workload key of a task.

        input_names : List[str]
            A list of input names.
        """
        self.wkl_key_to_input_names[workload_key] = input_names


@tvm._ffi.register_func("auto_scheduler.enter_layout_rewrite")
def enter_layout_rewrite():
    """Enter layout rewrite tracing environment"""
    env = TracingEnvironment(TracingMode.PREPARE_LAYOUT_REWRITE)
    env.__enter__()


@tvm._ffi.register_func("auto_scheduler.exit_layout_rewrite")
def exit_layout_rewrite():
    """Exit layout rewrite tracing environment"""
    env = TracingEnvironment.current
    env.__exit__(None, None, None)


def traverse_to_get_io_tensors(outs):
    """Traverse from a list of output tensors to get input/output tensors and
    other useful information.

    Parameters
    ----------
    outs: List[Tensor]
        The output tensors

    Returns
    -------
    io_tensors: List[Tensor]
        The input and output tensors with static shape
    has_layout_free: bool
        Whether the compute DAG has layout_free placeholders
    has_complex_op: bool
        Whether the topi compute function includes at least one complex (reduce) op
    """
    layout_free_ops = []
    inputs = []

    has_complex_op = False
    visited = set()

    def traverse(t):
        nonlocal has_complex_op

        # We cannot directly add tensors to the set, because the comparison of
        # two tensors with ndim=0 is ambiguous.
        assert t.handle is not None
        if t.handle.value in visited:
            return
        if isinstance(t.op, PlaceholderOp):
            inputs.append(t)
        elif isinstance(t.op, ComputeOp):
            has_complex_op = has_complex_op or any([isinstance(e, Reduce) for e in t.op.body])
            if "layout_free_placeholders" in t.op.attrs:
                layout_free_ops.append(t.op)
            for x in t.op.input_tensors:
                traverse(x)
        visited.add(t.handle.value)

    for t in outs:
        traverse(t)

    io_tensors = inputs + list(outs)
    for tensor in io_tensors:
        # Reject the compute if any of its I/O tensors has dynamic shape.
        if any([not isinstance(v, int) for v in get_const_tuple(tensor.shape)]):
            return ([], False, False)

    return (io_tensors, len(layout_free_ops) > 0, has_complex_op)


@tvm._ffi.register_func("auto_scheduler.relay_integration.auto_schedule_topi_compute")
def auto_schedule_topi(func_name, outs):
    """Use auto-scheduler to schedule any topi compute function.

    Note: This is used internally for relay integration. Do
    not use this as a general user-facing API.

    Parameters
    ----------
    func_name: str
        The name of the function being scheduled.

    outs: List[Tensor]
        The output tensors of topi compute functions

    Returns
    -------
    sch: Optional[te.Schedule]
        A tuned schedule or none (if not tuned) in the final build mode;
        None in the tracing mode so that the fallback topi schedule will be used.
    """

    # pylint: disable=import-outside-toplevel
    from tvm.auto_scheduler.measure import (
        prepare_input_map,
    )  # lazily import to avoid recursive dependency

    io_tensors, has_layout_free, has_complex_op = traverse_to_get_io_tensors(outs)
    if not io_tensors:  # The compute includes dynamic shapes which are not supported yet.
        return None

    try:
        dag = ComputeDAG(io_tensors)
    except tvm.error.TVMError as err:
        logger.info("Failed to create a ComputeDAG for auto_scheduler: %s", str(err))
        return None

    key = register_workload_tensors(dag.workload_key(), io_tensors)
    target = tvm.target.Target.current()

    dispatch_ctx = DispatchContext.current
    state = dispatch_ctx.query(target, key, has_complex_op, dag, func_name)
    schedule = None

    env = TracingEnvironment.current
    if env is None:
        # in the final build mode
        if state is None:
            return None

        schedule, _ = dag.apply_steps_from_state(state)
        return schedule

    if env.tracing_mode in [TracingMode.EXTRACT_TASK, TracingMode.EXTRACT_COMPLEX_TASK_ONLY]:
        # in the task extraction mode
        if has_complex_op or env.tracing_mode == TracingMode.EXTRACT_TASK:
            env.add_workload_key(func_name, key)
            input_map = prepare_input_map(io_tensors)
            if input_map:
                env.add_workload_input_names(key, list(input_map.values()))
    elif env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
        # in prepare_layout_rewrite mode
        if (
            LayoutRewriteOption.get_target_default(target, True) != LayoutRewriteOption.NO_REWRITE
            and has_layout_free
        ):
            if state is None:
                return None

            # rewrite the layout and update the context for the new dag
            new_dag = dag.rewrite_layout_from_state(state)
            new_key = new_dag.workload_key()
            if new_key != key:
                dispatch_ctx.update(target, new_key, state)
    else:
        raise ValueError("Invalid tracing mode: " + env.tracing_mode)

    return schedule


@tvm._ffi.register_func("auto_scheduler.relay_integration.te_compiler_update_weights")
def te_compiler_update_weights(function_weights):
    """A callback for updating the weights of extracted tasks. When using the TE compiler
    that avoids compiling the same function multiple times by caching, all extracted tasks
    have weight 1, so the TE compiler invokes this callback at the end. In this case,
    we override existing weights with the use_count in TE compiler cache.

    Parameters
    ----------
    function_weights: Dict[str, int]
        Mapping from function names to their weights.
    """
    env = TracingEnvironment.current
    if env is not None:
        # Override this map with the weights in the TE compiler.
        env.wkl_key_to_weight = {}

        for func_name, weight in function_weights.items():
            # If the function name is not in the map, then it means we are not interested in
            # this function during task extraction (e.g., a function without reduction).
            if func_name not in env.func_name_to_wkl_key:
                continue

            workload_key = env.func_name_to_wkl_key[func_name]
            if workload_key not in env.wkl_key_to_weight:
                env.wkl_key_to_weight[workload_key] = (0, set())

            # Note that the function appears multiple times in a model will be renamed
            # to make sure function names are unique, so we use the workload key generated
            # from the function's TE compute to determine their weights.
            old_weight, func_names = env.wkl_key_to_weight[workload_key]
            func_names.add(func_name)
            env.wkl_key_to_weight[workload_key] = (old_weight + weight, func_names)


def tensor_no_check_call(self, *indices):
    """An indexing function without any check.
    This is the same as `tvm.te.Tensor::__call__` except that the safety
    check is removed.
    """
    indices = convert_to_object(indices)
    args = []
    for x in indices:
        if isinstance(x, _expr.PrimExpr):
            args.append(x)
        elif isinstance(x, _expr.IterVar):
            args.append(x.var)
        else:
            raise ValueError("The indices must be expression")

    return _expr.ProducerLoad(self, args)


def remove_index_check(tensor):
    """Remove the safety check in the indexing function for a tensor.
    This is done by monkey patching its indexing function.
    After removing the check, we are allowed to create a
    temporary wrong IR and fix it later in other places.

    Parameters
    ----------
    tensor: Tensor
      The tensor to remove index check.
    """
    # Monkey patch the indexing function
    tensor.__call__ = tensor_no_check_call.__get__(tensor, Tensor)


def rewrite_compute_body(compute_tensor, new_layout):
    """Rewrite the body of a ComputeOp according to a new layout of a placeholder"""
    op = compute_tensor.op

    # Get layout free placeholders
    layout_free_placeholders = op.attrs["layout_free_placeholders"]
    assert len(layout_free_placeholders) == 1, "Only support one layout free placeholder"
    placeholder_op = layout_free_placeholders[0].op

    # Rewrite the index expression in body
    body = []
    for b in op.body:
        body.append(_ffi_api.RewriteIndexForNewLayout(placeholder_op, new_layout, b))
    op_node = tvm.te._ffi_api.ComputeOp(op.name, op.tag, op.attrs, op.axis, body)

    num = op_node.num_outputs
    outputs = tuple(op_node.output(i) for i in range(num))
    return outputs[0] if num == 1 else outputs


def is_auto_scheduler_enabled():
    """Return whether the auto-scheduler is enabled.

    Parameters
    ----------
    enabled: bool
        Whether the auto-scheduler is enabled
    """
    return PassContext.current().config.get("relay.backend.use_auto_scheduler", False)
