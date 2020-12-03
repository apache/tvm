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

import logging
import json
import threading

import tvm
from tvm import autotvm, te, transform
from tvm.ir.transform import PassContext
from tvm.runtime import convert_to_object
from tvm.te.tensor import ComputeOp, PlaceholderOp, Tensor
from tvm.tir import expr as _expr
from . import _ffi_api
from .compute_dag import ComputeDAG
from .dispatcher import DispatchContext
from .search_task import SearchTask
from .workload_registry import register_workload_tensors

logger = logging.getLogger("auto_scheduler")


def call_all_topi_funcs(mod, params, target):
    """Call all TOPI compute to extract auto_scheduler tasks in a Relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen

    # Turn off AutoTVM config not found warnings
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    with transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_auto_scheduler": True},
        disabled_pass={"AutoSchedulerLayoutRewrite"},
    ):
        opt_mod, _ = relay.optimize(mod, target, params)
        grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
        grc.codegen(opt_mod["main"])

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent


def extract_tasks(
    mod, params, target, target_host=None, hardware_params=None, include_simple_tasks=False
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

    Returns
    -------
    tasks: List[SearchTask]
        The tasks in this network
    weights: List[int]
        The weight (i.e. the number of appearance) of extracted tasks
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    if isinstance(target, str):
        target = tvm.target.Target(target)
    if isinstance(target_host, str):
        target_host = tvm.target.Target(target_host)

    # Run the compiler to collect all TOPI calls during compilation.
    env = TracingEnvironment(
        TracingMode.EXTRACT_TASK if include_simple_tasks else TracingMode.EXTRACT_COMPLEX_TASK_ONLY
    )
    with env:
        # Wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(target=call_all_topi_funcs, args=(mod, params, target))
        build_thread.start()
        build_thread.join()

    # query the compile engine to get the number of occurrence of all tasks
    engine = relay.backend.compile_engine.get()
    use_count_dict = {}
    for k, v in engine.items():
        use_count_dict[k] = v.use_count

    # create search tasks
    tasks = []
    weights = []
    for wkl_key, ccache_key in env.wkl_key_to_ccache_key.items():
        dag = ComputeDAG(wkl_key)
        tasks.append(SearchTask(dag, wkl_key, target, target_host, hardware_params))
        weights.append(use_count_dict[ccache_key] + 1)

    # clean the cached lowering results
    engine.clear()

    return tasks, weights


class TracingMode:
    """Two modes for tracing"""

    EXTRACT_TASK = 0  # trace all topi calls to extract tasks
    EXTRACT_COMPLEX_TASK_ONLY = 1  # same as EXTRACT_TASK but ignore the task without complex ops
    PREPARE_LAYOUT_REWRITE = 2  # trace topi calls to prepare layout rewrite


class TracingEnvironment:
    """Global environment for tracing all topi function calls"""

    current = None

    def __init__(self, tracing_mode):
        self.tracing_mode = tracing_mode
        self.relay_disable_build_cache = "false"
        self.wkl_key_to_ccache_key = {}

    def __enter__(self):
        TracingEnvironment.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TracingEnvironment.current = None

    def add_workload_key(self, workload_key, ccache_key):
        """Add the workload key of a search task

        Parameters
        ----------
        workload_key: str
            The workload key of a task
        ccache_key: CCacheKey
            The corresponding ccache_key of the task
        """
        self.wkl_key_to_ccache_key[workload_key] = ccache_key


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
    """Traverse from a list of output tensors to get both input and output tensors

    Parameters
    ----------
    outs: List[Tensor]
        The output tensors

    Returns
    -------
    io_tensors: List[Tensor]
        The input and output tensors
    has_layout_free: bool
        Whether the compute DAG has layout_free placeholders
    """
    layout_free_ops = []
    inputs = []

    visited = set()

    def traverse(t):
        if t in visited:
            return
        if isinstance(t.op, PlaceholderOp):
            inputs.append(t)
        elif isinstance(t.op, ComputeOp):
            if "layout_free_placeholders" in t.op.attrs:
                layout_free_ops.append(t.op)
            for x in t.op.input_tensors:
                traverse(x)
        visited.add(t)

    for t in outs:
        traverse(t)

    has_layout_free = len(layout_free_ops) > 0
    return inputs + list(outs), has_layout_free


@tvm._ffi.register_func("auto_scheduler.relay_integration.auto_schedule_topi_compute")
def auto_schedule_topi(outs, has_complex_op):
    """Use auto-scheduler to schedule any topi compute function.

    Note: This is used internally for relay integration. Do
    not use this as a general user-facing API.

    Parameters
    ----------
    outs: List[Tensor]
        The output tensors of topi compute functions
    has_complex_op: bool
        Whether the topi compute function includes at least one complex op.

    Returns
    -------
    sch: Optional[te.Schedule]
        A tuned schedule or none (if not tuned) in the final build mode;
        An initial schdule in the tracing mode.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    io_tensors, has_layout_free = traverse_to_get_io_tensors(outs)
    try:
        dag = ComputeDAG(io_tensors)
    except tvm.error.TVMError as err:
        logger.info("Failed to create a ComputeDAG for auto_scheduler: %s", str(err))
        return None

    key = register_workload_tensors(dag.hash_key(), io_tensors)

    # only enable layout rewrite for cpu backend
    target = tvm.target.Target.current()
    enable_layout_rewrite = "cpu" in target.keys

    env = TracingEnvironment.current
    if env is None:
        # in the final build mode
        state = DispatchContext.current.query(target, key, has_complex_op, dag)
        if state is None:
            return None

        schedule, _ = dag.apply_steps_from_state(state)
    elif env.tracing_mode in [TracingMode.EXTRACT_TASK, TracingMode.EXTRACT_COMPLEX_TASK_ONLY]:
        # in the task extraction mode
        if has_complex_op or env.tracing_mode == TracingMode.EXTRACT_TASK:
            engine = relay.backend.compile_engine.get()
            ccache_key = engine.get_current_ccache_key()
            env.add_workload_key(key, ccache_key)
        schedule = te.create_schedule([x.op for x in outs])
    elif env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
        # in prepare_layout_rewrite mode
        if enable_layout_rewrite and has_layout_free:
            dispatch_ctx = DispatchContext.current
            state = dispatch_ctx.query(target, key, has_complex_op, dag)
            if state is None:
                return None

            # rewrite the layout and update the context for the new dag
            dag = ComputeDAG(outs)
            new_dag = dag.rewrite_layout_from_state(state)
            new_key = json.dumps((new_dag.hash_key(),))
            if new_key != key:
                dispatch_ctx.update(target, new_key, state)
        return te.create_schedule([x.op for x in outs])
    else:
        raise ValueError("Invalid tracing mode: " + env.tracing_mode)

    return schedule


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
