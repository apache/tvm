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

import threading

import tvm
from tvm import te, transform
from tvm.te.tensor import ComputeOp, PlaceholderOp
from .compute_dag import ComputeDAG
from .dispatcher import DispatchContext
from .search_task import SearchTask
from .workload_registry import register_workload_tensors


def call_all_topi_funcs(mod, params, target):
    """Call all TOPI compute + schedule to extract tasks in a relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen

    with transform.PassContext(opt_level=3):
        opt_mod, _ = relay.optimize(mod, target, params)
        grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
        grc.codegen(opt_mod["main"])


def extract_tasks(mod, params, target, target_host=None, hardware_params=None):
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
    env = TracingEnvironment(TracingMode.EXTRACT_TASK)
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
    PREPARE_LAYOUT_REWRITE = 1  # trace topi calls to prepare layout rewrite


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


# The suffix of implementations that use the auto-scheduler in the OpStrategy.
auto_schedule_impl_suffix = ".auto_scheduler"


def auto_schedule_topi(outs):
    """Use auto-scheduler to schedule any topi compute function.

    Note: This is used internally for relay integration. Do
    not use this as a general user-facing API.

    Parameters
    ----------
    outs: List[Tensor]
        The output tensors of topi compute functions

    Returns
    -------
    sch: te.Schedule
        A topi schedule function
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    io_tensors, has_layout_free = traverse_to_get_io_tensors(outs)
    key = register_workload_tensors(io_tensors)

    # only enable layout rewrite for cpu backend
    enable_layout_rewrite = "cpu" in tvm.target.Target.current().keys

    env = TracingEnvironment.current
    if env is None:  # in the final build mode
        state = DispatchContext.current.query(tvm.target.Target.current(), key)
        if state is None:
            if "gpu" in tvm.target.Target.current().keys:
                raise RuntimeError("Cannot compile for GPU targets if no valid schedule is found.")
            return te.create_schedule([x.op for x in outs])

        dag = ComputeDAG(io_tensors)
        schedule, _ = dag.apply_steps_from_state(state)
    elif env.tracing_mode == TracingMode.EXTRACT_TASK:  # in the task extraction mode
        engine = relay.backend.compile_engine.get()
        ccache_key = engine.get_current_ccache_key()
        env.add_workload_key(key, ccache_key)
        schedule = te.create_schedule([x.op for x in outs])
    elif env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
        # todo(merrymercy, minminsun): port layout rewrite
        raise NotImplementedError
    else:
        raise ValueError("Invalid tracing mode: " + env.tracing_mode)

    return schedule
