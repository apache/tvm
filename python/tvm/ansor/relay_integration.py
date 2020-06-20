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
Integrate ansor into relay. It implements the following items:
1. Extract search tasks from a relay program
2. Provide auto-scheduling for all TOPI compute functions
"""
import os
import json
import threading

from tvm import target, te, transform
from tvm.te.tensor import PlaceholderOp, ComputeOp
from .dispatcher import DispatchContext
from .workload_registry import register_auto_scheduler_workload_bufs, compute_dag_hash
from .compute_dag import ComputeDAG, LayoutRewriteLevel
from .env import GLOBAL_SCOPE

def call_all_topi_funcs(mod, target, params):
    """Call all TOPI compute + schedule to extract tasks in a relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    with transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        bld_mod = relay.build_module.BuildModule()
        bld_mod.call_all_topi_funcs(mod, target=target, params=params)

def extract_from_program(mod, params, target, target_host=None):
    """ Extract tuning tasks from a relay program.

    This function is the single program version of extract_from_multiple_program.

    Parameters
    ----------
    mod : relay.Module
        The module to extract.
    params: dict of str to numpy array
        The associated parameters of the program
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target

    Returns
    -------
    workloads: Array of Tuple(wkl_key, target)
    """
    return extract_from_multiple_program([mod], [params], target, target_host)

def extract_from_multiple_program(mods, params, target, target_host=None):
    """ Extract tuning tasks from multiple relay programs.

    Parameters
    ----------
    mods : List of relay.Module
        The modules to extract.
    params: List of dict of str to numpy array
        The associated parameters of the programs
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target

    Returns
    -------
    workloads: Array of Tuple(wkl_key, target)
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    env = TracingEnvironment(TracingMode.EXTRACT_TASK)
    with env:
        # run compiler to collect all TOPI calls during compilation
        for mod, param in zip(mods, params):
            # wrap build call in a new thread to avoid the conflict
            # between python's multiprocessing and tvm's thread pool
            build_thread = threading.Thread(target=call_all_topi_funcs,
                                            args=(mod, target, param))
        build_thread.start()
        build_thread.join()
        relay.backend.compile_engine.get().clear()

    # create tasks for target
    wkl_keys = []
    wkl_weights = []
    for wkl_key, wkl_weight in env.wkl_key_collection.items():
        wkl_keys.append(wkl_key)
        wkl_weights.append(wkl_weight)

    return wkl_keys, wkl_weights


def prepare_layout_rewrite(mod, params, target):
    """
    Prepare for kernel layout rewrite. This function will write layout infos to a global static variable.
    Then these layout info will be used by a relay pass `kernel_layout_transform`.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    env = TracingEnvironment(TracingMode.PREPARE_LAYOUT_REWRITE)
    with env:
        # wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(target=call_all_topi_funcs,
                                        args=(mod, target, params))
        build_thread.start()
        build_thread.join()
        relay.backend.compile_engine.get().clear()

    if env.layout_rewrite_success_ct > 0:
        GLOBAL_SCOPE.topi_in_compute_rewrite_mode = True

def finish_layout_rewrite():
    """Clear the global flag for layout rewrite"""
    GLOBAL_SCOPE.topi_in_compute_rewrite_mode = False


class TracingMode:
    """Two modes for tracing"""
    EXTRACT_TASK = 0            # trace all topi calls to extract tasks
    PREPARE_LAYOUT_REWRITE = 1  # trace all topi calls to prepare layout rewrite

class TracingEnvironment:
    """Global environment for tracing all topi function calls"""
    current = None

    def __init__(self, tracing_mode):
        self.tracing_mode = tracing_mode
        self.relay_disable_build_cache = "false"
        self.layout_rewrite_success_ct = 0
        self.wkl_key_collection = {}

    def __enter__(self):
        self.relay_disable_build_cache = os.environ.get("TVM_RELAY_DISABLE_BUILD_CACHE", "false")
        os.environ["TVM_RELAY_DISABLE_BUILD_CACHE"] = "true"
        TracingEnvironment.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ["TVM_RELAY_DISABLE_BUILD_CACHE"] = self.relay_disable_build_cache
        TracingEnvironment.current = None

    def add_workload_key(self, key):
        """Add the workload key of an Ansor search task

        Parameters
        ----------
        key: str
        """
        if key in self.wkl_key_collection:
            self.wkl_key_collection[key] += 1
        else:
            self.wkl_key_collection[key] = 1


def traverse_to_get_io_tensors(outs):
    """Traverse from a list of output tensors to get a whole computational DAG"""
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

    has_layout_free = (len(layout_free_ops) > 0)
    return inputs + [t for t in outs], has_layout_free


def auto_schedule_topi(outs):
    """ Use ansor to auto-schedule a topi compute declaration """
    io_tensors, has_layout_free = traverse_to_get_io_tensors(outs)
    key = register_auto_scheduler_workload_bufs(io_tensors)

    env = TracingEnvironment.current
    if env is None:  # in the final build mode
        state = DispatchContext.current.query(target.Target.current(), key)
        dag = ComputeDAG(io_tensors)
        # Only update compute body, layout_rewrite_level = LayoutRewriteLevel.COMPUTE_REWRITE,
        # Since kernel layout has already been rewritten in relay pass
        schedule, _ = dag.apply_steps_from_state(state,
             layout_rewrite_level=LayoutRewriteLevel.COMPUTE_REWRITE)
        return schedule
    elif env.tracing_mode == TracingMode.EXTRACT_TASK:  # in the task extraction mode
        env.add_workload_key(key)
        return te.create_schedule([x.op for x in outs])
    elif env.tracing_mode == TracingMode.PREPARE_LAYOUT_REWRITE:
        # in prepare_layout_rewrite mode
        if has_layout_free:
            # Rewrite the DAG and update the transform history for
            # the new dag in DispatchContext
            dispatch_ctx = DispatchContext.current
            tgt = target.Target.current()
            state = dispatch_ctx.query(tgt, key)
            assert state is not None
            dag = ComputeDAG(outs)
            new_dag = dag.rewrite_layout_from_state(state)
            new_key = json.dumps((compute_dag_hash(new_dag),))
            dispatch_ctx.update(tgt, new_key, state)
            if new_key != key:
                env.layout_rewrite_success_ct += 1
        return te.create_schedule([x.op for x in outs])
    else:
        raise ValueError("Invalid tracing mode: " + env.tracing_mode)
