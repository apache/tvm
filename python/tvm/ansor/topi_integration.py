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
# pylint: disable=unused-variable,invalid-name,unused-argument
"""
Decorators for registering tunable templates to TOPI.

These decorators can make your simple implementation be able to use different configurations
for different workloads.
Here we directly use all arguments to the TOPI call as "workload", so make sure all the arguments
(except tvm.te.Tensor) in you calls are hashable. For tvm.te.Tensor,
we will serialize it to a hashable tuple.

See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
"""
import os
import json
import tvm.te._ffi_api
from tvm import target as _target
from tvm.te import tensor
from tvm.te.tensor import PlaceholderOp, ComputeOp

from .dispatcher import DispatchContext, BlockingEmptyContext
from .workload_registry import register_auto_scheduler_workload_bufs, \
    make_workload_key_bufs, compute_dag_hash
from .compute_dag import ComputeDAG

def traverse_to_get_io_tensors(outs):
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

# Task extractor for relay program
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from graph"""
    current = None
    registered = None

    def __init__(self, do_layout_rewrite=False):
        self.do_layout_rewrite = do_layout_rewrite
        self.wanted_relay_ops = None
        self.modified_funcs = []
        self.tracing = False
        self.relay_disable_build_cache_ = "false"
        self.layout_rewrite_success_ct = 0
        self.wkl_key_collection = {}

    def __enter__(self):
        self.tracing = True
        self.wkl_key_collection = {}
        self.relay_disable_build_cache_ = os.environ.get("TVM_RELAY_DISABLE_BUILD_CACHE", "false")
        os.environ["TVM_RELAY_DISABLE_BUILD_CACHE"] = "true"

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracing = False
        os.environ["TVM_RELAY_DISABLE_BUILD_CACHE"] = self.relay_disable_build_cache_

    def reset(self, wanted_relay_ops=None):
        """Reset task collections

        Parameters
        ----------
        wanted_relay_ops: List of tvm.ir.Op
            The relay ops to be extracted
        """
        self.wanted_relay_ops = wanted_relay_ops
        self.relay_disable_build_cache_ = "false"
        self.layout_rewrite_success_ct = 0
        self.wkl_key_collection = {}

    def add_task(self, key):
        """Add AutoTVM task

        Parameters
        ----------
        task_name: str
            AutoTVM task name.

        args: tuple
            Arguments to the TOPI function.
        """
        if key in self.wkl_key_collection:
            self.wkl_key_collection[key] += 1
        else:
            self.wkl_key_collection[key] = 1

    def get_tasks(self):
        """Get collected tasks

        Returns
        -------
        tasks: List of tuple(name, args)
            A list of tasks extracted from the graph
        """
        return self.wkl_key_collection

    def get_wkl_keys(self):
        """Get collected tasks

        Returns
        -------
        wkl_keys: List of autoschedule workload_key
        """
        return self.wkl_key_collection

    @staticmethod
    def get(do_layout_rewrite=False):
        """Get the single instance of TaskExtractEnv

        Parameters
        ----------

        Returns
        -------
        env: TaskExtractEnv
            The single instance of TaskExtractEnv
        """
        if not TaskExtractEnv.current:
            TaskExtractEnv.current = TaskExtractEnv(do_layout_rewrite)
        else:
            TaskExtractEnv.current.do_layout_rewrite = do_layout_rewrite
        return TaskExtractEnv.current

def register_topi_schedule(func=None):
    """Register a tunable template for a topi schedule function.

    The registration will wrap this topi schedule to take `cfg` as the first argument,
    followed by the original argument list.

    Note that this function will try to find "workload" from all the ComputeOp in the input.
    You can attach "workload" to your compute op by using :any:`register_topi_compute`.

    The task name has to be the same as that of the corresponding topi compute function.

    Parameters
    ----------
    task_name: str
        The AutoTVM task name

    func: None or callable
        If it is None, return a decorator.
        If is callable, decorate this function.

    Returns
    -------
    decorator: callable
        A decorator

    Examples
    --------
    See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
    """
    def _decorate(topi_schedule):
        def wrapper(outs, *args, **kwargs):
            io_tensors, has_layout_free = traverse_to_get_io_tensors(outs)
            key = register_auto_scheduler_workload_bufs(io_tensors)
            task_env = TaskExtractEnv.current
            if task_env is not None and task_env.tracing:
                if task_env.do_layout_rewrite and has_layout_free:
                    # Rewrite the dag and update the transform history for
                    # the new dag in DispatchContext
                    dispatch_ctx = DispatchContext.current
                    tgt = _target.Target.current()
                    state = dispatch_ctx.query(tgt, key)
                    dag = ComputeDAG(outs)
                    new_dag = dag.rewrite_layout_from_state(state)
                    new_key = json.dumps((compute_dag_hash(new_dag),))
                    dispatch_ctx.update(tgt, new_key, state)

                    if new_key != key:
                        task_env.layout_rewrite_success_ct += 1

                    # Call schedule_func under FallbackContext() to avoid layout rewrite
                    cfg = BlockingEmptyContext().query(tgt, key)
                    return topi_schedule(cfg, outs)

                task_env.add_task(key)

            """wrapper function for topi schedule"""
            tgt = _target.Target.current()
            cfg = DispatchContext.current.query(tgt, key)
            return topi_schedule(cfg, outs)
        return wrapper
    if func:
        return _decorate(func)
    return _decorate
