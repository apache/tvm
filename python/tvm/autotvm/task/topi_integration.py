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
import functools

import tvm.te._ffi_api
from tvm.target import Target
from tvm.te import tensor
from tvm.autotvm.env import GLOBAL_SCOPE
from tvm import runtime

from .task import (
    args_to_workload,
    serialize_args,
    DispatchContext,
    _register_task_compute,
    _register_task_schedule,
    _register_subgraph_task_schedule,
    _get_compute,
    deserialize_args,
)


# Task extractor for relay program
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from graph"""

    current = None
    registered = None

    def __init__(self, allow_duplicate=False):
        self.allow_duplicate = allow_duplicate
        self.task_collection = []
        self.wanted_relay_ops = None
        self.modified_funcs = []
        self.tracing = False

    def __enter__(self):
        self.task_collection = []
        self.tracing = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracing = False

    def reset(self, wanted_relay_ops=None):
        """Reset task collections

        Parameters
        ----------
        wanted_relay_ops: List of tvm.ir.Op
            The relay ops to be extracted
        """
        self.task_collection = []
        self.wanted_relay_ops = wanted_relay_ops

    def add_task(self, task_name, args, subgraph_name=None):
        """Add AutoTVM task

        Parameters
        ----------
        task_name: str
            AutoTVM task name.
        subgraph_name: str
            Subgraph name

        args: tuple
            Arguments to the TOPI function.
        """
        key = (task_name, subgraph_name, serialize_args(args))
        if self.allow_duplicate or key not in self.task_collection:
            self.task_collection.append(key)

    def get_tasks(self):
        """Get collected tasks

        Returns
        -------
        tasks: List of tuple(name, args)
            A list of tasks extracted from the graph
        """
        return self.task_collection

    @staticmethod
    def get(allow_duplicate=False):
        """Get the single instance of TaskExtractEnv

        Parameters
        ----------
        allow_duplicate : boolean
            Whether to fetch all workloads in the network,
            even though some of them are the same. This is
            useful for graph tuning.

        Returns
        -------
        env: TaskExtractEnv
            The single instance of TaskExtractEnv
        """
        if not TaskExtractEnv.current:
            TaskExtractEnv.current = TaskExtractEnv(allow_duplicate)
        else:
            TaskExtractEnv.current.allow_duplicate = allow_duplicate
        return TaskExtractEnv.current


def register_topi_compute(task_name, func=None):
    """Register a tunable template for a topi compute function.

    The registration will wrap this topi compute to take `cfg` as the first argument,
    followed by the original argument list. It uses all its argument as workload and
    stores this "workload" to its final ComputeOp, which can be used to reconstruct
    "workload" in the following topi_schedule call.

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

    def _decorate(topi_compute):
        @functools.wraps(topi_compute)
        @_register_task_compute(task_name)
        def wrapper(*args, **kwargs):
            """wrapper function for topi compute"""
            assert not kwargs, "Do not support kwargs in template function call"
            task_env = TaskExtractEnv.current
            # adding task in build phase is for subgraph to find tuable op lists
            if task_env is not None and (task_env.tracing or GLOBAL_SCOPE.tune_subgraph):
                task_env.add_task(task_name, args)
            workload = args_to_workload(args, task_name)
            tgt = Target.current()
            cfg = DispatchContext.current.query(tgt, workload)
            node = topi_compute(cfg, *args)

            # attach workload to return op
            op = node.op
            attrs = {}
            for k, v in node.op.attrs.items():
                attrs[k] = v
            attrs["workload"] = workload
            if isinstance(op, tensor.ComputeOp):
                op = tvm.te._ffi_api.ComputeOp(op.name, op.tag, attrs, op.axis, op.body)
            elif isinstance(op, tensor.ExternOp):
                op = tvm.te._ffi_api.ExternOp(
                    op.name,
                    op.tag,
                    attrs,
                    op.inputs,
                    op.input_placeholders,
                    op.output_placeholders,
                    op.body,
                )
            else:
                raise RuntimeError("Unsupported op type: " + str(type(op)))

            if isinstance(node, tensor.Tensor):
                return op.output(0)
            return [op.output(i) for i in range(len(node))]

        return wrapper

    if func:
        return _decorate(func)
    return _decorate


def register_topi_schedule(task_name, func=None):
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
        @functools.wraps(topi_schedule)
        @_register_task_schedule(task_name)
        def wrapper(outs, *args, **kwargs):
            """wrapper function for topi schedule"""
            task_env = TaskExtractEnv.current
            if GLOBAL_SCOPE.tune_subgraph and task_env is not None and not task_env.tracing:
                # get workload of subgraph
                workload = get_workload_post_order(outs)
            else:
                workload = get_workload(outs, task_name)
            if workload is None:
                raise RuntimeError(
                    f"Cannot find TOPI workload {task_name}. "
                    "Is it registered with `register_topi_compute`?"
                )
            tgt = Target.current()
            cfg = DispatchContext.current.query(tgt, workload)
            if (
                GLOBAL_SCOPE.tune_subgraph
                and isinstance(cfg, tvm.autotvm.task.space.FallbackConfigEntity)
                and len(cfg) == 1
            ):
                # get fallback config for subgraph workload
                workload = get_workload(outs, task_name)
                cfg = DispatchContext.current.query(tgt, workload)
            return topi_schedule(cfg, outs, *args, **kwargs)

        return wrapper

    if func:
        return _decorate(func)
    return _decorate


def register_topi_subgraph(best_impl_name, args, subgraph_name, outputs):
    """Register a tunable template for a subgraph.

    The registration adds subgraph as a tunable task to autotvm tasks. It uses subgraph_name and
    args as workload and stores this "workload" to its final ComputeOp, which can be used to
    reconstruct "workload" in the following topi_schedule call. Then it wrap the outputs which
    attatch the "workload" as the subgraph's topi compute. At last it employs the schedule
    corresponding to the anchor implementation as the subgraph's schedule.

    Parameters
    ----------
    best_impl_name: str
        The anchor implementation name of the subgraph

    args: tuple
        Arguments to the subgraph.

    subgraph_name: str
        The name of the subgraph

    outputs: list[tensor]
        The outputs of the subgraph

    Returns
    -------
    outs: list[tensor]
        The subgraph's outputs with workload attached to the last op.

    """
    task_env = TaskExtractEnv.current
    if task_env is not None and task_env.tracing:
        task_env.add_task(best_impl_name, args, subgraph_name=subgraph_name)
    # attach subgraph workload to return op
    subgraph_workload = args_to_workload(args, subgraph_name)
    node = outputs[0]
    op = node.op
    attrs = {}
    for k, v in node.op.attrs.items():
        attrs[k] = v
    attrs["workload"] = subgraph_workload
    if isinstance(op, tensor.ComputeOp):
        op = tvm.te._ffi_api.ComputeOp(op.name, op.tag, attrs, op.axis, op.body)
    elif isinstance(op, tensor.ExternOp):
        op = tvm.te._ffi_api.ExternOp(
            op.name,
            op.tag,
            attrs,
            op.inputs,
            op.input_placeholders,
            op.output_placeholders,
            op.body,
        )
    else:
        raise RuntimeError("Unsupported op type: " + str(type(op)))
    if isinstance(node, tensor.Tensor):
        outs = [op.output(0)]
    else:
        outs = [op.output(i) for i in range(len(node))]

    # get workload of anchor op
    workload = get_workload(outs, best_impl_name)
    if workload is None:
        raise RuntimeError(
            f"Cannot find TOPI workload {best_impl_name}. "
            "Is it registered with `register_topi_compute`?"
        )
    ancor_op_args = get_args_from_workload(workload)

    if task_env is not None and task_env.tracing:

        @_register_task_compute(subgraph_name)
        def wrapper(*args, **kwargs):
            """return outputs of the subgraph"""
            # get anchor op's config space by running fcompute
            fcompute = _get_compute(best_impl_name)
            fcompute(*ancor_op_args, **kwargs)

            # update subgraph's config space by anchor op's
            tgt = Target.current()
            cfg = DispatchContext.current.query(tgt, workload)
            DispatchContext.current.update(tgt, subgraph_workload, cfg)
            return outs

        # best_impl_name is the schedule name
        _register_subgraph_task_schedule(subgraph_name, best_impl_name)
    return outs


def get_workload(outs, task_name=None):
    """Retrieve the workload from outputs"""

    def traverse(tensors):
        """traverse all ops to find attached workload"""
        for t in tensors:
            op = t.op
            wkl = traverse(op.input_tensors)
            if wkl is not None:
                return wkl

            if "workload" in op.attrs:
                ret = args_to_workload(op.attrs["workload"])
                if task_name is None or ret[0] == task_name:
                    return ret
        return None

    outs = [outs] if isinstance(outs, tensor.Tensor) else outs
    return traverse(outs)


def get_workload_post_order(outs):
    """Retrieve the workload from outputs by post order"""

    def traverse(t):
        op = t.op
        if "workload" in op.attrs:
            wkl = args_to_workload(op.attrs["workload"])
            return wkl

        for x in t.op.input_tensors:
            if isinstance(x.op, tensor.ComputeOp):
                wkl = traverse(x)
                if wkl:
                    return wkl
        return None

    outs = [outs] if isinstance(outs, tensor.Tensor) else outs
    for t in outs:
        return traverse(t)


def get_args_from_workload(wkl):
    if isinstance(wkl[0], runtime.container.String):
        wkl = wkl[1:]
    return deserialize_args(wkl)
