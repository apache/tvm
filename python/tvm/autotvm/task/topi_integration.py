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
(except tvm.Tensor) in you calls are hashable. For tvm.Tensor, we will serialize it to a hashable
tuple.

See tvm/topi/python/topi/arm_cpu/depthwise_conv2d.py for example usage.
"""
<<<<<<< HEAD
import tvm.te._ffi_api

from ... import tensor, placeholder

from .task import args_to_workload, dispatcher, register
from ..util import get_const_tuple

# A table that records all registered dispatcher for all targets
_REGISTERED_DISPATCHER = {
}


def serialize_args(args):
    """serialize arguments of a topi function to a hashable tuple.

    Parameters
    ----------
    args: list of hashable or Tensor
    """
    ret = []
    for t in args:
        if isinstance(t, tensor.Tensor):
            ret.append(('TENSOR', get_const_tuple(t.shape), t.dtype))
        else:
            ret.append(t)
    return tuple(ret)


def deserialize_args(args):
    """The inverse function of :code:`serialize_args`.

    Parameters
    ----------
    args: list of hashable or Tensor
    """
    ret = []
    for t in args:
        if isinstance(t, tuple) and t[0] == 'TENSOR':
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret
=======
from tvm import target as _target

from ... import _api_internal, tensor
from .task import args_to_workload, DispatchContext, \
    register_task_compute, register_task_schedule, serialize_args
>>>>>>> relay op strategy


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
        wanted_relay_ops: List of relay.op.Op
            The relay ops to be extracted
        """
        self.task_collection = []
        self.wanted_relay_ops = wanted_relay_ops

    def add_task(self, task_name, args):
        """Add AutoTVM task

        Parameters
        ----------
        task_name: str
            AutoTVM task name.

        args: tuple
            Arguments to the TOPI function.

        cond: SpecializedCondition
            Specialized condition to enable the TOPI template.
        """
        key = (task_name, serialize_args(args))
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

    After the registration, this topi compute will become a configuration dispatcher. It uses
    all its argument as workload and dispatches configurations according to the input workload.

    It also stores this "workload" to its final ComputeOp, which can be used to reconstruct
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
<<<<<<< HEAD
    def _decorator(f):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTERED_DISPATCHER:
                _REGISTERED_DISPATCHER[target_key] = {}
            if topi_compute not in _REGISTERED_DISPATCHER[target_key]:
                @topi_compute.register(target_key)
                @dispatcher
                def config_dispatcher(*args, **kwargs):
                    """override topi call as a config dispatcher"""
                    assert not kwargs, "Do not support kwargs in template function call"
                    return args_to_workload(args, topi_compute)
                _REGISTERED_DISPATCHER[target_key][topi_compute] = config_dispatcher

            config_dispatcher = _REGISTERED_DISPATCHER[target_key][topi_compute]

            @config_dispatcher.register(template_keys, override=override)
            def template_call(cfg, *args, **kwargs):
                """call the topi func and attach workload to compute node"""
                assert not kwargs, "Do not support kwargs in template function call"

                if f == topi_compute.fdefault:
                    node = f(*args, **kwargs)
                else:
                    node = f(cfg, *args, **kwargs)

                # attach workload to return op
                op = node.op
                attrs = {}
                for k, v in node.op.attrs.items():
                    attrs[k] = v
                attrs['workload'] = args_to_workload(args, topi_compute)
                if isinstance(op, tensor.ComputeOp):
                    op = tvm.te._ffi_api.ComputeOp(
                        op.name, op.tag, attrs, op.axis, op.body)
                elif isinstance(op, tensor.ExternOp):
                    op = tvm.te._ffi_api.ExternOp(
                        op.name, op.tag, attrs,
                        op.inputs, op.input_placeholders,
                        op.output_placeholders, op.body)
                else:
                    raise RuntimeError("Unsupported op type: " + str(type(op)))

                if isinstance(node, tensor.Tensor):
                    return op.output(0)
                return [op.output(i) for i in range(len(node))]

        return f
=======
    def _decorate(topi_compute):
        @register_task_compute(task_name)
        def wrapper(*args, **kwargs):
            """wrapper function for topi compute"""
            assert not kwargs, "Do not support kwargs in template function call"
            task_env = TaskExtractEnv.current
            if task_env is not None and task_env.tracing:
                task_env.add_task(task_name, args)
            workload = args_to_workload(args, task_name)
            tgt = _target.current_target()
            cfg = DispatchContext.current.query(tgt, workload)
            node = topi_compute(cfg, *args)

            # attach workload to return op
            op = node.op
            attrs = {}
            for k, v in node.op.attrs.items():
                attrs[k] = v
            attrs['workload'] = workload
            if isinstance(op, tensor.ComputeOp):
                op = _api_internal._ComputeOp(
                    op.name, op.tag, attrs, op.axis, op.body)
            elif isinstance(op, tensor.ExternOp):
                op = _api_internal._ExternOp(
                    op.name, op.tag, attrs,
                    op.inputs, op.input_placeholders,
                    op.output_placeholders, op.body)
            else:
                raise RuntimeError("Unsupported op type: " + str(type(op)))
>>>>>>> relay op strategy

            if isinstance(node, tensor.Tensor):
                return op.output(0)
            return [op.output(i) for i in range(len(node))]

        return wrapper

    if func:
        return _decorate(func)
    return _decorate


def register_topi_schedule(task_name, func=None):
    """Register a tunable template for a topi schedule function.

    After the registration. This topi schedule will become a configuration dispatcher. It dispatches
    configurations according to the input workload.

    Note that this function will try to find "workload" from all the ComputeOp in the input.
    You can attach "workload" to your compute op by using :any:`register_topi_compute`.

    The task name need to match with the task name of the corresponding topi compute function.

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
        @register_task_schedule(task_name)
        def wrapper(outs, *args, **kwargs):
            """wrapper function for topi schedule"""
            workload = get_workload(outs)
            if workload is None:
                raise RuntimeError("Cannot find workload in attribute of this schedule")
            tgt = _target.current_target()
            cfg = DispatchContext.current.query(tgt, workload)
            return topi_schedule(cfg, outs, *args, **kwargs)
        return wrapper
    if func:
        return _decorate(func)
    return _decorate


def get_workload(outs):
    """Retrieve the workload from outputs"""
    def traverse(tensors):
        """traverse all ops to find attached workload"""
        for t in tensors:
            op = t.op
            if 'workload' in op.attrs:
                return args_to_workload(op.attrs['workload'])
            wkl = traverse(op.input_tensors)
            if wkl:
                return wkl
        return None
    outs = [outs] if isinstance(outs, tensor.Tensor) else outs
    return traverse(outs)
