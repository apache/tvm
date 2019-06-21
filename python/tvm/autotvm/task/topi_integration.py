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

from ... import _api_internal, tensor, placeholder

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


# Task extractor for nnvm graph, relay program
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from nnvm graph"""
    current = None
    registered = None

    def __init__(self, allow_duplicate=False):
        import topi

        # topi compute -> autotvm task name
        self.topi_to_task = {
            topi.nn.conv2d: "topi_nn_conv2d",
            topi.nn.depthwise_conv2d_nchw: "topi_nn_depthwise_conv2d_nchw",
            topi.nn.group_conv2d_nchw: "topi_nn_group_conv2d_nchw",
            topi.nn.conv2d_transpose_nchw: "topi_nn_conv2d_transpose_nchw",
            topi.nn.conv2d_NCHWc: "topi_x86_conv2d_NCHWc",
            topi.nn.dense: "topi_nn_dense",
            topi.nn.bitserial_conv2d_nchw: "topi_nn_bitserial_conv2d_nchw",
            topi.nn.bitserial_conv2d_nhwc: "topi_nn_bitserial_conv2d_nhwc",
            topi.nn.bitserial_dense: "topi_nn_bitserial_dense",
            topi.nn.deformable_conv2d_nchw: "topi_nn_deformable_conv2d_nchw",
        }

        self.topi_to_schedule = {
            topi.nn.conv2d: [topi.generic.schedule_conv2d_nchw,
                             topi.generic.schedule_conv2d_nhwc],
            topi.nn.depthwise_conv2d_nchw: [topi.generic.schedule_depthwise_conv2d_nchw,
                                            topi.generic.schedule_depthwise_conv2d_nhwc],
            topi.nn.group_conv2d_nchw: [topi.generic.schedule_group_conv2d_nchw],
            topi.nn.conv2d_transpose_nchw: [topi.generic.schedule_conv2d_transpose_nchw],
            topi.nn.conv2d_NCHWc: [topi.generic.schedule_conv2d_NCHWc],
            topi.nn.dense: [topi.generic.schedule_dense],
            topi.nn.bitserial_conv2d_nchw: [topi.generic.schedule_bitserial_conv2d_nchw],
            topi.nn.bitserial_conv2d_nhwc: [topi.generic.schedule_bitserial_conv2d_nhwc],
            topi.nn.bitserial_dense: [topi.generic.schedule_bitserial_dense],
            topi.nn.deformable_conv2d_nchw: [topi.generic.schedule_deformable_conv2d_nchw],
        }

        # function reflection for tracing
        self.func_to_reflection = {
            topi.nn.conv2d:                 lambda x: setattr(topi.nn, 'conv2d', x),
            topi.nn.conv2d_NCHWc:           lambda x: setattr(topi.nn, 'conv2d_NCHWc', x),
            topi.nn.depthwise_conv2d_nchw:  lambda x: setattr(topi.nn, 'depthwise_conv2d_nchw', x),
            topi.nn.group_conv2d_nchw:      lambda x: setattr(topi.nn, 'group_conv2d_nchw', x),
            topi.nn.conv2d_transpose_nchw:  lambda x: setattr(topi.nn, 'conv2d_transpose_nchw', x),
            topi.nn.dense:                  lambda x: setattr(topi.nn, 'dense', x),
            topi.nn.bitserial_conv2d_nchw:  lambda x: setattr(topi.nn, 'bitserial_conv2d_nchw', x),
            topi.nn.bitserial_conv2d_nhwc:  lambda x: setattr(topi.nn, 'bitserial_conv2d_nhwc', x),
            topi.nn.bitserial_dense:        lambda x: setattr(topi.nn, 'bitserial_dense', x),
            topi.nn.deformable_conv2d_nchw: lambda x: setattr(topi.nn, 'deformable_conv2d_nchw', x),
        }

        self.allow_duplicate = allow_duplicate
        self._register_topi_task()
        self.task_collection = []
        self.wanted_topi_funcs = list(self.topi_to_task.keys())
        self.modified_funcs = []

    def __enter__(self):
        self.task_collection = []
        self.modified_funcs = []

        for topi_compute in self.wanted_topi_funcs:
            def _local_scope(compute_func):
                """start a scope to hold the local function in for loop"""

                def _tracing_wrapper(*args, **kwargs):
                    assert not kwargs, "Do not support extracting tuning tasks when " \
                                       "kwargs is used in TOPI function call. " \
                                       "Please modify it to use only positional args."
                    key = (self.topi_to_task[compute_func], serialize_args(args))
                    if self.allow_duplicate or key not in self.task_collection:
                        self.task_collection.append(key)

                    return compute_func(*args, **kwargs)

                self.func_to_reflection[compute_func](_tracing_wrapper)
                self.modified_funcs.append(compute_func)

            _local_scope(topi_compute)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # revert modification
        for func in self.modified_funcs:
            self.func_to_reflection[func](func)

    def _register_topi_task(self):
        """register tuning wrapper for topi function"""
        import topi

        # Avoid double registration for certain targets
        if TaskExtractEnv.registered:
            return
        TaskExtractEnv.registered = True

        # Tuning wrapper for topi functions
        @register("topi_nn_conv2d")
        def _topi_nn_conv2d(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            layout = args[-2]
            assert layout == 'NCHW', "only support NCHW currently"
            C = topi.nn.conv2d(*args, **kwargs)
            s = topi.generic.schedule_conv2d_nchw([C])
            return s, [A, W, C]

        @register("topi_nn_depthwise_conv2d_nchw")
        def _topi_nn_depthwise_conv2d_nchw(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.depthwise_conv2d_nchw(*args, **kwargs)
            s = topi.generic.schedule_depthwise_conv2d_nchw([C])
            return s, [A, W, C]

        @register("topi_nn_group_conv2d_nchw")
        def _topi_nn_group_conv2d_nchw(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.group_conv2d_nchw(*args, **kwargs)
            s = topi.generic.schedule_group_conv2d_nchw([C])
            return s, [A, W, C]

        @register("topi_nn_conv2d_transpose_nchw")
        def _topi_nn_conv2d_transpose_nchw(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.conv2d_transpose_nchw(*args, **kwargs)
            s = topi.generic.schedule_conv2d_transpose_nchw([C])
            return s, [A, W, C]

        @register("topi_nn_dense")
        def _topi_nn_dense(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            if len(args) > 2:
                data, weight, bias = args[:3]
            else:
                data, weight = args
                bias = None
            C = topi.nn.dense(*args, **kwargs)
            s = topi.generic.schedule_dense([C])
            if bias is not None:
                return s, [data, weight, bias, C]
            return s, [data, weight, C]

        @register("topi_nn_bitserial_conv2d_nhwc")
        def _topi_bitserial_conv2d_nhwc(*args, **kwargs):
            args = deserialize_args(args)
            C = topi.nn.bitserial_conv2d_nhwc(*args, **kwargs)
            s = topi.generic.nn.schedule_bitserial_conv2d_nhwc([C])
            A, W = args[:2]
            return s, [A, W, C]

        @register("topi_nn_bitserial_conv2d_nchw")
        def _topi_bitserial_conv2d_nchw(*args, **kwargs):
            args = deserialize_args(args)
            C = topi.nn.bitserial_conv2d_nchw(*args, **kwargs)
            s = topi.generic.nn.schedule_bitserial_conv2d_nchw([C])
            A, W = args[:2]
            return s, [A, W, C]

        @register("topi_nn_bitserial_dense")
        def _topi_nn_bitserial_dense(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.bitserial_dense(*args, **kwargs)
            s = topi.generic.schedule_bitserial_dense([C])
            return s, [A, W, C]

        @register("topi_nn_deformable_conv2d_nchw")
        def _topi_nn_deformable_conv2d_nchw(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, Offset, W = args[:3]
            C = topi.nn.deformable_conv2d_nchw(*args, **kwargs)
            s = topi.generic.schedule_deformable_conv2d_nchw([C])
            return s, [A, Offset, W, C]

        @register("topi_nn_conv2d_NCHWc")
        def _topi_nn_conv2d_NCHWc(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.conv2d_NCHWc(*args, **kwargs)
            s = topi.generic.schedule_conv2d_NCHWc([C])
            return s, [A, W, C]

    def reset(self, wanted_topi_funcs):
        """Reset task collections

        Parameters
        ----------
        wanted_topi_funcs: List of function
            The topi function to be extracted
        """
        self.task_collection = []
        self.wanted_topi_funcs = wanted_topi_funcs

    def get_tasks(self):
        """Get collected tasks

        Returns
        -------
        tasks: List of tuple(name, args)
            A list of tasks extracted from the nnvm graph
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


def register_topi_compute(topi_compute, target_keys, template_keys, func=None, override=False):
    """Register a tunable template for a topi compute function.

    After the registration, this topi compute will become a configuration dispatcher. It uses
    all its argument as workload and dispatches configurations according to the input workload.

    It also stores this "workload" to its final ComputeOp, which can be used to reconstruct
    "workload" in the following topi_schedule call.

    Parameters
    ----------
    topi_compute: GenericFunc
        The topi compute function that will be overloaded
    target_keys: str or list of str
        The compilation target. The same as the argument of GenericFunc.register.
    template_keys: str or list of str
        The template key.
        We might have several strategies for a single operator (e.g. direct, im2col, winograd).
        The template key is used to identity the algorithm strategy.
        Every operator must have a "direct" template, which is used by default.
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
                    op = _api_internal._ComputeOp(
                        op.name, op.tag, attrs, op.axis, op.body)
                elif isinstance(op, tensor.ExternOp):
                    op = _api_internal._ExternOp(
                        op.name, op.tag, attrs,
                        op.inputs, op.input_placeholders,
                        op.output_placeholders, op.body)
                else:
                    raise RuntimeError("Unsupported op type: " + str(type(op)))

                if isinstance(node, tensor.Tensor):
                    return op.output(0)
                return [op.output(i) for i in range(len(node))]

        return f

    if func:
        _decorator(func)

    return _decorator


def register_topi_schedule(topi_schedule, target_keys, template_keys, func=None, override=False):
    """Register a tunable template for a topi schedule function.

    After the registration. This topi schedule will become a configuration dispatcher. It dispatches
    configurations according to the input workload.

    Note that this function will try to find "workload" from all the ComputeOp in the input.
    You can attach "workload" to your compute op by using :any:`register_topi_compute`.

    Parameters
    ----------
    topi_schedule: GenericFunc
        The topi schedule function that will be overloaded
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key.
        We might have several strategies for a single operator (e.g. direct, im2col, winograd).
        The template key is used to identity the algorithm strategy.
        Every operator must have a "direct" template, which is used by default.
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
    def _decorator(f):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTERED_DISPATCHER:
                _REGISTERED_DISPATCHER[target_key] = {}
            if topi_schedule not in _REGISTERED_DISPATCHER[target_key]:
                @topi_schedule.register(target_key)
                @dispatcher
                def config_dispatcher(outs, *args, **kwargs):
                    """override topi call as a workload dispatcher"""
                    def traverse(tensors):
                        """traverse all ops to find attached workload"""
                        for t in tensors:
                            op = t.op
                            if 'workload' in op.attrs:
                                return op.attrs['workload']
                            wkl = traverse(op.input_tensors)
                            if wkl:
                                return wkl
                        return None

                    outs = [outs] if isinstance(outs, tensor.Tensor) else outs
                    workload = traverse(outs)

                    if workload is None:
                        raise RuntimeError("Cannot find workload in attribute of this schedule")

                    return args_to_workload(workload)

                _REGISTERED_DISPATCHER[target_key][topi_schedule] = config_dispatcher

            config_dispatcher = _REGISTERED_DISPATCHER[target_key][topi_schedule]

            @config_dispatcher.register(template_keys, override=override)
            def template_call(cfg, outs, *args, **kwargs):
                """call the schedule func"""
                if f == topi_schedule.fdefault:
                    return f(outs, *args, **kwargs)
                return f(cfg, outs, *args, **kwargs)

        return f

    if func:
        _decorator(func)

    return _decorator
