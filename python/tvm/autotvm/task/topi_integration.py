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

from ... import _api_internal, tensor, placeholder, create_schedule

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

    def __init__(self):
        import topi

        # topi compute -> autotvm task name
        self.topi_to_task = {
            topi.nn.conv2d: "topi_nn_conv2d",
            topi.nn.depthwise_conv2d_nchw: "topi_nn_depthwise_conv2d_nchw",
            topi.nn.group_conv2d_nchw: "topi_nn_group_conv2d_nchw",
            topi.nn.conv2d_transpose_nchw: "topi_nn_conv2d_transpose_nchw",
            topi.nn.dense: "topi_nn_dense",
        }

        self.topi_to_schedule = {
            topi.nn.conv2d: [topi.generic.schedule_conv2d_nchw,
                             topi.generic.schedule_conv2d_nhwc],
            topi.nn.depthwise_conv2d_nchw: [topi.generic.schedule_depthwise_conv2d_nchw,
                                            topi.generic.schedule_depthwise_conv2d_nhwc],
            topi.nn.group_conv2d_nchw: [topi.generic.schedule_group_conv2d_nchw],
            topi.nn.conv2d_transpose_nchw: [topi.generic.schedule_conv2d_transpose_nchw],
            topi.nn.dense: [topi.generic.schedule_dense],
        }

        self._register_tracing()
        self._register_topi_task()
        self.task_collection = []
        self.wanted_topi_funcs = list(self.topi_to_task.keys())

    def _register_tracing(self):
        """Register tracing function to track the topi function call"""
        # register topi compute for "tracing" target
        for topi_compute in self.topi_to_task:
            def _local_scope(compute_func):
                """start a scope to hold the local function in for loop"""

                @compute_func.register("tracing", )
                def _tracing_topi_compute(*args, **kwargs):
                    assert not kwargs, "Do not support extracting tuning tasks when" \
                                       "kwargs is used in TOPI function call." \
                                       "Please modify it to use only positional args."

                    if compute_func in self.wanted_topi_funcs:  # record this call
                        key = (self.topi_to_task[compute_func], serialize_args(args))
                        if key not in self.task_collection:
                            self.task_collection.append(key)

                    return compute_func.fdefault(*args)
            _local_scope(topi_compute)

        # register topi schedule for "tracing" target
        for topi_compute in self.topi_to_task:
            for topi_schedule in self.topi_to_schedule[topi_compute]:
                def _local_scope_(schedule_func):
                    """start a scope to hold the local function in for loop"""

                    @schedule_func.register("tracing", )
                    def _tracing_topi_compute(outs):
                        outs = [outs] if isinstance(outs, tensor.Tensor) else outs
                        return create_schedule([x.op for x in outs])
                _local_scope_(topi_schedule)

    def _register_topi_task(self):
        """register tuning wrapper for topi function"""
        import topi

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
            data, weight, bias = args
            C = topi.nn.dense(*args, **kwargs)
            s = topi.generic.schedule_dense([C])
            if bias is not None:
                return s, [data, weight, bias, C]
            return s, [data, weight, C]

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
    def get():
        """Get the single instance of TaskExtractEnv

        Returns
        -------
        env: TaskExtractEnv
            The single instance of TaskExtractEnv
        """
        if not TaskExtractEnv.current:
            TaskExtractEnv.current = TaskExtractEnv()
        return TaskExtractEnv.current


def register_topi_compute(topi_compute, target_keys, template_keys, func=None):
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

            @config_dispatcher.register(template_keys)
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


def register_topi_schedule(topi_schedule, target_keys, template_keys, func=None):
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

            @config_dispatcher.register(template_keys)
            def template_call(cfg, outs, *args, **kwargs):
                """call the schedule func"""
                if f == topi_schedule.fdefault:
                    return f(outs, *args, **kwargs)
                return f(cfg, outs, *args, **kwargs)

        return f

    if func:
        _decorator(func)

    return _decorator
