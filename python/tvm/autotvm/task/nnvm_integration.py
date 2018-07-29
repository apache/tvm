# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and NNVM
"""
import warnings

from ... import _api_internal, tensor, placeholder, target as _target

from ..util import get_const_tuple, get_func_name
from .task import args_to_workload, dispatcher, create, register

# Decorators for registering templates to topi
# a table that records all registered dispatcher for all targets
_REGISTED_DISPATHCER = {
}

def register_topi_compute(topi_compute, target_keys, template_keys):
    """Register a tunable template for a topi compute function

    Parameters
    ----------
    topi_compute: callable
        The overloaded topi compute call
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key

    Returns
    -------
    decorator: callable
        A decorator
    """
    fname = get_func_name(topi_compute)

    def _decorator(func=None):
        """If call this function without argument, then we will reuse the function body
        of original function"""
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTED_DISPATHCER:
                _REGISTED_DISPATHCER[target_key] = {}
            if topi_compute not in _REGISTED_DISPATHCER:
                @topi_compute.register(target_key)
                @dispatcher
                def config_dispatcher(*args, **kwargs):
                    """override topi call as a config dispatcher"""
                    assert not kwargs, "Do not support kwargs in template function call"
                    return (fname, ) + args_to_workload(args)
                _REGISTED_DISPATHCER[target_key][topi_compute] = config_dispatcher

            config_dispatcher = _REGISTED_DISPATHCER[target_key][topi_compute]

            @config_dispatcher.register(template_keys)
            def template_call(cfg, *args, **kwargs):
                """call the topi func and attach workload to compute node"""
                assert not kwargs, "Do not support kwargs in template function call"
                if func is None:
                    node = topi_compute.fdefault(*args, **kwargs)
                else:
                    node = func(cfg, *args, **kwargs)

                # attach workload to return op
                op = node.op
                attrs = {}
                for k, v in node.op.attrs.items():
                    attrs[k] = v
                attrs['workload'] = (fname, ) + args_to_workload(args)
                if isinstance(op, tensor.ComputeOp):
                    op = _api_internal._ComputeOp(
                        op.name, op.tag, attrs, op.axis, op.body)
                elif isinstance(op, tensor.ExternOp):
                    op = _api_internal._ExternOp(
                        op.name, op.tag, attrs,
                        op.inputs, op.input_placeholders,
                        op.output_placeholders, op.body)
                else:
                    raise RuntimeError("Unsupported op type: " + type(op))

                if isinstance(node, tensor.Tensor):
                    return op.output(0)
                return [op.output(i) for i in range(len(node))]

    return _decorator

def register_topi_schedule(topi_schedule, target_keys, template_keys):
    """Register a tunable template for a topi schedule function

    Parameters
    ----------
    topi_schedule: callable
        The overloaded topi schedule call
    target_keys: str or list of str
        The compilation target
    template_keys: str or list of str
        The template key

    Returns
    -------
    decorator: callable
        A decorator
    """
    def _decorator(func):
        targets = [target_keys] if isinstance(target_keys, str) else target_keys
        for target_key in targets:
            if target_key not in _REGISTED_DISPATHCER:
                _REGISTED_DISPATHCER[target_key] = {}
            if topi_schedule not in _REGISTED_DISPATHCER[target_key]:
                @topi_schedule.register(target_key)
                @dispatcher
                def config_dispatcher(outs):
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

                _REGISTED_DISPATHCER[target_key][topi_schedule] = config_dispatcher

            config_dispatcher = _REGISTED_DISPATHCER[target_key][topi_schedule]

            @config_dispatcher.register(template_keys)
            def template_call(cfg, outs):
                """call the schedule func"""
                return func(cfg, outs)

    return _decorator


def serialize_args(args):
    ret = []
    for t in args:
        if isinstance(t, tensor.Tensor):
            ret.append(('TENSOR', get_const_tuple(t.shape), t.dtype))
        else:
            ret.append(t)
    return tuple(ret)


def deserialize_args(args):
    ret = []
    for t in args:
        if isinstance(t, tuple) and t[0] == 'TENSOR':
            ret.append(placeholder(shape=t[1], dtype=t[2]))
        else:
            ret.append(t)
    return ret


# Task extractor for nnvm graph
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from nnvm graph"""
    current = None

    def __init__(self):
        import topi
        import nnvm

        self.symbol2topi = {
            nnvm.sym.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw]
        }

        self.topi_to_task = {
            topi.nn.conv2d: "topi_nn_conv2d",
            topi.nn.depthwise_conv2d_nchw: "topi_nn_depthwise_conv2d_nchw",
        }

        self._register_dummy()
        self._register_topi_task()
        self.task_collection = []

    def _register_dummy(self):
        """Register dummy function to track the topi function call"""
        for func in self.topi_to_task:
            def _local_scope(local_func):
                """build a scope to holds the function"""
                @local_func.register("dummy", )
                def _dummy_func(*args, **kwargs):
                    assert not kwargs, "Do not support extracting tuning tasks when" \
                                       "kwargs is used in TOPI function call." \
                                       "Please modify it to use only positional args."

                    if (self.topi_to_task[local_func], serialize_args(args)) \
                            not in self.task_collection:
                        self.task_collection.append((self.topi_to_task[local_func],
                                                     serialize_args(args)))
                    with _target.create("opencl"):
                        return local_func(*args)

            _local_scope(func)

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

    def reset(self):
        """Reset task collections"""
        self.task_collection = []

    def get_tasks(self):
        """Get collected tasks"""
        return self.task_collection

    @staticmethod
    def get():
        """Get the single instance of TaskExtractEnv"""
        if not TaskExtractEnv.current:
            TaskExtractEnv.current = TaskExtractEnv()
        return TaskExtractEnv.current

def extract_from_graph(graph, shape, dtype, target, symbols, target_host=None):
    """ Extract tuning tasks from a nnvm graph

    Parameters
    ----------
    graph : Graph
        The graph to tune
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    target: tvm.target.Target
        The compilation target
    symbols : Array of nnvm.symbol
        Array of nnvm symbols
    target_host: tvm.target.Target
        The host compilation target

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    import nnvm.compiler

    env = TaskExtractEnv.get()

    topi_funcs = []
    for sym_name in symbols:
        if sym_name in env.symbol2topi:
            topi_funcs.extend(env.symbol2topi[sym_name])
        else:
            warnings.warn("Symbol %s is not tunable, ignored" % sym_name)

    # run compiler to collect all TOPI calls during compilation
    env.reset()
    dummy_target = _target.create("opencl -device=dummy")
    nnvm.compiler.build(graph, target=dummy_target, shape=shape, dtype=dtype)

    tasks = []
    for task_name, args in env.get_tasks():
        tasks.append(create(task_name, args,
                            target=target, target_host=target_host,
                            template_key='vanilla'))

    return tasks
