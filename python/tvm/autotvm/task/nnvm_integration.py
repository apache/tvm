# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and NNVM

"""
import warnings
import logging


from ... import tensor, placeholder, create_schedule, target as _target

from ..util import get_const_tuple
from .task import create, register

logger = logging.getLogger('autotvm')

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


# Task extractor for nnvm graph
class TaskExtractEnv:
    """Global environment for extracting tuning tasks from nnvm graph"""
    current = None

    def __init__(self):
        import topi
        import nnvm

        # NOTE: To add more symbols, you only need to change the following lists
        # nnvm symbol -> topi compute
        self.symbol2topi = {
            nnvm.sym.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw],
            nnvm.sym.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
            nnvm.sym.dense: [topi.nn.dense],
        }

        # topi compute -> autotvm task name
        self.topi_to_task = {
            topi.nn.conv2d: "topi_nn_conv2d",
            topi.nn.depthwise_conv2d_nchw: "topi_nn_depthwise_conv2d_nchw",
            topi.nn.conv2d_transpose_nchw: "topi_nn_conv2d_transpose_nchw",
            topi.nn.dense: "topi_nn_dense",
        }

        self.topi_to_schedule = {
            topi.nn.conv2d: [topi.generic.schedule_conv2d_nchw,
                             topi.generic.schedule_conv2d_nhwc],
            topi.nn.depthwise_conv2d_nchw: [topi.generic.schedule_depthwise_conv2d_nchw,
                                            topi.generic.schedule_depthwise_conv2d_nhwc],
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


def extract_from_graph(graph, shape, dtype, target, symbols, target_host=None):
    """ Extract tuning tasks from a nnvm graph.

    This function collects tuning tasks by building the graph
    with a "tracing" target and tracing all the calls to topi.

    Parameters
    ----------
    graph : Graph
        The graph to tune
    shape : dict of str to tuple
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    target: tvm.target.Target
        The compilation target
    symbols : Array of nnvm.symbol
        Array of nnvm symbols want to be tuned
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
    env.reset(topi_funcs)

    # disable logger temporarily
    old_state = logger.disabled
    logger.disabled = True

    # use a "tracing" target to do a fake compile for collecting topi calls
    tracing_target = _target.create("llvm -device=tracing")
    nnvm.compiler.engine.clear_cache()
    nnvm.compiler.build(graph, target=tracing_target, shape=shape, dtype=dtype)

    logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        tasks.append(create(task_name, args,
                            target=target, target_host=target_host,
                            template_key='direct'))

    return tasks


def extract_from_multiple_graph(graphs, shapes, dtypes, target, symbols, target_host=None):
    """ Extract tuning tasks from multiple nnvm graphs.

    This function is the multiple graph version of extract_from_graph

    Parameters
    ----------
    graphs : List of Graph
        The list of graphs to tune
    shapes : List of dict of str to tuple
        The input shape to the graph
    dtypes : List of str or dict of str to str
        The input types to the graph
    target: tvm.target.Target
        The compilation target
    symbols : Array of nnvm.symbol
        Array of nnvm symbols want to be tuned
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
    env.reset(topi_funcs)

    # disable logger temporarily
    old_state = logger.disabled
    logger.disabled = True

    # use a "tracing" target to do a fake compile for collecting topi calls
    tracing_target = _target.create("llvm -device=tracing")

    nnvm.compiler.engine.clear_cache()
    for graph, shape, dtype in zip(graphs, shapes, dtypes):
        nnvm.compiler.build(graph, target=tracing_target, shape=shape, dtype=dtype)

    logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        tasks.append(create(task_name, args,
                            target=target, target_host=target_host,
                            template_key='direct'))

    return tasks
