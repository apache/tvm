# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and NNVM

"""
import warnings

from ... import tensor, placeholder, target as _target

from ..util import get_const_tuple
from .task import create, register


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

        self.symbol2topi = {
            nnvm.sym.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw],
            nnvm.sym.conv2d_transpose: [topi.nn.conv2d_transpose],
        }

        self.topi_to_task = {
            topi.nn.conv2d: "topi_nn_conv2d",
            topi.nn.depthwise_conv2d_nchw: "topi_nn_depthwise_conv2d_nchw",
            topi.nn.conv2d_transpose_nchw: "topi_nn_conv2d_transpose_nchw",
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

        @register("topi_nn_conv2d_transpose_nchw")
        def _topi_nn_conv2d_transpose_nchw(*args, **kwargs):
            assert not kwargs, "Do not support kwargs in template function call"
            args = deserialize_args(args)
            A, W = args[:2]
            C = topi.nn.conv2d_transpose_nchw(*args, **kwargs)
            s = topi.generic.schedule_conv2d_transpose_nchw([C])
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
    """ Extract tuning tasks from a nnvm graph.

    This function collects tunning tasks by building the graph
    with a "dummy" target and tracing all the calls to topi.

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
                            template_key='direct'))

    return tasks
