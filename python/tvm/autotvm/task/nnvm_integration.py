# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and NNVM

"""
import warnings
import logging


from ... import target as _target

from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger('autotvm')


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
    import nnvm
    import topi

    env = TaskExtractEnv.get()

    #NOTE: To add more symbols, you only need to change the following lists
    #nnvm symbol -> topi compute
    SYMBOL2TOPI = {
        nnvm.sym.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw,
                          topi.nn.group_conv2d_nchw],
        nnvm.sym.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
        nnvm.sym.dense: [topi.nn.dense],
    }

    topi_funcs = []
    for sym_name in symbols:
        if sym_name in SYMBOL2TOPI:
            topi_funcs.extend(SYMBOL2TOPI[sym_name])
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
    import nnvm
    import topi

    env = TaskExtractEnv.get()

    #NOTE: To add more symbols, you only need to change the following lists
    #nnvm symbol -> topi compute
    SYMBOL2TOPI = {
        nnvm.sym.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw,
                          topi.nn.group_conv2d_nchw],
        nnvm.sym.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
        nnvm.sym.dense: [topi.nn.dense],
    }

    topi_funcs = []
    for sym_name in symbols:
        if sym_name in SYMBOL2TOPI:
            topi_funcs.extend(SYMBOL2TOPI[sym_name])
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
