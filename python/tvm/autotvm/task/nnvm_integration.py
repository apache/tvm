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
Decorator and utilities for the integration with TOPI and NNVM

"""
import threading
import warnings
import logging


from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger('autotvm')


def extract_from_graph(graph, shape, dtype, target, symbols, params=None, target_host=None):
    """ Extract tuning tasks from a nnvm graph.

    This function collects tuning tasks by building the graph
    and trace all the calls to topi.

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
    params : dict of str to NDArray
        The parameter dictionary.
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

    # NOTE: To add more symbols, you only need to change the following lists
    # nnvm symbol -> topi compute
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
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        nnvm.compiler.engine.clear_cache()
        # wrap build call in thread to avoid multiprocessing problems
        build_thread = threading.Thread(target=nnvm.compiler.build,
                                        args=(graph,
                                              target,
                                              shape,
                                              dtype,
                                              params,
                                              target_host))
        build_thread.start()
        build_thread.join()

        logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            tsk = create(task_name, args,
                         target=target, target_host=target_host,
                         template_key='direct')
            tasks.append(tsk)
        except topi.InvalidShapeError:
            print("[Warning] Invalid shape during AutoTVM task creation")

    return tasks


def extract_from_multiple_graph(graphs, shapes, dtypes, target, symbols, params, target_host=None):
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
    params : dict of str to NDArray
        The parameter dictionary.
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
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        for graph, shape, dtype in zip(graphs, shapes, dtypes):
            nnvm.compiler.engine.clear_cache()
            # wrap build call in thread to avoid multiprocessing problems
            build_thread = threading.Thread(target=nnvm.compiler.build,
                                            args=(graph,
                                                  target,
                                                  shape,
                                                  dtype,
                                                  params,
                                                  target_host))
            build_thread.start()
            build_thread.join()

        logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            tsk = create(task_name, args,
                         target=target, target_host=target_host,
                         template_key='direct')
            tasks.append(tsk)
        except topi.InvalidShapeError:
            print("[Warning] Invalid shape during AutoTVM task creation")

    return tasks
