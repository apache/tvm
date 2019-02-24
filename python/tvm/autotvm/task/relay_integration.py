# pylint: disable=unused-variable,invalid-name
"""
Decorator and utilities for the integration with TOPI and Relay
99.9% copy-paste of implementation by @MerryMercy

"""
import threading
import warnings
import logging


from ... import target as _target

from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger('autotvm')


def extract_from_program(func, params, ops, target, target_host=None):
    """ Extract tuning tasks from a relay program.

    This function collects tuning tasks by building the program
    with a "tracing" target and tracing all the calls to topi.

    Parameters
    ----------
    func: relay.expr.Function
        The func to tune
    params: dict of str to numpy array
        The associated parameters of the program
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    env = TaskExtractEnv.get()
    import tvm.relay.op
    from tvm import relay
    import topi

    # NOTE: To add more ops, you only need to change the following lists
    # relay op -> topi compute
    OP2TOPI = {
        tvm.relay.op.nn.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw,
                                 topi.nn.group_conv2d_nchw],
        tvm.relay.op.nn.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
        tvm.relay.op.nn.dense: [topi.nn.dense],
    }

    topi_funcs = []
    for op_name in ops:
        if op_name in OP2TOPI:
            topi_funcs.extend(OP2TOPI[op_name])
        else:
            warnings.warn("Op %s is not tunable, ignored" % op_name)

    # run compiler to collect all TOPI calls during compilation
    env.reset(topi_funcs)

    # disable logger temporarily
    old_state = logger.disabled
    logger.disabled = True

    # use a "tracing" target to do a fake compile for collecting topi calls
    tracing_target = _target.create("llvm -device=tracing")
    relay.backend.compile_engine.get().clear()
    # wrap build call in thread to avoid multiprocessing problems
    build_thread = threading.Thread(target=relay.build, args=(func,
                                                              tracing_target,
                                                              target_host,
                                                              params))
    build_thread.start()
    build_thread.join()
    logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        tasks.append(create(task_name, args,
                            target=target, target_host=target_host,
                            template_key='direct'))

    return tasks


def extract_from_multiple_program(funcs, params, ops, target, target_host=None):
    """ Extract tuning tasks from multiple relay programs.

    This function is the multiple program version of extract_from_program

    Parameters
    ----------
    funcs: List of relay.expr.Function
        The list of functions to tune
    params: List of dict of str to numpy array
        The associated parameters of the programs
    ops: List of relay op
        List of relay ops to be tuned
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    env = TaskExtractEnv.get()
    import tvm.relay.op
    from tvm import relay
    import topi

    # NOTE: To add more ops, you only need to change the following lists
    # relay op -> topi compute
    OP2TOPI = {
        tvm.relay.op.nn.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw,
                                 topi.nn.group_conv2d_nchw],
        tvm.relay.op.nn.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
        tvm.relay.op.nn.dense: [topi.nn.dense],
    }

    topi_funcs = []
    for op_name in ops:
        if op_name in OP2TOPI:
            topi_funcs.extend(OP2TOPI[op_name])
        else:
            warnings.warn("Op %s is not tunable, ignored" % op_name)

    # run compiler to collect all TOPI calls during compilation
    env.reset(topi_funcs)

    # disable logger temporarily
    old_state = logger.disabled
    logger.disabled = True

    # use a "tracing" target to do a fake compile for collecting topi calls
    tracing_target = _target.create("llvm -device=tracing")

    for func, param in zip(funcs, params):
        # wrap build call in thread to avoid multiprocessing problems
        build_thread = threading.Thread(target=relay.build, args=(func,
                                                                  tracing_target,
                                                                  target_host,
                                                                  params))
        build_thread.start()
        build_thread.join()

    logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        tasks.append(create(task_name, args,
                            target=target, target_host=target_host,
                            template_key='direct'))

    return tasks
