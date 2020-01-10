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
Decorator and utilities for the integration with TOPI and Relay
99.9% copy-paste of implementation by @MerryMercy

"""
import threading
import warnings
import logging


from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger('autotvm')


# TODO(moreau89) find a more elegant way to lower for VTAs
def _lower(func,
           target,
           params):
    """ Helper to lower VTA properly.
    """

    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen

    if hasattr(target, 'device_name') and target.device_name == "vta":
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            import vta
            with vta.build_config():
                mod, _ = relay.optimize(func, target, params)
                grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
                return grc.codegen(mod["main"])
    # default case
    mod, _ = relay.optimize(func, target, params)
    grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
    return grc.codegen(mod["main"])


def extract_from_program(func, params, ops, target, target_host=None,
                         template_keys=None):
    """ Extract tuning tasks from a relay program.

    This function is the single program version of extract_from_multiple_program.

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
    template_keys: dict of topi op to str
        The tuning template keys map for schedules, default to None.
        Example: {topi.nn.conv2d: 'direct'}

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    return extract_from_multiple_program([func], [params], ops, target, target_host,
                                         template_keys=template_keys)


def extract_from_multiple_program(funcs, params, ops, target, target_host=None,
                                  template_keys=None):
    """ Extract tuning tasks from multiple relay programs.

    This function collects tuning tasks by building a list of programs
    with a "tracing" target and tracing all the calls to topi.

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
    template_keys: dict of topi op to str
        The tuning template keys map for schedules, default to None.
        Example: {topi.nn.conv2d: 'direct'}

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    import tvm.relay.op
    from tvm import relay
    import topi

    env = TaskExtractEnv.get()

    # NOTE: To add more ops, you only need to change the following lists
    # relay op -> topi compute
    OP2TOPI = {
        tvm.relay.op.nn.conv2d: [topi.nn.conv2d, topi.nn.depthwise_conv2d_nchw,
                                 topi.nn.group_conv2d_nchw, topi.nn.conv2d_NCHWc],
        tvm.relay.op.nn.conv2d_transpose: [topi.nn.conv2d_transpose_nchw],
        tvm.relay.op.nn.dense: [topi.nn.dense],
        tvm.relay.op.nn.batch_matmul: [topi.nn.batch_matmul],
        tvm.relay.op.nn.deformable_conv2d: [topi.nn.deformable_conv2d_nchw],
    }

    topi_funcs = []
    for op_name in ops:
        if op_name in OP2TOPI:
            topi_funcs.extend(OP2TOPI[op_name])
        else:
            warnings.warn("Op %s is not tunable, ignored" % op_name)

    # run compiler to collect all TOPI calls during compilation
    env.reset(topi_funcs)
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        for func, param in zip(funcs, params):
            relay.backend.compile_engine.get().clear()
            # wrap build call in thread to avoid multiprocessing problems
            mod = relay.Module.from_expr(func)
            build_thread = threading.Thread(target=_lower,
                                            args=(mod, target, param))
            build_thread.start()
            build_thread.join()

        logger.disabled = old_state

    # convert *topi op to template key* map to *task name to template key* map
    task_name_to_keys = {}
    if template_keys is not None:
        for op in template_keys.keys():
            if op in env.topi_to_task:
                task_name_to_keys[env.topi_to_task[op]] = template_keys[op]
            else:
                logger.warning("Invalid template key, fallback to direct")
                task_name_to_keys[env.topi_to_task[op]] = 'direct'

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            key = task_name_to_keys[task_name] if task_name in task_name_to_keys else 'direct'
            tsk = create(task_name, args,
                         target=target, target_host=target_host,
                         template_key=key)
            tasks.append(tsk)
        except topi.InvalidShapeError:
            logger.warning("Invalid shape during AutoTVM task creation")

    return tasks
