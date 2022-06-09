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
# pylint: disable=unused-variable,invalid-name, not-context-manager
"""
Decorator and utilities for the integration with TOPI and Relay
99.9% copy-paste of implementation by @MerryMercy

"""
import threading
import logging

import tvm
from tvm.autotvm.task.dispatcher import DispatchContext, FallbackContext
from tvm.target import Target
from .task import create
from .topi_integration import TaskExtractEnv

logger = logging.getLogger("autotvm")


# TODO(moreau89) find a more elegant way to lower for VTAs
def _lower(mod, target, params, opt_level=3):
    """Helper to lower VTA properly."""
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm.relay.backend import graph_executor_codegen

    if hasattr(target, "device_name") and target.device_name == "vta":
        import vta

        with vta.build_config(opt_level=opt_level, disabled_pass={"AlterOpLayout"}):
            mod, _ = relay.optimize(mod, target=target, params=params)
            grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
            grc.codegen(mod, mod["main"])
            return

    # Alter op layout code has been written expecting that tuning is applied
    # without it, so we disable AlterOpLayout to maintain that behavior.
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass={"AlterOpLayout"}):
        compiler = relay.vm.VMCompiler()
        if params:
            compiler.set_params(params)
        compiler.lower(mod, target=target)


def extract_from_program(mod, params, target, target_host=None, ops=None):
    """Extract tuning tasks from a relay program.

    This function is the single program version of extract_from_multiple_program.

    Parameters
    ----------
    mod: tvm.IRModule or relay.function.Function
        The module or function to tune
    params: dict of str to numpy array
        The associated parameters of the program
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target
    ops: List[tvm.ir.Op] or None
        List of relay ops to be tuned. If not specified, all tunable ops will be extracted.

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    target, target_host = Target.canon_target_and_host(target, target_host)
    return extract_from_multiple_program([mod], [params], target, ops=ops)


def extract_from_multiple_program(mods, params, target, target_host=None, ops=None):
    """Extract tuning tasks from multiple relay programs.

    This function collects tuning tasks by building a list of programs
    with a "tracing" target and tracing all the calls to topi.

    Parameters
    ----------
    mods: List[tvm.IRModule] or List[relay.function.Function]
        The list of modules or functions to tune
    params: List of dict of str to numpy array
        The associated parameters of the programs
    target: tvm.target.Target
        The compilation target
    target_host: tvm.target.Target
        The host compilation target
    ops: List[tvm.ir.Op] or None
        List of relay ops to be tuned.  If not specified, all tunable ops will be extracted.

    Returns
    -------
    task: Array of autotvm.task.Task
        collected tasks
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm import topi

    env = TaskExtractEnv.get()

    # merge target and target host
    target, target_host = Target.canon_target_and_host(target, target_host)

    # run compiler to collect all TOPI calls during compilation
    env.reset(ops)
    with env:
        # disable logger temporarily
        old_state = logger.disabled
        logger.disabled = True

        for mod, param in zip(mods, params):
            if isinstance(mod, relay.function.Function):
                mod = tvm.IRModule.from_expr(mod)
            assert isinstance(
                mod, tvm.IRModule
            ), "only support relay Module or Function to be tuned"
            relay.backend.te_compiler.get().clear()
            # wrap build call in thread to avoid multiprocessing problems
            build_thread = threading.Thread(target=_lower, args=(mod, target, param))
            build_thread.start()
            build_thread.join()
            relay.backend.te_compiler.get().clear()
            # Clear the warning message cache in FallbackContext
            if isinstance(DispatchContext.current, FallbackContext):
                DispatchContext.current.memory = {}
                DispatchContext.warning_messages = set()

        logger.disabled = old_state

    # create tasks for target
    tasks = []
    for task_name, args in env.get_tasks():
        try:
            tsk = create(task_name, args, target=target)
            tasks.append(tsk)
        except topi.InvalidShapeError:
            logger.warning("Invalid shape during AutoTVM task creation")

    return tasks
