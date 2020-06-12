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
import tvm


from .topi_integration import TaskExtractEnv
from .dispatcher import BlockingEmptyContext
from .env import GLOBAL_SCOPE

def _lower(mod,
           target,
           params):
    """ Helper to lower VTA properly.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen

    if hasattr(target, 'device_name') and target.device_name == "vta":
        import vta
        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            mod, _ = relay.optimize(mod, target, params)
            grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
            grc.codegen(mod["main"])
            return

    # default case
    # Try graph codegen first to extract autotvm tasks.
    # If failed to compile, then fallback to use VM compiler.
    # TODO: Currently VM compiler is likely to stack overflow for large models.
    try:
      with relay.build_config(opt_level=3):
        opt_mod, _ = relay.optimize(mod, target, params)
        grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
        grc.codegen(opt_mod["main"])
    except tvm.TVMError:
        compiler = relay.vm.VMCompiler()
        if params:
            compiler.set_params(params)
        compiler.lower(mod, target=target)

OP_TO_SCHEDULE = {}

def init_op_to_schedule_map():
    # init the global map OP_TO_SCHEDULE inside a function, this is used to resolve import issues
    global OP_TO_SCHEDULE
    from tvm import relay
    import topi

    if OP_TO_SCHEDULE:
        return

    OP_TO_SCHEDULE = {
        relay.op.nn.conv2d: [topi.generic.schedule_conv2d_nchw,
                             topi.generic.schedule_conv2d_nhwc,
                             topi.generic.schedule_depthwise_conv2d_nchw,
                             topi.generic.schedule_depthwise_conv2d_nhwc,
                             topi.generic.schedule_group_conv2d_nchw,
                             topi.generic.schedule_conv2d_winograd_without_weight_transform],
        relay.op.nn.conv2d_transpose: [topi.generic.schedule_conv2d_transpose_nchw],
        relay.op.nn.dense: [topi.generic.schedule_dense],
        relay.op.nn.softmax: [topi.generic.schedule_softmax],
        relay.op.nn.max_pool2d: [topi.generic.schedule_pool],
        relay.op.nn.avg_pool2d: [topi.generic.schedule_pool],
        relay.op.nn.global_avg_pool2d: [topi.generic.schedule_adaptive_pool],
        relay.op.nn.global_max_pool2d: [topi.generic.schedule_adaptive_pool],
        relay.op.nn.deformable_conv2d: [topi.generic.schedule_deformable_conv2d_nchw],
        relay.op.mean: [topi.generic.schedule_reduce],
        relay.op.prod: [topi.generic.schedule_reduce],
        relay.op.nn.conv3d: [topi.generic.schedule_conv3d_ncdhw,
                             topi.generic.schedule_conv3d_ndhwc],
        relay.op.nn.adaptive_avg_pool3d: [topi.generic.schedule_adaptive_pool],
        relay.op.nn.batch_matmul: [topi.generic.schedule_batch_matmul],
    }

def extract_from_program(mod, params, ops, target, target_host=None):
    """ Extract tuning tasks from a relay program.

    This function is the single program version of extract_from_multiple_program.

    Parameters
    ----------
    mod : relay.Module
        The module to extract.
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
    workloads: Array of Tuple(wkl_key, target)
    """
    return extract_from_multiple_program([mod], [params], ops, target, target_host)

def extract_from_multiple_program(mods, params, ops, target, target_host=None):
    """ Extract tuning tasks from multiple relay programs.

    This function collects tuning tasks by building a list of programs
    with a "tracing" target and tracing all the calls to topi.

    Parameters
    ----------
    mods : List of relay.Module
        The modules to extract.
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
    workloads: Array of Tuple(wkl_key, target)
    """
    from tvm import relay

    env = TaskExtractEnv.get()

    init_op_to_schedule_map()
    topi_scheds = []
    for op_name in ops:
        if op_name in OP_TO_SCHEDULE:
            topi_scheds.extend(OP_TO_SCHEDULE[op_name])
        else:
            warnings.warn("Op %s is not tunable, ignored." % op_name)

    # run compiler to collect all TOPI calls during compilation
    env.reset(topi_scheds)
    with env:
        for mod, param in zip(mods, params):
            # wrap build call in thread to avoid multiprocessing problems
            with BlockingEmptyContext():
                build_thread = threading.Thread(target=_lower,
                                        args=(mod, target, param))
        build_thread.start()
        build_thread.join()
        relay.backend.compile_engine.get().clear()

    # create tasks for target
    wkl_keys = []
    wkl_weights = []
    for wkl_key, wkl_weight in env.get_wkl_keys().items():
        wkl_keys.append(wkl_key)
        wkl_weights.append(wkl_weight)

    return wkl_keys, wkl_weights

def prepare_layout_rewrite(mod, params, ops, target):
    """Prepare for kernel layout rewrite. This function will write layout infos to a global static variable,
       then these layout info will be used by a relay pass `kernel_layout_transform`.
    """
    from .. import relay

    env = TaskExtractEnv.get(do_layout_rewrite=True)

    init_op_to_schedule_map()
    topi_scheds = []
    for op_name in ops:
        if op_name in OP_TO_SCHEDULE:
            topi_scheds.extend(OP_TO_SCHEDULE[op_name])
        else:
            warnings.warn("Op %s is not tunable, ignored." % op_name)

    with env:
        env.reset(topi_scheds)

        # wrap build call in thread to avoid multiprocessing problems
        build_thread = threading.Thread(target=_lower,
                                        args=(mod, target, params))
        build_thread.start()
        build_thread.join()
        relay.backend.compile_engine.get().clear()

    if env.layout_rewrite_success_ct > 0:
        GLOBAL_SCOPE.topi_in_compute_rewrite_mode = True

def finish_layout_rewrite():
    """Clear the global flag for layout rewrite"""
    GLOBAL_SCOPE.topi_in_compute_rewrite_mode = False
