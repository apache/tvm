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
# pylint: disable=unused-argument
"""VTA specific buildin for runtime."""
from __future__ import absolute_import as _abs

import tvm
from tvm import rpc
from . import ir_pass
from .environment import get_env
from .testing import simulator


def lift_coproc_scope(x):
    """Lift coprocessings cope to the """
    x = ir_pass.lift_alloc_to_scope_begin(x)
    x = tvm.ir_pass.LiftAttrScope(x, "coproc_scope", False)
    return x

def early_rewrite(stmt):
    """Try to do storage rewrite in early pass."""
    try:
        return tvm.ir_pass.StorageRewrite(stmt)
    except tvm.TVMError:
        return stmt


def build_config(debug_flag=0, **kwargs):
    """Build a build config for VTA.

    Parameters
    ----------
    debug_flag : int
        The dbeug flag to be passed.

    kwargs : dict
        Additional configurations.

    Returns
    -------
    build_config: BuildConfig
        The build config that can be used in TVM.

    Example
    --------
    .. code-block:: python

      # build a vta module.
      with vta.build_config():
          vta_module = tvm.build(s, ...)
    """
    env = get_env()
    def add_debug(stmt):
        debug = tvm.call_extern(
            "int32", "VTASetDebugMode",
            env.dev.command_handle,
            debug_flag)

        return tvm.make.stmt_seq(debug, stmt)
    pass_list = [(1, ir_pass.inject_dma_intrin),
                 (1, ir_pass.inject_skip_copy),
                 (1, ir_pass.annotate_alu_coproc_scope),
                 (1, lambda x: tvm.ir_pass.LiftAttrScope(x, "coproc_uop_scope", True)),
                 (1, lift_coproc_scope),
                 (1, ir_pass.inject_coproc_sync),
                 (1, early_rewrite)]
    if debug_flag:
        pass_list.append((1, add_debug))
    pass_list.append((2, ir_pass.inject_alu_intrin))
    pass_list.append((3, ir_pass.fold_uop_loop))
    pass_list.append((3, ir_pass.cpu_access_rewrite))
    return tvm.build_config(add_lower_pass=pass_list, **kwargs)


def lower(*args, **kwargs):
    """Thin wrapper of tvm.lower

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.lower : The original TVM's lower function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.lower(*args, **kwargs)
    return tvm.lower(*args, **kwargs)


def build(*args, **kwargs):
    """Thin wrapper of tvm.build

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.build : The original TVM's build function
    """
    cfg = tvm.build_module.current_build_config()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)

# TODO(tmoreau89) unify the build with the rest of the build modules
def vta_autotvm_build_func(measure_input, tmp_dir, **kwargs):
    """Custom build func for VTA. Used for autotvm"""

    import time
    import os
    from random import getrandbits
    from tvm.autotvm.util import get_const_tuple
    from tvm.autotvm.measure.measure_methods import BuildResult, InstantiationError

    tic = time.time()
    try:
        filename = os.path.join(tmp_dir, "tmp_func_%0x.tar" % getrandbits(64))
        target, task, config = measure_input

        with target:
            s, args = task.instantiate(config)
            if not config.valid():
                raise InstantiationError(config.errors)

            func = build(s, args, target_host=task.target_host)
            sim = build(s, args)

        arg_info = tuple((get_const_tuple(x.shape), x.dtype) for x in args)
        func.export_library(filename)

        # When targeting VTA test the schedule on simulator first
        # in order to catch runtime errors
        if measure_input.target.device_name == 'vta':
            from vta import reconfig_runtime
            # Note: if you're not running the RPC locally, you cannot benefit
            # from rumtime recompilation...
            local_rpc_port = int(os.environ.get("VTA_LOCAL_SIM_RPC_PORT", "0"))
            if local_rpc_port:
                remote = rpc.connect("localhost", local_rpc_port)
                reconfig_runtime(remote)
            else:
                remote = rpc.LocalSession()
            sim_path = os.path.join(tmp_dir, "tmp_func_%0x.tar" % getrandbits(64))
            sim.export_library(sim_path)
            remote.upload(sim_path)
            f = remote.load_module(os.path.split(sim_path)[1])
            ctx = remote.context(str(measure_input.target), 0)
            args = [tvm.nd.empty(x[0], dtype=x[1], ctx=ctx) for x in arg_info]
            # Skip execution just to verify correctness
            simulator.debug_mode(simulator.DEBUG_SKIP_EXEC)
            f(*args)

        # check by local simulator
        ctx = tvm.context(str(target))
        args = [tvm.nd.empty(x[0], dtype=x[1], ctx=ctx) for x in arg_info]
        sim(*args)

    except Exception as exc: # pylint: disable=broad-except
        return BuildResult(None, None, exc, time.time() - tic)
    return BuildResult(filename, arg_info, None, time.time() - tic)
