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
# pylint: disable=unused-argument, invalid-name
"""VTA specific buildin for runtime."""
import tvm
from . import transform
from .environment import get_env


def EarlyRewrite():
    """Try to do storage rewrite in early pass."""
    def _transform(mod, ctx):
        try:
            return tvm.tir.transform.StorageRewrite()(mod)
        except tvm.error.TVMError:
            return mod
    return tvm.transform.module_pass(
        _transform, opt_level=0, name="tir.vta.EarlyRewrite")


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

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def add_debug(f, *_):
        debug = tvm.tir.call_extern(
            "int32", "VTASetDebugMode",
            env.dev.command_handle,
            debug_flag)

        return f.with_body(tvm.tir.stmt_seq(debug, f.body))


    pass_list = [(0, transform.InjectConv2DTransposeSkip()),
                 (1, transform.InjectDMAIntrin()),
                 (1, transform.InjectSkipCopy()),
                 (1, transform.AnnotateALUCoProcScope()),
                 (1, tvm.tir.transform.LiftAttrScope("coproc_uop_scope")),
                 (1, transform.LiftAllocToScopeBegin()),
                 (1, tvm.tir.transform.LiftAttrScope("coproc_scope")),
                 (1, transform.InjectCoProcSync()),
                 (1, EarlyRewrite())]
    if debug_flag:
        pass_list.append((1, add_debug))
    pass_list.append((2, transform.InjectALUIntrin()))
    pass_list.append((3, tvm.tir.transform.LowerDeviceStorageAccessInfo()))
    pass_list.append((3, transform.FoldUopLoop()))
    pass_list.append((3, transform.CPUAccessRewrite()))
    return tvm.target.build_config(add_lower_pass=pass_list, **kwargs)


def lower(*args, **kwargs):
    """Thin wrapper of tvm.lower

    This wrapper automatically applies VTA's build_config
    if there is no user specified build_config in context.

    See Also
    --------
    tvm.lower : The original TVM's lower function
    """
    cfg = tvm.target.BuildConfig.current()
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
    cfg = tvm.target.BuildConfig.current()
    if not cfg.add_lower_pass:
        with build_config():
            return tvm.build(*args, **kwargs)
    return tvm.build(*args, **kwargs)
