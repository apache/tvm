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
"""MetaSchedule-Relay integration"""
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore
from tvm import nd
from tvm._ffi import get_global_func
from tvm.ir import IRModule, transform
from tvm.runtime import NDArray
from tvm.target import Target

from .extracted_task import ExtractedTask
from .utils import autotvm_silencer


def extract_task_from_relay(
    mod: IRModule,
    target: Target,
    params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    disabled_pass: Optional[List[str]] = None,
    tir_converter: str = "default",
) -> List[ExtractedTask]:
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod : IRModule
        The module or function to tune
    target : tvm.target.Target
        The compilation target
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    opt_level : int
        The optimization level of the compiler
    pass_config : Optional[Dict[str, Any]]
        The pass config of the compiler
    disabled_pass : Optional[List[str]]
        The list of disabled passes of the compiler
    tir_converter : str
        The filter function to filter out the extracted tasks. Builtin filters:
          - "default"
          - "allow_extern"
        The converter is a PackedFunc registered as f"relay.backend.tir_converter.{tir_converter}",
        with the signature below:
            (args: List[te.Tensor], constants: List[NDArray]) -> Optional[tir.PrimFunc]

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this network
    """
    # pylint: disable=import-outside-toplevel
    from tvm import autotvm
    from tvm.relay import Function as RelayFunc

    # pylint: enable=import-outside-toplevel

    extract_task_func = get_global_func(
        "relay.backend.MetaScheduleExtractTask",
        allow_missing=False,
    )

    if isinstance(mod, RelayFunc):
        mod = IRModule.from_expr(mod)
    if not isinstance(target, Target):
        target = Target(target)
    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": tir_converter,
        }
    if params is None:
        params = {}
    relay_params = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = nd.array(param)
        relay_params[name] = param

    with target, autotvm_silencer(), transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
    ):
        if target.kind.name != "cuda" and isinstance(
            autotvm.DispatchContext.current, autotvm.FallbackContext
        ):
            tophub_context = autotvm.tophub.context(target)
        else:
            tophub_context = autotvm.utils.EmptyContext()
        with tophub_context:
            return list(extract_task_func(mod, target, relay_params))


def is_meta_schedule_enabled() -> bool:
    """Return whether the meta-schedule is enabled.

    Returns
    -------
    enabled: bool
        Whether the meta schedule is enabled
    """
    return transform.PassContext.current().config.get(
        "relay.backend.use_meta_schedule",
        False,
    )


def is_meta_schedule_dispatch_enabled() -> bool:
    """Return whether the meta-schedule dispatch is enabled.

    Returns
    -------
    enabled: bool
        Whether the meta schedule is enabled
    """
    return transform.PassContext.current().config.get(
        "relay.backend.use_meta_schedule_dispatch",
        False,
    )
