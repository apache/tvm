# pylint: disable=inconsistent-return-statements
#!/usr/bin/env python

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
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""
optimize_torch: a function similar to `torch.jit.trace`,
which is used to optimize the `torch.nn.module` by TVM metaSchedule,
and returns a custom TorchScript operator
"""
import base64
import contextlib
import tempfile
from typing import Dict, Optional, Tuple, Union
import warnings

import torch
import torch.utils.dlpack

import tvm
from tvm import relay
from tvm._ffi import get_global_func, register_func
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.meta_schedule import TuneConfig, default_config
from tvm.meta_schedule.apply_history_best import ApplyHistoryBest
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.utils import autotvm_silencer
from tvm.runtime import vm
from tvm.runtime.module import Module
from tvm.runtime.ndarray import NDArray
from tvm.target.target import Target


# The python wrapper for GraphExecutorFactory
class GraphExecutorFactoryWrapper(torch.nn.Module):
    def __init__(self, module: tvm.runtime.Module):
        super().__init__()
        self.inner_module = module

    def forward(self, *torch_inputs: Tuple[torch.Tensor]):
        ret = self.inner_module.forward(torch_inputs)
        if len(ret) == 1:
            return ret[0]
        return ret


def llvm_target():
    return "llvm -num-cores"


@register_func("script_torch.save_to_base64")
def save_to_base64(obj) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".so") as tmpfile:
        obj.export_library(tmpfile.name)
        with open(tmpfile.name, "rb") as tfile:
            return base64.b64encode(tfile.read())


def tune_relay_auto(
    mod: IRModule,
    target: Union[str, Target],
    config: TuneConfig,
    work_dir: str,
    backend: str = "graph",
    params: Optional[Dict[str, NDArray]] = None,
) -> Union[Module, vm.Executable]:
    """A wrapper of `tune_relay` but provide a default setting for the config.

    Parameters
    ----------
    mod : IRModule
        The module to tune.
    target : Union[str, Target]
        The target to tune for.
    config : TuneConfig
        The search strategy config.
    params : Optional[Dict[str, tvm.runtime.NDArray]]
        The associated parameters of the program
    work_dir : Optional[str]
        The working directory to save intermediate results.
    backend : str = "graph"
        The backend to use for relay compilation(graph / vm).

    Returns
    -------
    lib : Union[Module, tvm.runtime.vm.Executable]
        The built runtime module or vm Executable for the given relay workload.
    """
    target = default_config.target(target)
    extracted_tasks = extract_task_from_relay(mod, target, params)
    if config is None:
        config = TuneConfig(
            num_trials_per_iter=16,
            max_trials_global=16 * len(extracted_tasks),
        )
    database = tune_extracted_tasks(extracted_tasks, config, work_dir)
    relay_build = {"graph": relay.build, "vm": relay.vm.compile}[backend]
    with target, autotvm_silencer(), ApplyHistoryBest(database):
        with PassContext(
            opt_level=3,
            config={
                "relay.backend.use_meta_schedule": True,
                "relay.backend.use_meta_schedule_dispatch": target.kind.name != "cuda",
            },
        ):
            return relay_build(mod, target=target, params=params)


def optimize_torch(
    func,
    example_inputs,
    tuning_config=None,
    target=None,
    work_dir=None,
):
    """Load PyTorch model that could be traced by TorchScript, then optimize it via MetaSchedule.

    Parameters
    ----------
    func : callable or torch.nn.Module
        A Python function or nn.Module that could run by TorchScript's trace.
        (ie: torch.jit.trace(model, input))

    example_inputs : tuple or torch.Tensor
        Inputs to `torch.jit.trace`.

    tuning_config : tvm.meta_schedule.TuneConfig
        The configuration for tuning by MetaSchedule.
        If user doesn't set the config, the tuning will run with a default setting.
        Here, the total number of trials is proportional
        to the number of tunable tasks in the input module.

    target : Optional[Union[str, Target]]
        The target of the compilation.
        If user doesn't set the target, the module will be built for the CPU target.

    work_dir : Optional[str]
        The working directory to save intermediate results.

    Returns
    -------
    mod : GraphExecutorFactoryWrapper
        It will return an object of GraphExecutorFactoryWrapper,
        which is the subclass of the original nn.Module.
    """

    if target is None:
        target = llvm_target()

    if tuning_config is None:
        warning_msg = (
            "Using the default tuning parameters.",
            "The default number of trials is set to a small value to let tuning finish quickly.",
            "For optimal performance, it is recommended to provide",
            "the `tuning_config` argument with a bigger number of trials.",
        )
        warnings.warn(" ".join(warning_msg), stacklevel=2)

    # If `func` is already a traced module this statement makes no effect
    jit_mod = torch.jit.trace(func, example_inputs)

    if isinstance(example_inputs, torch.Tensor):
        example_inputs = [example_inputs]

    shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)  # IRmodule
    if work_dir:
        context_manager = contextlib.nullcontext(work_dir)
    else:
        context_manager = tempfile.TemporaryDirectory()
    with context_manager as work_dir_path:
        executor_factory = tune_relay_auto(
            mod=mod, params=params, config=tuning_config, target=target, work_dir=work_dir_path
        )

    save_runtime_mod = get_global_func("tvmtorch.save_runtime_mod")
    save_runtime_mod(executor_factory.module)

    return GraphExecutorFactoryWrapper(torch.classes.tvm_torch.GraphExecutorFactoryWrapper())
