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
from typing import Optional, Tuple, Union

import torch
import torch.utils.dlpack
import tvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm._ffi import get_global_func, register_func
from tvm.target import Target


class GraphExecutorFactoryWrapper(torch.nn.Module):
    def __init__(self, module: tvm.runtime.Module):
        super().__init__()
        self.inner_module = module

    def forward(self, *torch_inputs: Tuple[torch.Tensor]):
        ret = self.inner_module.forward(torch_inputs)
        if len(ret) == 1:
            return ret[0]
        return ret


@register_func("script_torch.save_to_base64")
def save_to_base64(obj) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".so") as tmpfile:
        obj.export_library(tmpfile.name)
        with open(tmpfile.name, "rb") as temp_file:
            return base64.b64encode(temp_file.read())


def optimize_torch(
    func,
    example_inputs,
    *,
    max_trials_global: int,
    work_dir=None,
    target: Union[str, Target] = "cpu",
    max_trials_per_task: Optional[int] = None,
    num_trials_per_iter: int = 64,
    builder: ms.Builder.BuilderType = "local",
    runner: ms.Runner.RunnerType = "local",
    database: ms.Database.DatabaseType = "json",
    cost_model: ms.CostModel.CostModelType = "xgb",
    measure_callbacks: ms.MeasureCallback.CallbackListType = "default",
    task_scheduler: ms.TaskScheduler.TaskSchedulerType = "gradient",
    space: ms.SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: ms.SearchStrategy.SearchStrategyType = "evolutionary",
    seed: Optional[int] = None,
):
    """Load PyTorch model that could be traced by TorchScript, then optimize it via MetaSchedule.

    Parameters
    ----------
    func : callable or torch.nn.Module
        A Python function or nn.Module that could run by TorchScript's trace.
        (ie: torch.jit.trace(model, input))
    example_inputs : tuple or torch.Tensor
        Inputs to `torch.jit.trace`.
    max_trials_global : int
        The maximum number of trials to run globally.
    work_dir : Optional[str]
        The working directory to save intermediate results.
    target : Optional[Union[str, Target]]
        The target of the compilation.
        If user doesn't set the target, the module will be built for the CPU target.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator to use.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy to use.
    seed : Optional[int]
        The random seed to use.

    Returns
    -------
    mod : GraphExecutorFactoryWrapper
        It will return an object of GraphExecutorFactoryWrapper,
        which is the subclass of the original nn.Module.
    """

    if target == "cpu":
        target = f"llvm --num-cores {ms.utils.cpu_count(logical=False)}"
    if not isinstance(target, Target):
        target = Target(target)

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
    with context_manager as work_dir:  # pylint: disable=redefined-argument-from-local
        database = ms.relay_integration.tune_relay(
            mod=mod,
            params=params,
            target=target,
            work_dir=work_dir,
            max_trials_global=max_trials_global,
            max_trials_per_task=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            builder=builder,
            runner=runner,
            database=database,
            cost_model=cost_model,
            measure_callbacks=measure_callbacks,
            task_scheduler=task_scheduler,
            space=space,
            strategy=strategy,
            seed=seed,
        )
        executor_factory = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=target,
            params=params,
            backend="graph",
        )

    save_runtime_mod = get_global_func("tvmtorch.save_runtime_mod")
    save_runtime_mod(executor_factory.module)

    return GraphExecutorFactoryWrapper(torch.classes.tvm_torch.GraphExecutorFactoryWrapper())
