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
as_torch: a decorator, which is used to wrap the TVMScript code to `torch.nn.module`.
"""
import tempfile
from typing import Callable, List, Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on

import torch
import torch.utils.dlpack
import tvm
from tvm import meta_schedule as ms
from tvm.target.target import Target
from tvm.tir import PrimFunc


# python wrapper for OperatorModule
class OperatorModuleWrapper(torch.nn.Module):
    def __init__(
        self,
        module: Union[
            tvm.ir.module.IRModule,
            tvm.tir.function.PrimFunc,
        ],
    ):
        super().__init__()
        self.rt_module = None  # runtime module
        self.ir_module = module  # IR modules

    def tune(
        self,
        target: Union[str, Target] = "cpu",
        max_trials_global: int = 32,
        *,
        num_trials_per_iter: int = 32,
        builder: ms.Builder.BuilderType = "local",
        runner: ms.Runner.RunnerType = "local",
        database: ms.Database.DatabaseType = "json",
        cost_model: ms.CostModel.CostModelType = "xgb",
        measure_callbacks: ms.MeasureCallback.CallbackListType = "default",
        task_scheduler: ms.TaskScheduler.TaskSchedulerType = "round-robin",
        space: ms.SpaceGenerator.SpaceGeneratorType = "post-order-apply",
        strategy: ms.SearchStrategy.SearchStrategyType = "replay_trace",
        task_name: str = "main",
        num_threads: Union[Literal["physical", "logical"], int] = "physical",
        seed: Optional[int] = None,
    ) -> None:
        """
        Tune the TVMScript code.

        Parameters
        ----------
        config: Optional[TuneConfig]
            The tuning configuration.

        target : Optional[str, Target]
            The target to tune for.
        """
        if target == "cpu":
            target = f"llvm --num-cores {ms.utils.cpu_count(logical=False)}"

        with tempfile.TemporaryDirectory() as work_dir:
            database = ms.tir_integration.tune_tir(
                mod=self.ir_module,
                target=target,
                work_dir=work_dir,
                max_trials_global=max_trials_global,
                num_trials_per_iter=num_trials_per_iter,
                builder=builder,
                runner=runner,
                database=database,
                cost_model=cost_model,
                measure_callbacks=measure_callbacks,
                task_scheduler=task_scheduler,
                space=space,
                strategy=strategy,
                task_name=task_name,
                num_threads=num_threads,
                seed=seed,
            )
            sch = ms.tir_integration.compile_tir(database, self.ir_module, target)
            self.ir_module = sch.mod
            self.build(target)

    def script(self):
        return self.ir_module.script()

    def build(self, target=None):
        runtime_module = tvm.build(self.ir_module, target=target)
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.rt_module = torch.classes.tvm_torch.OperatorModuleWrapper()

    def forward(self, *torch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.rt_module is None:
            if torch_inputs[0].is_cuda:
                self.build(target="cuda")
            elif torch_inputs[0].device.type == "cpu":
                self.build()
            else:
                raise Exception(f"the target {torch_inputs[0].device.type} is not supported yet")

        return self.rt_module.forward(torch_inputs)


def as_torch(func: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]):
    """A decorator of converting TensorIR to PyTorch nn.Module.

    Parameters
    ----------
    func: Optional[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
        The function written by TVMScript.

    Returns
    -------
    mod : Union[OperatorModuleWrapper, Callable]
        It will return an object, or a templated function of OperatorModuleWrapper,
        which is the subclass of the original nn.Module.

    """
    if isinstance(func, (tvm.ir.module.IRModule, PrimFunc)):
        return OperatorModuleWrapper(func)
    if callable(func):

        def func_get_param(*args, **kwargs):
            return OperatorModuleWrapper(func(*args, **kwargs))

        return func_get_param
