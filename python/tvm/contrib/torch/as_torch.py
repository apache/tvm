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
as_torch: a decorator, which is used to wrap the TVMscript code to `torch.nn.module`.
"""
import tempfile
from typing import Callable, List, Union

import torch
import torch.utils.dlpack

import tvm
from tvm.meta_schedule import default_config
from tvm.meta_schedule.database.database import TuningRecord
from tvm.meta_schedule.extracted_task import ExtractedTask
from tvm.meta_schedule.tune import TuneConfig, tune_extracted_tasks
from tvm.target.target import Target
from tvm.tir.schedule.schedule import Schedule


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

    def tune_tir_auto(self, mod: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc]):
        with tempfile.TemporaryDirectory() as work_dir:
            sch: Schedule = self.tune_tir_inner(
                mod=mod,
                target=Target("llvm --num-cores=16"),
                work_dir=work_dir,
            )
            return sch.mod

    def tune_tir_inner(
        self,
        mod: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc],
        target: Union[str, Target],
        work_dir: str,
    ):
        """Tune a TIR IRModule with a given target.

        Parameters
        ----------
        mod : Union[IRModule, PrimFunc]
            The module to tune.
        target : Union[str, Target]
            The target to tune for.
        work_dir : Optional[str]
            The working directory to save intermediate results.

        Returns
        -------
        sch : Optional[Schedule]
            The tuned schedule.
        """
        mod = default_config.mod(mod)
        target = default_config.target(target)

        extracted_task = ExtractedTask(
            task_name="main",
            mod=mod,
            dispatched=[mod],
            target=target,
            weight=1,
        )
        config = TuneConfig(
            # Default setting
            strategy="replay_trace",
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=32,
        )
        database = tune_extracted_tasks(
            extracted_tasks=[extracted_task], config=config, work_dir=work_dir
        )
        bests: List[TuningRecord] = database.get_top_k(database.commit_workload(mod), top_k=1)
        if not bests:
            return None
        assert len(bests) == 1
        sch = Schedule(mod)
        bests[0].trace.apply_to_schedule(sch, remove_postproc=False)

        return sch

    def build(self, target=None):
        tuned_module = self.tune_tir_auto(self.ir_module)
        runtime_module = tvm.build(tuned_module, target=target)
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
    func : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
        The function to be parsed.


    Returns
    -------
    mod : OperatorModuleWrapper
        It will return an object of OperatorModuleWrapper,
        which is the subclass of the original nn.Module.
    """
    if isinstance(func, (tvm.ir.module.IRModule, tvm.tir.function.PrimFunc)):
        return OperatorModuleWrapper(func)
    if isinstance(func, Callable):

        def func_get_param(*args, **kargs):
            return OperatorModuleWrapper(func(*args, **kargs))

        return func_get_param
