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
from tvm.meta_schedule.tune import TuneConfig, tune_tir
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
        config: TuneConfig = None,
    ):
        super().__init__()
        self.rt_module = None  # runtime module
        self.ir_module = module  # IR modules
        self.config = config

    def tune_tir_auto(self, mod, target):
        if target is None:
            target = Target("llvm --num-cores=16")
        with tempfile.TemporaryDirectory() as work_dir:
            sch: Schedule = tune_tir(
                mod=mod,
                target=target,
                config=self.config,
                work_dir=work_dir,
            )
            return sch.mod

    def build(self, target=None):
        if self.config is not None:
            module = self.tune_tir_auto(self.ir_module, target)
        else:
            module = self.ir_module
        runtime_module = tvm.build(module, target=target)
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.rt_module = torch.classes.tvm_torch.OperatorModuleWrapper()

    def forward(self, *torch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.rt_module is None:
            if torch_inputs[0].is_cuda:
                self.build(target=Target("nvidia/geforce-rtx-3070"))
            elif torch_inputs[0].device.type == "cpu":
                self.build()
            else:
                raise Exception(f"the target {torch_inputs[0].device.type} is not supported yet")

        return self.rt_module.forward(torch_inputs)


def as_torch(inp):
    """A decorator of converting TensorIR to PyTorch nn.Module.

    Parameters
    ----------
    config: Optional[TuneConfig]
        The configuration for tuning by MetaSchedule.
        If user doesn't set the config, the tuning will run with a default setting.

    Returns
    -------
    mod : Union[OperatorModuleWrapper, Callable]
        It will return an object, or a templated function of OperatorModuleWrapper,
        which is the subclass of the original nn.Module.

    """

    def as_torch_inner(func):
        if isinstance(inp, TuneConfig):
            config = inp
        else:
            config = None
        if isinstance(func, (tvm.ir.module.IRModule, tvm.tir.function.PrimFunc)):
            return OperatorModuleWrapper(func, config)
        if isinstance(func, Callable):

            def func_get_param(*args, **kargs):
                return OperatorModuleWrapper(func(*args, **kargs), config)

            return func_get_param
        raise Exception("Incorrect `as_torch` formatting.")

    if isinstance(inp, TuneConfig):
        return as_torch_inner
    return as_torch_inner(inp)
