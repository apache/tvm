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

import base64
import functools
import tempfile
from typing import Callable, Dict, Tuple, Union, List

import torch
import torch.utils.dlpack

import tvm
from tvm import relay
from tvm._ffi import get_global_func, register_func
from tvm.meta_schedule import TuneConfig


class GraphExecutorFactoryWrapper(torch.nn.Module):
    def __init__(
        self,
        module: tvm.runtime.Module
    ):
        super().__init__()
        self.inner_module = module

    def forward(self, *torch_inputs: Tuple[torch.Tensor]):
        ret = self.inner_module.forward(torch_inputs)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret


class TVMScriptIRModule(torch.nn.Module):
    def __init__(self, module: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, tvm.contrib.graph_executor.GraphModule]):
        super().__init__()
        self.engine_cpu = None
        self.engine_cuda = None
        self.ir_module = module

    def __save_cpu_rt_module(self, runtime_module):
        func = tvm.get_global_func("tvmtorch.save_runtime_mod")
        func(runtime_module)

        self.engine_cpu = torch.classes.tvm_torch.TVMScriptRuntime()

    def __build_cpu(self):
        runtime_module = tvm.build(self.ir_module)
        self.__save_cpu_rt_module(runtime_module)

    def __save_cuda_rt_module(self, runtime_module):
        self.engine_cuda = runtime_module

    def __build_cuda(self):
        runtime_module = tvm.build(self.ir_module, target=tvm.target.cuda())
        self.__save_cuda_rt_module(runtime_module)

    def forward(self, *torch_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        if torch_inputs[0].is_cuda:
            if self.engine_cuda is None:
                self.__build_cuda()
            return self.engine_cuda.forward(torch_inputs)
        else:
            if self.engine_cpu is None:
                self.__build_cpu()
            return self.engine_cpu.forward(torch_inputs)


def as_torch(
    func: Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
):
    """A decorator of converting TensorIR to PyTorch nn.Module.

    Parameters
    ----------
    func : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
        The function to be parsed.


    Returns
    -------
    mod : TVMScriptIRModule
        It will return an object of TVMScriptIRModule, which is the subclass of the original nn.Module.
    """
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return TVMScriptIRModule(func)
    elif isinstance(func, Callable):
        def func_get_param(*args, **kargs):
            return TVMScriptIRModule(func(*args, **kargs))
        return func_get_param


@functools.lru_cache(None)
def llvm_target():
    return "llvm -num-cores"


def tuning_relay(mod: tvm.ir.module.IRModule, params: Dict, config: TuneConfig, target, work_dir: str = None):
    from tvm.meta_schedule.tune import tune_relay
    with tempfile.TemporaryDirectory() as tmp_work_dir:
        return tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=config,
            work_dir=work_dir if work_dir else tmp_work_dir,
        )


@register_func("script_torch.save_to_base64")
def save_to_base64(obj) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".so") as tmpfile:
        obj.export_library(tmpfile.name)
        with open(tmpfile.name, "rb") as tfile:
            return base64.b64encode(tfile.read())


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
        A Python function or nn.Module that could run by TorchScript's trace. (ie: torch.jit.trace(model, input))

    example_inputs : tuple or torch.Tensor 
        A tuple of example inputs that
        will run together with `func` by providing the shape information.

    tuning_config : tvm.meta_schedule.TuneConfig
        The configuration of tuning by MetaSchedule.

    target : Optional[Union[str, Target]]
        The target of the compilation.
        If user doesn't set the target, the module is built upon the LLVM.

    work_dir : Optional[str]
        The working directory to save intermediate results.

    Returns
    -------
    mod : GraphExecutorFactoryWrapper
        It will return an object of GraphExecutorFactoryWrapper, which is the subclass of the original nn.Module.
    """

    if target:
        pass
    else:
        target = llvm_target()

    if tuning_config:
        pass
    else:
        # Default setting. For a better tuning result the number could be set larger.
        tuning_config = TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=4,
            max_trials_per_task=16,
            max_trials_global=16,
        )

    jit_mod = torch.jit.trace(func, example_inputs)
    if isinstance(example_inputs, torch.Tensor):
        example_inputs = [example_inputs]
    shape_list = [(f"inp_{idx}", i.shape)
                  for idx, i in enumerate(example_inputs)]
    mod, params = relay.frontend.from_pytorch(jit_mod, shape_list)  # IRmodule
    executor_factory = tuning_relay(
        mod, params, tuning_config, target, work_dir)

    save_runtime_mod = get_global_func("tvmtorch.save_runtime_mod")
    save_runtime_mod(executor_factory.module)

    return GraphExecutorFactoryWrapper(torch.classes.tvm_tuning.RelayRuntime())
