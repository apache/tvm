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
from typing import Callable, List, Tuple, Union

import torch
import torch.utils.dlpack

import tvm


class OperatorModuleWrapper(torch.nn.Module):
    def __init__(
        self,
        module: Union[
            tvm.ir.module.IRModule,
            tvm.tir.function.PrimFunc,
            tvm.contrib.graph_executor.GraphModule,
        ],
    ):
        super().__init__()
        self.rt_module = None  # runtime module
        self.ir_module = module  # IR moudle

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
    func : Union[tvm.ir.module.IRModule, tvm.tir.function.PrimFunc, Callable]
        The function to be parsed.


    Returns
    -------
    mod : OperatorModuleWrapper
        It will return an object of OperatorModuleWrapper, which is the subclass of the original nn.Module.
    """
    if isinstance(func, tvm.ir.module.IRModule) or isinstance(func, tvm.tir.function.PrimFunc):
        return OperatorModuleWrapper(func)
    elif isinstance(func, Callable):

        def func_get_param(*args, **kargs):
            return OperatorModuleWrapper(func(*args, **kargs))

        return func_get_param
