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
"""Test script for torch module"""
import torch
import time
import tvm
from tvm.contrib.torch import compile, TraceTvmModule, pytorch_tvm


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x * y


model = Model()
x = torch.rand([1, 2, 3])
y = torch.rand([1, 2, 3])
model_jit = torch.jit.script(model)

option = {
    "input_infos": [("x", (1, 2, 3)), ("y", (1, 2, 3))],
    "default_dtype": "float32",
    "export_dir": "pytorch_compiled",
    "num_outputs": 1,
    "tuning_n_trials": 0,  # set zero to skip tuning
    "tuning_log_file": "tuning.log",
    "target": "llvm",
    "device": tvm.cpu(),
}

# use TraceTvmModule to convert List[Tensor] input/output
# to tuple of Tensors
pytorch_tvm_module = compile(model_jit, option)
scripted = torch.jit.script(pytorch_tvm_module)
traced = torch.jit.trace(TraceTvmModule(scripted), (x, y))

res_traced = traced.forward(x, y)
res_expected = pytorch_tvm_module.forward([x, y])[0]
tvm.testing.assert_allclose(res_traced, res_expected)
