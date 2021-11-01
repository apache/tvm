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
from tvm.contrib.torch import compile


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * x


model = Model()
x = torch.rand([1, 3, 224, 224])
model_jit = torch.jit.trace(model, x)
print(model_jit.graph)

print("run torchscript...")
for i in range(20):
    t = time.time()
    model_jit(x)
    print(time.time() - t)


option = {
    "input_infos": [
        ("x", (1, 3, 224, 224)),
    ],
    "default_dtype": "float16",
    "export_dir": "pytorch_compiled",
    "num_outputs": 1,
    "tuning_n_trials": 1,  # set zero to skip tuning
    "tuning_log_file": "tuning.log",
    "target": "llvm",
    "device": tvm.cpu(),
}

pytorch_tvm_module = compile(model_jit, option)
torch.jit.script(pytorch_tvm_module).save("model_tvm.pt")


print("Run PyTorch...")
for i in range(20):
    t = time.time()
    outputs = pytorch_tvm_module.forward([x.cpu()])
    print(1000 * (time.time() - t))
print(outputs[0].shape)
