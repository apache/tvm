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
import os
import torch
import time
import numpy as np
import tvm
import tvm.testing
import tempfile
from tvm.contrib.torch import PyTorchTVMModule, compile


class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y.softmax(1))


model = Model()
model.cuda().half()
x = torch.rand([1280, 2464, 4]).cuda().half()
y = torch.rand([1280, 4, 1]).cuda().half()
for i in range(20):
    t = time.time()
    o = model(x, y)
    torch.cuda.synchronize()
    print(1000 * (time.time() - t))
print(o.shape)


model_jit = torch.jit.script(model)
print(model_jit.graph)
input_shapes = [("x", list(x.shape)), ("y", list(y.shape))]
dtype = "float16"
export_dir = tempfile.mkdtemp("pytorch_compiled")
print("tmp export_dir:", export_dir)


mod = PyTorchTVMModule()
print("Converting...")
mod.from_pytorch(model_jit, input_shapes, dtype)

log_file = os.path.join(export_dir, "tuning.log")
if not os.path.exists(log_file):
    print("Tuning...")
    mod.tune_tvm(log_file=log_file, n_trial=20)

print("Building...")
tvm_mod = mod.build_tvm(export_dir)
pytorch_mod = mod.build_pytorch_module(num_inputs=2, num_outputs=1)


## Or you can load from a prebuilt tvm module
# mod = PyTorchTVMModule()
# tvm_mod = mod.load_tvm(export_dir)
# pytorch_mod = mod.build_pytorch_module(num_inputs=2, num_outputs=1, input_infos=input_shapes)


print("Run TVM...")
tvm_x = tvm.nd.array(x.cpu().numpy().astype(dtype), device=tvm.gpu(0))
tvm_y = tvm.nd.array(y.cpu().numpy().astype(dtype), device=tvm.gpu(0))
for i in range(20):
    t = time.time()
    tvm_mod.run(x=tvm_x, y=tvm_y)
    print(1000 * (time.time() - t))
tvm_output = tvm_mod.get_output(0)
print(tvm_output.shape)


print("Run PyTorch...")
for i in range(20):
    t = time.time()
    outputs = pytorch_mod.forward([x, y])
    torch.cuda.synchronize()
    print(1000 * (time.time() - t))
print(outputs[0].shape)


class EnsembleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.jit.script(pytorch_mod)

    def forward(self, x, y, z) -> torch.Tensor:
        if x > 1:
            out = self.layer(y, z)[0]
        else:
            out = torch.ones([1280, 2464, 1])
        return out


print("Exporting...")
scripted = torch.jit.script(EnsembleModel())
print(scripted.graph)
scripted_path = os.path.join(export_dir, "model_tvm.pt")
scripted.save(scripted_path)


# print(o == outputs[0])
# print(o - outputs[0])
