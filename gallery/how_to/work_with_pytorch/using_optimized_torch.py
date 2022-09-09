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
"""
Compile PyTorch Models
======================
**Author**: 
`Yaoda Zhou <https://github.com/juda>`_,
`Masahiro Masuda <https://github.com/masahi>`_

This article is an introductory tutorial to optimize PyTorch models by using `tvm.contrib.torch.optimize_torch`.
To follow this tutorial, PyTorch, as well as TorchVision, should be installed.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import library for profiling
import torch.utils.benchmark as benchmark
from torchvision.models import resnet18

# Import `optimize_torch` function
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig

######################################################################
# Define a simple module written by PyTorch
# ------------------------------


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))


######################################################################
# Optimized SimpleModel by TVM MetaSchedule
# ------------------------------
# We provide a `optimize_torch` function, which has the similar usage as `torch.jit.trace`.
# The PyTorch model to optimize, along with its example input, are provided by users.
# We can optimized the PyTorch's module by calling the `optimized_torch` method in default setting.
# Without providing extra information, the model will be tuned for CPU.

simple_model = SimpleModel()
example_input = torch.randn(20, 1, 10, 10)
model_optimized_by_tvm = optimize_torch(simple_model, example_input)

######################################################################
# Save/Load module
# ------------------------------
# We can save and load our tuned module like the standard `nn.module`.

# Let us run our tuned module.
ret1 = model_optimized_by_tvm(example_input)

torch.save(model_optimized_by_tvm, "model_optimized.pt")
model_loaded = torch.load("model_optimized.pt")

# We load the module and run it again.
ret2 = model_loaded(example_input)

# We show that the results from original SimpleModel,
# optimized model and loaded model are the same.

ret3 = simple_model.forward(example_input)
testing.assert_allclose(ret1.detach().numpy(), ret2.detach().numpy(), atol=1e-5, rtol=1e-5)
testing.assert_allclose(ret1.detach().numpy(), ret3.detach().numpy(), atol=1e-5, rtol=1e-5)

######################################################################
# Resnet18 optimized by TVM MetaSchedule
# ------------------------------
# In the following, we will show that our approach is able to
# accelerate common and large models, such as Resnet18.

# We will tune our model on the GPU.
target_cuda = "nvidia/geforce-rtx-3070"

# For PyTorch users, the nn.Module could be written as usual, except for
# applying "optimize_torch" function on the resnet18 model.


class MyResNet18(torch.nn.Module):
    def __init__(self, target):
        super(MyResNet18, self).__init__()
        # Here we impose the `optimize_torch` function
        # The default setting is adapted automatically by the number of operations of the optimized model.
        self.resnet = optimize_torch(resnet18(), torch.rand(1, 3, 224, 224), target=target)

    def forward(self, input):
        return self.resnet(input)


tvm_module_resnet18 = MyResNet18(target_cuda)


######################################################################
# Resnet18 optimized by TorchScript
# ------------------------------
# Let us write down a resnet18 model in a standard way.


class JitModule(torch.nn.Module):
    def __init__(self):
        super(JitModule, self).__init__()
        # Here we impose the `optimize_for_inference` function
        # TorchScript also provides a built-in "optimize_for_inference" function to accelerate the inference.
        self.resnet = torch.jit.optimize_for_inference(torch.jit.script(resnet18().cuda().eval()))

    def forward(self, input):
        return self.resnet(input)


jit_module_resnet18 = JitModule()


######################################################################
# Compare the performance between two approaches.
# ------------------------------
# Using PyTorch's benchmark Compare class, we can have a direct comparison result between two inference models.

results = []
for i in range(5):
    test_input = torch.rand(1, 3, 224, 224).cuda()
    sub_label = f"[test {i}]"
    results.append(
        benchmark.Timer(
            stmt="tvm_module_resnet18(test_input)",
            setup="from __main__ import tvm_module_resnet18",
            globals={"test_input": test_input},
            sub_label=sub_label,
            description="tuning by meta",
        ).blocked_autorange()
    )
    results.append(
        benchmark.Timer(
            stmt="jit_module_resnet18(test_input)",
            setup="from __main__ import jit_module_resnet18",
            globals={"test_input": test_input},
            sub_label=sub_label,
            description="tuning by jit",
        ).blocked_autorange()
    )

compare = benchmark.Compare(results)
compare.print()

# In author's environment, the average inference time of `tvm_module_resnet18` is 620.0 us (TVM version is 0.9.0),
# while the average inference time of `jit_module_resnet18` is 980.0 us (PyTorch version is 1.11.0),
# showing the speedup of around 38%.
