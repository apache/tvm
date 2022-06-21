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
**Author**: `Yaoda Zhou <https://github.com/juda/>`_
This article is an introductory tutorial to optimize PyTorch models by MetaSchedule.
For us to follow this tutorial, PyTorch as well as TorchVision should be installed.
For avoiding potential "undefined symbol" issue, we strongly recommend to install PyTorch built with Cxx11 ABI from Conda, as
.. code-block:: bash
    conda install -c conda-forge pytorch-gpu
"""
# Import TVM and PyTorch
import tvm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import tvm.testing

# Import `optimize_torch` function
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig

# Import library for profiling
import torch.utils.benchmark as benchmark


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


# #####################################################################
# Optimized SimpleModel by TorchScript
# ------------------------------
# We trace this module, then run the `optimize_for_inference` function from TorchScript
simple_model = SimpleModel()
example_input = torch.randn(20, 1, 10, 10)
model_optimized_by_jit = torch.jit.optimize_for_inference(
    torch.jit.trace(simple_model, example_input))

######################################################################
# Optimized SimpleModel by TVM MetaSchedule
# ------------------------------
# We provide a `optimize_torch` function, which have a similar usage as `torch.jit.trace`.
# For the function, we have five parameters need to provide.
# If the third parameter `tuning_config` is not provided, a default configuration is loaded.
# If the parameter `target` is empty, the model will deploy on CPU.

tuning_config = TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=64,
    max_trials_per_task=2000,
    max_trials_global=2000,
)

# We use default configuration for the first example
model_optimized_by_meta = optimize_torch(
    SimpleModel(), example_input)

######################################################################
# Compare the performance between two scheduling approaches.
# ------------------------------
# Using PyTorch's benchmark Compare class, we can have a straightforward comparison between two inference models.

results = []
for i in range(20):
    test_input = torch.rand(20, 1, 10, 10)
    sub_label = f'[test {i}]'
    results.append(benchmark.Timer(
        stmt='model_optimized_by_meta(test_input)',
        setup='from __main__ import model_optimized_by_meta',
        globals={'test_input': test_input},
        sub_label=sub_label,
        description='tuning by meta',
    ).blocked_autorange())
    results.append(benchmark.Timer(
        stmt='model_optimized_by_jit(test_input)',
        setup='from __main__ import model_optimized_by_jit',
        globals={'test_input': test_input},
        sub_label=sub_label,
        description='tuning by jit',
    ).blocked_autorange())

# We can print the results on screen.
compare = benchmark.Compare(results)
compare.print()

######################################################################
# Save/Load module
# ------------------------------
# We can save and load our tuned module like standard nn.module

# Let us run our tuned module and see the result
ret1 = model_optimized_by_meta(example_input)

torch.save(model_optimized_by_meta, "meta_model.pt")
model_loaded = torch.load("meta_model.pt")

# We load the module and run again and it will return the same result as above.
ret2 = model_loaded(example_input)

tvm.testing.assert_allclose(ret1.numpy(), ret2.numpy(), atol=1e-5, rtol=1e-5)

######################################################################
# Define the resnet18 optimized by MetaSchedule
# ------------------------------
# Another example, we compare the two optimizers about the performance of resnet18
# For learning how to define a resnet18 model via PyTorch's nn.Module,
# you can refer to https://pytorch.org/docs/stable/jit.html#mixing-tracing-and-scripting

# In our working machine, the GPU model is nvidia/geforce-rtx-3070.
target_cuda = "nvidia/geforce-rtx-3070"

# For PyTorch users, you can write your nn.Module in a normal way.
# By applying "optimize_torch" function on the resnet18 model, we obtain a new resnet18 model optimized by MetaSchedule


class MyResNet18(torch.nn.Module):
    def __init__(self, config, target=None):
        super(MyResNet18, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).cuda()
        self.resnet = optimize_torch(
            resnet18(), [torch.rand(1, 3, 224, 224)], config, target)

    def forward(self, input):
        return self.resnet(input - self.means)


# Since the setting of the number of trials is large, the initialization could be slow (sometimes more than 3 hours!)
meta_module_resnet18 = MyResNet18(tuning_config, target_cuda)


######################################################################
# Define the resnet18 optimized by TorchScript
# ------------------------------
# Besides, let us define a resnet18 model in a standard way.
# TorchScript also provide a built-in "optimize_for_inference" function to accelerate the inference.

class JitModule(torch.nn.Module):
    def __init__(self):
        super(JitModule, self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68])
                                        .resize_(1, 3, 1, 1)).cuda()
        self.resnet = torch.jit.optimize_for_inference(
            torch.jit.script(resnet18().cuda().eval()))

    def forward(self, input):
        return self.resnet(input - self.means)


jit_module_resnet18 = JitModule()

######################################################################
# Compare the performance between two scheduling approaches.
# ------------------------------
# Using PyTorch's benchmark Compare class, we can have a straightforward comparison between two inference models.

results = []
for i in range(20):
    test_input = torch.rand(1, 3, 224, 224).half().cuda()
    sub_label = f'[test {i}]'
    results.append(benchmark.Timer(
        stmt='meta_module_resnet18(test_input)',
        setup='from __main__ import meta_module_resnet18',
        globals={'test_input': test_input},
        sub_label=sub_label,
        description='tuning by meta',
    ).blocked_autorange())
    results.append(benchmark.Timer(
        stmt='jit_module_resnet18(test_input)',
        setup='from __main__ import jit_module_resnet18',
        globals={'test_input': test_input},
        sub_label=sub_label,
        description='tuning by jit',
    ).blocked_autorange())

# We can print the results on screen.
compare = benchmark.Compare(results)
compare.print()

# As above, we can save the module for future use
torch.save(meta_module_resnet18, "meta_tuned_resnet18.pt")
