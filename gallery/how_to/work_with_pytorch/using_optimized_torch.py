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
`Yaoda Zhou <https://github.com/juda>`_

This article is a tutorial to optimize PyTorch models by using decorator `optimize_torch`.
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
# Optimize SimpleModel by TVM MetaSchedule
# ------------------------------
# We provide the `optimize_torch` function, which has the similar usage as `torch.jit.trace`.
# The PyTorch model to optimize, along with its example input, are provided by users.
# The PyTorch module will be tuned by TVM for the target hardware.
# Without providing extra information, the model will be tuned for CPU.

simple_model = SimpleModel()
example_input = torch.randn(20, 1, 10, 10)
model_optimized_by_tvm = optimize_torch(simple_model, example_input)

######################################################################
# Save/Load module
# ------------------------------
# We can save and load our tuned module like the standard `nn.Module`.

# Let us run our tuned module.
ret1 = model_optimized_by_tvm(example_input)

torch.save(model_optimized_by_tvm, "model_optimized.pt")
model_loaded = torch.load("model_optimized.pt")

# We load the module and run it again.
ret2 = model_loaded(example_input)

# We will show 2 results:
# (1) we can safely load and save model by showing the result of model
# after save and load operations is still the same as original one;
# (2) the model we optimize returns the same result as the original PyTorch model.

ret3 = simple_model(example_input)
testing.assert_allclose(ret1.detach().numpy(), ret2.detach().numpy(), atol=1e-5, rtol=1e-5)
testing.assert_allclose(ret1.detach().numpy(), ret3.detach().numpy(), atol=1e-5, rtol=1e-5)

######################################################################
# Optimize resnet18
# ------------------------------
# In the following, we will show that our approach is able to
# accelerate common models, such as resnet18.

# We will tune our model for the GPU.
target_cuda = "nvidia/geforce-rtx-3070"

# For PyTorch users, the code could be written as usual, except for
# applying "optimize_torch" function on the resnet18 model.

resnet18_tvm = optimize_torch(
    resnet18().cuda().eval(), [torch.rand(1, 3, 224, 224).cuda()], target=target_cuda
)

# TorchScript also provides a built-in "optimize_for_inference" function to accelerate the inference.
resnet18_torch = torch.jit.optimize_for_inference(torch.jit.script(resnet18().cuda().eval()))


######################################################################
# Compare the performance between two approaches.
# ------------------------------

results = []
for i in range(5):
    test_input = torch.rand(1, 3, 224, 224).cuda()
    sub_label = f"[test {i}]"
    results.append(
        benchmark.Timer(
            stmt="resnet18_tvm(test_input)",
            setup="from __main__ import resnet18_tvm",
            globals={"test_input": test_input},
            sub_label=sub_label,
            description="tuning by meta",
        ).blocked_autorange()
    )
    results.append(
        benchmark.Timer(
            stmt="resnet18_torch(test_input)",
            setup="from __main__ import resnet18_torch",
            globals={"test_input": test_input},
            sub_label=sub_label,
            description="tuning by jit",
        ).blocked_autorange()
    )

compare = benchmark.Compare(results)
compare.print()

# In author's environment, the average inference time of `resnet18_tvm` is 620.0 us,
# while the average inference time of `resnet18_torch` is 980.0 us (PyTorch version is 1.11.0),
# showing the speedup of around 38%.
