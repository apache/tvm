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
# Import TVM
import tvm
import tvm.testing
# Import `optimize_torch` function
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig

# Import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
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

######################################################################
# Optimized SimpleModel by TVM MetaSchedule
# ------------------------------
# We provide a `optimize_torch` function, which have a similar usage as `torch.jit.trace`.
# For the function, we have five parameters need to provide.
# If the third parameter `tuning_config` is not provided, a default configuration is loaded.
# If the parameter `target` is empty, the model will deploy on CPU.


example_input = torch.randn(20, 1, 10, 10)

tuning_config = TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=2,
    max_trials_per_task=2,
    max_trials_global=2,
)


# We use default configuration for the first example
model_optimized_by_meta = optimize_torch(
    SimpleModel(), example_input, tuning_config)

# #####################################################################
# Optimized SimpleModel by TorchScript
# ------------------------------
# As a comparison, we trace this module via `optimize_for_inference` function from TorchScript
model_optimized_by_jit = torch.jit.optimize_for_inference(
    torch.jit.trace(SimpleModel(), example_input))


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
