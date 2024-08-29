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
.. _optimize_model:

End-to-End Optimize Model
=========================
This tutorial demonstrates how to optimize a machine learning model using Apache TVM. We will
use a pre-trained ResNet-18 model from PyTorch and end-to-end optimize it using TVM's Relax API.
Please note that default end-to-end optimization may not suit complex models.
"""

######################################################################
# Preparation
# -----------
# First, we prepare the model and input information. We use a pre-trained ResNet-18 model from
# PyTorch.

import os
import sys
import numpy as np
import torch
from torch import fx
from torchvision.models.resnet import ResNet18_Weights, resnet18

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT)

######################################################################
# Review Overall Flow
# -------------------
# .. figure:: https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_overall_flow.svg
#    :align: center
#    :width: 80%
#
# The overall flow consists of the following steps:
#
# - **Construct or Import a Model**: Construct a neural network model or import a pre-trained
#   model from other frameworks (e.g. PyTorch, ONNX), and create the TVM IRModule, which contains
#   all the information needed for compilation, including high-level Relax functions for
#   computational graph, and low-level TensorIR functions for tensor program.
# - **Perform Composable Optimizations**: Perform a series of optimization transformations,
#   such as graph optimizations, tensor program optimizations, and library dispatching.
# - **Build and Universal Deployment**: Build the optimized model to a deployable module to the
#   universal runtime, and execute it on different devices, such as CPU, GPU, or other accelerators.
#


######################################################################
# Convert the model to IRModule
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next step, we convert the model to an IRModule using the Relax frontend for PyTorch for further
# optimization. Besides the model, we also need to provide the input shape and data type.

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_fx

torch_model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Give the input shape and data type
input_info = [((1, 3, 224, 224), "float32")]

# Convert the model to IRModule
with torch.no_grad():
    torch_fx_model = fx.symbolic_trace(torch_model)
    mod = from_fx(torch_fx_model, input_info, keep_params_as_input=True)

mod, params = relax.frontend.detach_params(mod)
mod.show()

######################################################################
# IRModule Optimization
# ---------------------
# Apache TVM Unity provides a flexible way to optimize the IRModule. Everything centered
# around IRModule optimization can be composed with existing pipelines. Note that each
# transformation can be combined as an optimization pipeline via ``tvm.ir.transform.Sequential``.
#
# In this tutorial, we focus on the end-to-end optimization of the model via auto-tuning. We
# leverage MetaSchedule to tune the model and store the tuning logs to the database. We also
# apply the database to the model to get the best performance.
#

TOTAL_TRIALS = 8000  # Change to 20000 for better performance if needed
target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")  # Change to your target device
work_dir = "tuning_logs"

# Skip running in CI environment
IS_IN_CI = os.getenv("CI", "") == "true"
if IS_IN_CI:
    sys.exit(0)

with target:
    mod = tvm.ir.transform.Sequential(
        [
            # Convert BatchNorm into a sequence of simpler ops for fusion
            relax.transform.DecomposeOpsForInference(),
            # Canonicalize the bindings
            relax.transform.CanonicalizeBindings(),
            # Run default optimization pipeline
            relax.get_pipeline("zero"),
            # Tune the model and store the log to database
            relax.transform.MetaScheduleTuneIRMod({}, work_dir, TOTAL_TRIALS),
            # Apply the database
            relax.transform.MetaScheduleApplyDatabase(work_dir),
        ]
    )(mod)

# Only show the main function
mod["main"].show()

######################################################################
# Build and Deploy
# ----------------
# Finally, we build the optimized model and deploy it to the target device.

ex = relax.build(mod, target="cuda")
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
gpu_data = tvm.nd.array(np.random.rand(1, 3, 224, 224).astype("float32"), dev)
gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]
gpu_out = vm["main"](gpu_data, *gpu_params).numpy()

print(gpu_out.shape)
