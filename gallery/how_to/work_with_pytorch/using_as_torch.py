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
Wrap Your TVMScript as PyTorch Module
=====================================
**Author**:
`Yaoda Zhou <https://github.com/juda>`_

This article is a tutorial on wrapping the TVMScript code as the PyTorch module.
Using the decorator `as_torch`, users can wrap TVMScript code into a PyTorch nn.Module naturally.
To follow the tutorial, PyTorch should be installed:

.. code-block:: bash

    %%shell
    pip install torch

"""


# Import PyTorch, as well as necessary libraries
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import tvm
from tvm.contrib.torch import as_torch
from tvm.script import tir as T

######################################################################
# Write your own PyTorch operator by TVMScript
# --------------------------------------------
# PyTorch is a very popular machine learning framework which contains
# optimized implementations of most commonly used operators.
# Nevertheless, sometimes you might want to write your own operators in PyTorch.
# In that case, the performance of such custom operators might not be satisfactory for your needs.
#
# For example, suppose that we are going to define a 1-d depthwise convolution operator.
# Assume the number of in_channel and out_channel are both 70,
# the width is 80 and the kernel size is 20,
# then the 1-d depthwise conv could be written in PyTorch in one line:

in_channel = 70
out_channel = 70
width = 80
kernel_size = 20


def torch_depthwise(inputs, filters):
    return F.conv1d(inputs, filters.view(out_channel, 1, kernel_size), groups=out_channel)


# We can run this function as:

inputs = torch.randn(in_channel, width)
filters = torch.randn(out_channel, kernel_size)
ret_torch = torch_depthwise(inputs, filters)


# The `torch_depthwise` function, in a plain Python code, could be written as:


def vanilla_depthwise(input, weight):
    ret = torch.zeros(out_channel, width - kernel_size + 1)
    for j in range(out_channel):
        for i in range(width - kernel_size + 1):
            for k in range(kernel_size):
                ret[j, i] += weight[j, k] * input[j, i + k]
    return ret


# Then, we plan to optimize the `depthwise` function by leveraging the power of TVM.
# TVM community proposes an embedded Domain Specific Language in Python called TVMScript,
# which serves as the high-level frontend for TVM's Tensor IR.
# The depthwise 1D convolution code above can be translated to TVMScript as follows.
# We provide an `as_torch` decorator, which converts the TVMScript code to PyTorch's nn.Module automatically.


@as_torch
@T.prim_func
def tvm_depthwise(
    A: T.Buffer((70, 80), "float32"),
    B: T.Buffer((70, 20), "float32"),
    C: T.Buffer((70, 61), "float32"),
) -> None:
    for j, i, k in T.grid(70, 61, 20):
        with T.block():
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vj, vi] = T.float32(0)
            C[vj, vi] += B[vj, vk] * A[vj, vi + vk]


# We can build the TVMScript code by calling the `tune` method in default setting.
# Without providing extra information, the model will be tuned for CPU.

tvm_depthwise.tune()

# We can print out the tuned TVMScript code to see how the program is transformed, as

print(tvm_depthwise.script())

# We can verify that the two outputs are the same:

ret_tvm = torch.zeros(out_channel, width - kernel_size + 1)
tvm_depthwise(inputs, filters, ret_tvm)

testing.assert_allclose(ret_torch.cpu().numpy(), ret_tvm.cpu().numpy(), atol=1e-5, rtol=1e-5)


######################################################################
# Benchmark
# ---------

results = []
for i in range(5):
    inputs = torch.randn(out_channel, width)
    filters = torch.randn(out_channel, kernel_size)
    res = torch.zeros(out_channel, width - kernel_size + 1)
    sub_label = f"[test {i}]"
    results.append(
        benchmark.Timer(
            stmt="tvm_depthwise(inputs, filters, res)",
            setup="from __main__ import tvm_depthwise",
            globals={"inputs": inputs, "filters": filters, "res": res},
            sub_label=sub_label,
            description="TVMScript",
        ).blocked_autorange()
    )
    results.append(
        benchmark.Timer(
            stmt="torch_depthwise(inputs, filters)",
            setup="from __main__ import torch_depthwise",
            globals={
                "inputs": inputs,
                "filters": filters,
            },
            sub_label=sub_label,
            description="PyTorch",
        ).blocked_autorange()
    )
compare = benchmark.Compare(results)
compare.print()

# In author's environment, the average inference time of `tvm_depthwise` is 120.0 us,
# while the average inference time of `torch_depthwise` is 196.0 us (PyTorch version is 1.11.0),
# showing the speedup of around 38%.
