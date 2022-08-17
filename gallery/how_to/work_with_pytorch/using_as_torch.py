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
Wrap Your TVMscript with PyTorch Module
======================
**Author**: `Yaoda Zhou <https://github.com/juda/>`_
This article is an introductory tutorial on wrapping the TVMscript code with the PyTorch module.
By the decorator `as_torch`, users can wrap a TVMscript code into a PyTorch nn.Module naturally.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

# Import PyTorch, as well as necessary libraries
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import tvm
from tvm.contrib.torch import as_torch
from tvm.script import tir as T

######################################################################
# Write your own PyTorch operator by TVMscript
# -------------------------------
# PyTorch is a very popular machine learning framework in which
# it highly optimizes most commonly used operators.
# Nevertheless, sometimes you might want to write your own operators
# in PyTorch, but the performance could be not satisfactory.
#
# For example, assume you are writing a variance of MobileNet,
# and you need to define a 1-d depthwise convolution operator.
# Assume the number of in_channel and out_channel are both 700,
# the width is 800 and the kernel size is 50,
# then the 1-d depthwise conv could be written by PyTorch in one line:

in_channel = 70
out_channel = 70
width = 80
kernel_size = 20


def torch_depthwise(inputs, filters):
    global out_channel
    global kernel_size
    return F.conv1d(inputs, filters.view(out_channel, 1, kernel_size), groups=out_channel)


# We can run this function as:

inputs = torch.randn(in_channel, width)
filters = torch.randn(out_channel, kernel_size)
ret_torch = torch_depthwise(inputs, filters)

# The `torch_depthwise` function, in a plain python code, could be written as:


def vanilla_depthwise(input, weight):
    ret = torch.zeros(out_channel, width - kernel_size + 1)
    for j in range(out_channel):
        for i in range(width - kernel_size + 1):
            for k in range(kernel_size):
                ret[j, i] += weight[j, k] * input[j, i + k]
    return ret


# We plan to optimize the `depthwise` function by leveraging the power of TVMscript.
# Firstly, we can write such a simple python code:


@as_torch
def tvm_depthwise_initializer(Channels: int, Width: int, Kernel: int, Output: int, dtype: str):
    @T.prim_func
    def f(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (Channels, Width), dtype)
        B = T.match_buffer(b, (Channels, Kernel), dtype)
        C = T.match_buffer(c, (Channels, Output), dtype)
        for j, i, k in T.grid(Channels, Output, Kernel):
            with T.block():
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vj, vi] = T.float32(0)
                C[vj, vi] += B[vj, vk] * A[vj, vi + vk]

    return f


# Then we fill out the parameters and generate the TVMscript program.

tvm_depthwise = tvm_depthwise_initializer(
    out_channel, width, kernel_size, width - kernel_size + 1, "float32"
)

# We can tune the TVMscript code by providing a target device.
# The model will deploy on CPU, and the optimization (e.g. tiling) will conduct automatically.

tvm_depthwise.tune()

# We can print out the tuned TVMscript code, as

print(tvm_depthwise.script())

# Hint: If user plan to deploy on GPU, the GPU target should be provided,
# and all the PyTorch tensors should convert into GPU version.
# The thread bindings will be added automatically.

# We can verify that the two functions are the same:

ret_tvm = torch.zeros(out_channel, width - kernel_size + 1)
tvm_depthwise(inputs, filters, ret_tvm)

testing.assert_allclose(ret_torch.cpu().numpy(), ret_tvm.cpu().numpy(), atol=1e-5, rtol=1e-5)


######################################################################
# Benchmark
# -------------------------------
# We will compare two operators by using PyTorch's benchmark toolkit.

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
            description="TVMscript",
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

# In the working machine, the average inference time of `tvm_depthwise` is 120.0 us (TVM version is 0.9.0),
# while the average inference time of `torch_depthwise` is 210.0 us (PyTorch version is 1.11.0),
# showing the performance arises by around 43%.
