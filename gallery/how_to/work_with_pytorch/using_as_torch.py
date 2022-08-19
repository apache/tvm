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
**Author**: 
`Yaoda Zhou <https://github.com/juda>`_,
`Masahiro Masuda <https://github.com/masahi>`_

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
# PyTorch is a very popular machine learning framework which contains
# optimized implementations of most commonly used operators.
# Nevertheless, sometimes you might want to write your own operators in PyTorch.
# In that case, the performance of such custom operators might not be satisfactory for your needs.
#
# One of the examples is to define a 1-d depthwise convolution operator.
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
# TVM community proposes an embedded Domain Specific Language on Python call TVMscript,
# which serves for an abstraction of program on various hardware backends.

# As a concrete example, we can write such a TVMscript for 1-d depthwise conv code as below.
# The computation procedure of `tvm_depthwise` is corresponding to the code snippet of `vanilla_depthwise`.

# In our `tvm_depthwise` function, both inputs and outputs are set to be function parameters
# that held on the multi-dimension buffers. For each buffer, the shape and data type information are required.
# In the function body, there is a syntactic sugar `T.grid` for writing multiple nested iterators.
# In the body of the loop, each computation is wrapped in an additional construct named `T.block`.
# A block is a basic unit of computation. Inside the block, we need to provide a few more information about the block axes.
# Here, 2 spatial and 1 reduce block iterators are created and bound to the loop iterators i, j and k.
# The computations and machine learning compilation analysis will be defined around them.
# The last 3 lines are computation statements, including an initialization of `C[vj, vi]` and the summing up along the axis k.
# Finally, we place 2 decorators `T.prim_func` and `as_torch` above the definition of function,
# which converts the Python AST to TVMscript AST and then converts to PyTorch's `nn.Module`.


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


# We can build the TVMscript code by calling the `tune` method.
# Without providing more information, the model will be tuned for CPU.

tvm_depthwise.tune()

# We can print out the tuned TVMscript code to see how the program is transformed, as

print(tvm_depthwise.script())

# We can verify that the two outputs are the same:

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

# In author's environment, the average inference time of `tvm_depthwise` is 120.0 us (TVM version is 0.9.0),
# while the average inference time of `torch_depthwise` is 196.0 us (PyTorch version is 1.11.0),
# showing the speedup of around 38%.
