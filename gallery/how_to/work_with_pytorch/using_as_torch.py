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

in_channel = 700
out_channel = 700
width = 800
kernel_size = 50


def torch_depthwise(inputs, filters):
    return F.conv1d(inputs, filters.view(700, 1, 50), groups=700)


# We can run this function as:

inputs = torch.randn(700, 800).cuda()
filters = torch.randn(700, 50).cuda()
ret_torch = torch_depthwise(inputs, filters)

# The `torch_depthwise` function, in a plain python code,
# could be written as:


def vanilla_depthwise(input, weight):
    ret = torch.zeros(700, 800 - 50 + 1).cuda()
    for j in range(700):
        for i in range(800 - 50 + 1):
            for k in range(50):
                ret[j, i] += weight[j, k] * input[j, i + k]
    return ret


# We plan to optimize the `depthwise` function by
# leveraging the power of TVMscript.
# We can write such a simple TVMscript code:


@as_torch
@tvm.script.ir_module
class tvm_depthwise:
    @T.prim_func
    def f(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (700, 800), "float32")
        B = T.match_buffer(b, (700, 50), "float32")
        C = T.match_buffer(c, (700, 751), "float32")
        for j in T.thread_binding(0, 700, thread="blockIdx.x"):
            for i in T.thread_binding(0, 751, thread="threadIdx.x"):
                for k in range(50):
                    with T.block("output"):
                        C[j, i] += B[j, k] * A[j, i + k]


# We can verify that the two functions are the same:

ret_tvm = torch.zeros(700, 800 - 50 + 1).cuda()
tvm_depthwise(inputs, filters, ret_tvm)

testing.assert_allclose(ret_torch.cpu().numpy(), ret_tvm.cpu().numpy(), atol=1e-5, rtol=1e-5)

# Tip: We also provide an optional method `tune(config, target)` for additional optimization.
# In this case, users could call `tvm_depthwise.tune(target="nvidia/geforce-rtx-3070")`
# for trying to tune the operators via TVM MetaSchedule.

######################################################################
# Benchmark
# -------------------------------
# We will compare two operators by using PyTorch's benchmark toolkit.

results = []
for i in range(5):
    inputs = torch.randn(700, 800).cuda()
    filters = torch.randn(700, 50).cuda()
    res = torch.zeros(700, 800 - 50 + 1).cuda()
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

# In the working machine, the average inference time of `tvm_depthwise` is 42.5 us,
# while the average inference time of `torch_depthwise` is 66.0 us,
# showing the performance arises by around 1/3.
