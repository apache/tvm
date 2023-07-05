# pylint: disable=missing-class-docstring
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
"""Test script for tvm torch module"""
import tempfile

import torch
from torch.utils import benchmark
from torchvision.models import resnet18

import tvm
import tvm.testing
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig


def test_matmul_tuning_relay():
    def matmul(x, w):
        return torch.matmul(x, w)

    x = torch.randn(15, 20)
    w = torch.randn(20, 30)
    example_inputs = (x, w)

    rt_mod = optimize_torch(matmul, example_inputs)
    torch_answer = torch.matmul(x, w).numpy()
    tvm_answer = rt_mod(x, w).numpy()

    tvm.testing.assert_allclose(torch_answer, tvm_answer, atol=1e-5, rtol=1e-5)


class InnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 20, 5)

    def forward(self, x):
        return torch.nn.functional.relu(self.conv(x))


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(20, 20, 5)
        self.relu = InnerModel()

    def forward(self, x):
        x = self.relu(x)
        return torch.nn.functional.relu(self.conv(x))


def test_nested_module():
    simple_module = SimpleModel()
    example_input = torch.randn(20, 1, 10, 10)
    optimized_module = optimize_torch(simple_module, example_input)
    ret1 = simple_module(example_input).detach().numpy()
    ret2 = optimized_module(example_input).detach().numpy()
    tvm.testing.assert_allclose(ret1, ret2, atol=1e-5, rtol=1e-5)


def test_save_load_function():
    def foo(x):
        return 2 * x + 1

    example_input = torch.rand(3)
    opt_foo = optimize_torch(foo, example_input)
    ret1 = opt_foo(example_input)
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save(opt_foo, tmp.name)
        loaded_mod = torch.load(tmp.name)
        ret2 = loaded_mod(example_input)
    tvm.testing.assert_allclose(ret1.numpy(), ret2.numpy(), atol=1e-5, rtol=1e-5)


class MyResNet18(torch.nn.Module):
    def __init__(self, config, target=None):
        super(MyResNet18, self).__init__()
        self.means = torch.nn.Parameter(
            torch.tensor([103.939, 116.779, 123.68]).resize_(1, 3, 1, 1)
        ).cuda()
        self.resnet = optimize_torch(resnet18(), [torch.rand(1, 3, 224, 224)], config, target)

    def forward(self, input):
        return self.resnet(input - self.means)


class JitModule(torch.nn.Module):
    def __init__(self):
        super(JitModule, self).__init__()
        self.means = torch.nn.Parameter(
            torch.tensor([103.939, 116.779, 123.68]).resize_(1, 3, 1, 1)
        ).cuda()
        self.resnet = torch.jit.optimize_for_inference(torch.jit.script(resnet18().cuda().eval()))

    def forward(self, input):
        return self.resnet(input - self.means)


# default config for testing
config = TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=4,
    max_trials_per_task=8,
    max_trials_global=16,
)

if torch.cuda.is_available():
    target_cuda = "nvidia/geforce-rtx-3070"
    meta_module_resnet18 = MyResNet18(config, target_cuda)
    jit_module_resnet18 = JitModule()


def compare_optimize_resnet18_to_torchscript():
    results = []
    for i in range(20):
        test_input = torch.rand(1, 3, 224, 224).half().cuda()
        sub_label = f"[test {i}]"
        results.append(
            benchmark.Timer(
                stmt="meta_module_resnet18(test_input)",
                setup="from __main__ import meta_module_resnet18",
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


if __name__ == "__main__":
    test_matmul_tuning_relay()
    test_nested_module()
    test_save_load_function()
    if torch.cuda.is_available():
        compare_optimize_resnet18_to_torchscript()
