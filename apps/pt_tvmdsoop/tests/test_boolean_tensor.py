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
"""Test script for boolean tensor support"""
import numpy as np

import torch

import tvm
import tvm.testing
from tvm.contrib.torch import optimize_torch


def negate(x):
    return x.logical_not()


def sum_up_tensor(x, y):
    return torch.sum(x[y])


def test_bool_tensor_negate():
    input = torch.ones(1, dtype=torch.bool)
    optimized_negate = optimize_torch(
        negate,
        input,
    )
    output = optimized_negate(negate(input))
    tvm.testing.assert_allclose(input.numpy(), output.numpy(), atol=1e-5, rtol=1e-5)


def test_sum_up_tensor():
    x = torch.randint(0, 2, (8,))
    y = x.bool()
    optimized_func = optimize_torch(sum_up_tensor, (x, y))
    ret1 = torch.sum(x).numpy()
    ret2 = optimized_func(x, y).numpy()
    tvm.testing.assert_allclose(ret1, ret2, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_bool_tensor_negate()
    test_sum_up_tensor()
