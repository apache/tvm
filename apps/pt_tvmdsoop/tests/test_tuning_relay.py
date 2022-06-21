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
import tvm
import torch
from tvm.contrib.torch import optimize_torch
from tvm.meta_schedule import TuneConfig
import tvm.testing



def matmul(x, w):
    return torch.matmul(x, w)


def test_matmul_tuning_relay():
    config = TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=4,
                max_trials_per_task=4,
                max_trials_global=4,
                search_strategy_config={
                    "genetic_num_iters": 10,
                },
            )
    x = torch.randn(15, 20)
    w = torch.randn(20, 30)
    example_inputs = (x, w)
    
    rt_mod = optimize_torch(matmul, example_inputs, config)

    torch_answer = torch.matmul(x, w).numpy()
    tvm_answer = rt_mod(x, w).numpy()
    
    tvm.testing.assert_allclose(torch_answer, tvm_answer, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_matmul_tuning_relay()
    