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
# pylint: disable=invalid-name, pointless-exception-statement
"""
Helper functions for running TorchBench through the benchmark functions
from TorchDynamo.
"""

import functools
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Set

import torch  # type: ignore


class DisallowedOperator(Enum):
    """
    The operators to disallow in the fx graph produced by TorchDynamo.
    This is to workaround the limitation in TVM's PyTorch frontend.

    - inplace_copy: aten::copy_ as inplace assign A[...] = ..., or method call A.copy_(...)
    - einsum: torch.functional.einsum
    - multihead_attention: torch.nn.MultiheadAttention
    - as_stride: Tensor.as_stride
    """

    INPLACE_COPY = "inplace_copy"
    EINSUM = "einsum"
    MULTIHEAD_ATTENTION = "multihead_attention"
    AS_STRIDE = "as_stride"


def find_torchdynamo() -> str:
    """
    Find the directory of TorchDynamo repo.

    It can't directly import the benchmark runner in TorchDynamo
    becuase it isn't designed to be used as a Python package.
    """
    candidates = [
        "torchdynamo",
        "../torchdynamo",
        "../../torchdynamo",
    ]
    for library_dir in candidates:
        if os.path.exists(f"{library_dir}/benchmarks"):
            return library_dir

    raise RuntimeError(
        """
        Cannot find directory for torchdynamo.
        You need to clone https://github.com/pytorch/torchdynamo to the parent directory of cwd.
        """
    )


DYNAMO_DIR = find_torchdynamo()
sys.path.insert(
    0, DYNAMO_DIR
)  # opacus_cifar10 depends on opacus, which installs a package called 'benchmarks'
sys.path.append(f"{DYNAMO_DIR}/benchmarks")

# pylint: disable=wrong-import-position, unused-import
import torchdynamo  # type: ignore
from benchmarks.common import same, timed  # type: ignore
from torchbench import TorchBenchmarkRunner  # type: ignore

# pylint: disable=wrong-import-position, unused-import


def _disallow_operators(disallowed_ops: Set[DisallowedOperator]):
    """
    Disallow certain operators in the fx graph produced by TorchDynamo.
    There are two ways to disallow operator in TorchDynamo,
    1. Use the disallow_in_graph API, which only applies to free function call.
    2. Patch the TensorVariable class, which applies to method call on torch.Tensor.
    """
    disallowed_tensor_methods: Set[str] = set()

    if DisallowedOperator.INPLACE_COPY in disallowed_ops:
        torchdynamo.disallow_in_graph(torch.Tensor.copy_)
        disallowed_tensor_methods.update({"copy_", "__setitem__"})

    if DisallowedOperator.EINSUM in disallowed_ops:
        torchdynamo.disallow_in_graph(torch.functional.einsum)

    if DisallowedOperator.MULTIHEAD_ATTENTION in disallowed_ops:
        torchdynamo.disallow_in_graph(torch.nn.MultiheadAttention)

    if DisallowedOperator.AS_STRIDE in disallowed_ops:
        disallowed_tensor_methods.add("as_stride")

    tensor_variable_cls = torchdynamo.variables.tensor.TensorVariable
    old_call_method = tensor_variable_cls.call_method

    @functools.wraps(old_call_method)
    def call_method(self, translator, name, args, kwargs):
        if name in disallowed_tensor_methods:
            raise torchdynamo.exc.Unsupported(f"Tensor.{name} not supported by TVM.")
        return old_call_method(self, translator, name, args, kwargs)

    tensor_variable_cls.call_method = call_method


def load_torchdynamo_benchmark_runner(
    is_cuda: bool,
    cosine_similarity: bool = False,
    float32: bool = False,
    disallowed_operators: Set[DisallowedOperator] = None,
) -> TorchBenchmarkRunner:
    """
    Load the benchmark runner from TorchDynamo.
    """

    @dataclass
    class RunnerArgs:
        """
        This class simulates the parsed args required by the benchmark code from TorchDynamo.
        """

        ci: bool = False  # Whether runs in CI mode. pylint: disable=invalid-name
        training: bool = False  # Whether it benchmarks training workload.
        use_eval_mode: bool = True  # Whether the model should be in eval mode.
        dynamic_shapes: bool = False  # Whether runs the model in dynamic shape mode.
        float16: bool = False  # Whether to cast model and inputs to float16
        float32: bool = False  # Whether to cast model and inputs to float32

        accuracy: bool = False  # Whether to perform a accuracy test
        performance: bool = True  # Whether to perform a performance test

        cosine: bool = False  # Whether to use consine similarity to check if output is correct.

    args = RunnerArgs(cosine=cosine_similarity, float32=float32)

    runner = TorchBenchmarkRunner()
    runner.args = args
    runner.model_iter_fn = runner.forward_pass

    if disallowed_operators:
        _disallow_operators(disallowed_operators)

    if is_cuda:
        # pylint: disable=import-outside-toplevel
        import benchmarks.common  # type: ignore

        # pylint: enable=import-outside-toplevel

        benchmarks.common.synchronize = torch.cuda.synchronize

    return runner
