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
Helper functions for running TorchBench through the benchmark functions
from TorchDynamo.
"""

import os
import sys
from dataclasses import dataclass

import torch  # type: ignore


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
from benchmarks.common import same, timed  # type: ignore
from torchbench import TorchBenchmarkRunner  # type: ignore

# pylint: disable=wrong-import-position, unused-import


def load_torchdynamo_benchmark_runner(
    is_cuda: bool, cosine_similarity: bool = False, float32: bool = False
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

    if is_cuda:
        # pylint: disable=import-outside-toplevel
        import benchmarks.common  # type: ignore

        # pylint: enable=import-outside-toplevel

        benchmarks.common.synchronize = torch.cuda.synchronize

    return runner
