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

"""Configuration objects for the XNNPACK Relax backend."""

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_PRECISIONS = ("fp32", "fp16_hint", "fp16_force")
SUPPORTED_PARTITION_POLICIES = ("greedy", "cost", "debug_all_supported")
SUPPORTED_LAYOUT_POLICIES = ("auto", "NHWC", "preserve")
SUPPORTED_DYNAMIC_SHAPE_POLICIES = ("none", "batch_only")


@dataclass
class XNNPACKRuntimeConfig:
    """Runtime options serialized into XNNPACK external modules."""

    precision: str = "fp32"
    use_weights_cache: bool = False
    use_workspace: bool = False
    profile: bool = False
    dont_spin_workers: bool = False
    transient_indirection_buffer: bool = False
    num_threads: int = 1

    def validate(self) -> None:
        if self.precision not in SUPPORTED_PRECISIONS:
            raise ValueError(
                "Unsupported XNNPACK precision. Expected one of "
                f"{SUPPORTED_PRECISIONS}, but got {self.precision!r}."
            )
        if self.num_threads < 1:
            raise ValueError("XNNPACK num_threads must be at least 1.")

    def run_codegen_options(self) -> dict[str, Any]:
        self.validate()
        return {
            "precision": self.precision,
            "use_weights_cache": self.use_weights_cache,
            "use_workspace": self.use_workspace,
            "profile": self.profile,
            "dont_spin_workers": self.dont_spin_workers,
            "transient_indirection_buffer": self.transient_indirection_buffer,
            "num_threads": self.num_threads,
        }


@dataclass
class XNNPACKCostConfig:
    """Partition policy and reporting options."""

    partition_policy: str = "greedy"
    layout: str = "auto"
    min_subgraph_size: int = 2
    min_compute_to_copy_ratio: float = 8.0
    allow_isolated_elementwise: bool = False
    allow_layout_rewrite: bool = True
    allow_cast_boundary: bool = False
    report_partition_decisions: bool = False

    def validate(self) -> None:
        if self.partition_policy not in SUPPORTED_PARTITION_POLICIES:
            raise ValueError(
                "Unsupported XNNPACK partition_policy. Expected one of "
                f"{SUPPORTED_PARTITION_POLICIES}, but got {self.partition_policy!r}."
            )
        if self.layout not in SUPPORTED_LAYOUT_POLICIES:
            raise ValueError(
                "Unsupported XNNPACK layout policy. Expected one of "
                f"{SUPPORTED_LAYOUT_POLICIES}, but got {self.layout!r}."
            )
        if self.min_subgraph_size < 1:
            raise ValueError("min_subgraph_size must be at least 1.")
        if self.min_compute_to_copy_ratio < 0:
            raise ValueError("min_compute_to_copy_ratio must be non-negative.")


@dataclass
class XNNPACKPartitionConfig:
    """Options for Relax BYOC partitioning into XNNPACK regions."""

    runtime: XNNPACKRuntimeConfig = field(default_factory=XNNPACKRuntimeConfig)
    cost: XNNPACKCostConfig = field(default_factory=XNNPACKCostConfig)
    dynamic_shape_policy: str = "none"
    dynamic_batch_bounds: dict[str, int | tuple[int, int] | list[int]] | None = None

    def validate(self) -> None:
        self.runtime.validate()
        self.cost.validate()
        if self.dynamic_shape_policy not in SUPPORTED_DYNAMIC_SHAPE_POLICIES:
            raise ValueError(
                "Unsupported XNNPACK dynamic_shape_policy. Expected one of "
                f"{SUPPORTED_DYNAMIC_SHAPE_POLICIES}, but got {self.dynamic_shape_policy!r}."
            )
