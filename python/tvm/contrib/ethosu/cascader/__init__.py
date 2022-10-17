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
"""The NPU cascader.

This component performs inter-operator scheduling to optimize
for both performance and memory usage on Arm(R) Ethos(TM)-U NPUs.
"""
from .stripe_config import StripeConfig
from .block_config import BlockConfig
from .propagator import Propagator
from .graph import (
    PerformanceInfo,
    Tensor,
    Part,
    TESubgraph,
    CascaderGraph,
    BufferMode,
    register_matcher,
    create_cascader_graph,
)
from .parts import InlinePart, EthosuPart
from .device_config import EthosuDeviceConfig
from .tensor_config import TensorConfigState, MemoryRegion, TensorConfig
from .plan import Plan
from .scheduler import apply_proposal, cascade, extract_memory_info
from .logging import Logging
from .cascader_options import CascaderOptions
