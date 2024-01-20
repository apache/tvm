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
import tvm
from tvm.target import Target
from .arch_base import Arch
from typing import List, Dict

def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1

class TensorInstruction(object):
    def __init__(
        self,
        name: str,
        intrin_group: Dict,
        shape: List[int],
    ):
        self.name: str = name
        self.intrin_group: Dict = intrin_group
        # only mantain the shape of M and N
        self.shape: List[int] = shape

class CUDA(Arch):
    def __init__(self, target: Target):
        self.target = target
        self.sm_version = check_sm_version(self.target.arch)
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        self.device: tvm.runtime.Device = device
        self.platform: str = "CUDA"
        self.smem_cap = device.max_shared_memory_per_block
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        self.reg_cap: int = 65536
        self.max_smem_usage: int = 2 * self.smem_cap
        self.sm_partition: int = 4
        self.l2_cache_size_bytes: int = target.l2_cache_size_bytes
        # the number of transaction size in bytes
        self.transaction_size: List[int] = [32, 128]  # in bytes
        # bandwidth in MB/s, will be used for recommend basic tile size
        # TODO(lei): find some way to get the real bandwidth
        # However, the ratio of bandwidth between different devices can
        # be similar. The bandwidth can work for another devices as well.
        self.bandwidth: List[int] = [750, 12080]
        # the tensor instruction informations
        
        from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group, get_mma_intrin_group

        self.available_tensor_instructions = (
            TensorInstruction("mma", get_mma_intrin_group, [16, 16]),
            TensorInstruction("wmma", get_wmma_intrin_group, [16, 16]),
        )

    def get_avaliable_tensorintrin_shapes(self):
        return [t.shape for t in self.available_tensor_instructions]