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
"""Common utility for topi test"""

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import FallbackConfigEntity
import topi

def get_all_backend():
    """return all supported target

    Returns
    -------
    targets: list
        A list of all supported targets
    """
    return ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan', 'nvptx',
            'llvm -device=arm_cpu', 'opencl -device=mali', 'aocl_sw_emu']

class Int8Fallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        self.memory[key] = cfg
        cfg.is_fallback = False
        return cfg
