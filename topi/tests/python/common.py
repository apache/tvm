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

_injective_schedule = {
    "generic": topi.generic.schedule_injective,
    "cpu": topi.x86.schedule_injective,
    "arm_cpu": topi.arm_cpu.schedule_injective,
    "gpu": topi.cuda.schedule_injective,
    "hls": topi.hls.schedule_injective,
    "opengl": topi.opengl.schedule_injective
}

_reduce_schedule = {
    "generic": topi.generic.schedule_reduce,
    "cpu": topi.x86.schedule_reduce,
    "gpu": topi.cuda.schedule_reduce,
    "hls": topi.cuda.schedule_reduce
}

def get_schedule_injective(target):
    if isinstance(target, str):
        target = tvm.target.create(target)
    for key in target.keys:
        if key in _injective_schedule:
            return _injective_schedule[key]
    return _injective_schedule["generic"]

def get_schedule_reduce(target):
    if isinstance(target, str):
        target = tvm.target.create(target)
    for key in target.keys:
        if key in _reduce_schedule:
            return _reduce_schedule[key]
    return _reduce_schedule["generic"]

get_schedule_broadcast = get_schedule_injective
get_schedule_elemwise = get_schedule_injective

class Int8Fallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        self.memory[key] = cfg
        cfg.is_fallback = False
        return cfg
