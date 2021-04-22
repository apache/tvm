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
import numpy as np

import tvm
from tvm.runtime import profiler_vm
from tvm import relay
from tvm.relay.testing import mlp


@tvm.testing.parametrize_targets
def test_basic(dev, target):
    mod, params = mlp.get_workload(batch_size=1)
    if not profiler_vm.enabled():
        return

    exe = relay.vm.compile(mod, target, params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    res = vm.profile(tvm.nd.array(data), func_name="main")
    assert "softmax" in str(res)


if __name__ == "__main__":
    test_basic(tvm.cpu(), tvm.target.Target("llvm"))
