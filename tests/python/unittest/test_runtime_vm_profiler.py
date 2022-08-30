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
import tvm.testing
from tvm.runtime import profiler_vm
from tvm import relay
from tvm.relay.testing import mlp


@tvm.testing.parametrize_targets
def test_basic(dev, target):
    mod, params = mlp.get_workload(batch_size=1)
    if not profiler_vm.enabled():
        return

    exe = relay.vm.compile(mod, target, params=params)
    code, lib = exe.save()
    des_exe = tvm.runtime.vm.Executable.load_exec(code, lib)
    vm = profiler_vm.VirtualMachineProfiler(des_exe, dev)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    res = vm.profile(tvm.nd.array(data), func_name="main")
    assert "softmax" in str(res)


def test_vm_reshape_and_copy():
    target = "llvm"
    dev = tvm.gpu()
    x_np = np.random.uniform(size=(8, 16)).astype("float32")
    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.reshape(x, [-1, 4, 8])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert "reshape_tensor" in exec.bytecode
    vm = profiler_vm.VirtualMachineProfiler(exec, dev)
    vm.profile(tvm.nd.array(x_np))


if __name__ == "__main__":
    tvm.testing.main()
