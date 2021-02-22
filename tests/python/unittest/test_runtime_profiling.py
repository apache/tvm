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
import pytest

import tvm.testing
from tvm.runtime import profiler_vm
from tvm.contrib import graph_runtime
from tvm import relay
from tvm.relay.testing import mlp
from tvm.contrib.debugger import debug_runtime


@pytest.mark.skipif(not profiler_vm.enabled(), reason="VM Profiler not enabled")
@tvm.testing.parametrize_targets
def test_vm(target, ctx):
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, target, params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, ctx)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = vm.profile("main", [data])
    assert "fused_nn_softmax" in report
    assert "Total time" in report


@tvm.testing.parametrize_targets
def test_graph_runtime(target, ctx):
    mod, params = mlp.get_workload(1)

    exe = relay.build(mod, target, params=params)
    gr = debug_runtime.create(exe.get_json(), exe.lib, ctx)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = gr.profile(data=data)
    assert "fused_nn_softmax" in report
    assert "Total time" in report
