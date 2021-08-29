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
from io import StringIO
import csv

import tvm.testing
from tvm.runtime import profiler_vm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.contrib.debugger import debug_executor


@pytest.mark.skipif(not profiler_vm.enabled(), reason="VM Profiler not enabled")
@tvm.testing.parametrize_targets
def test_vm(target, dev):
    mod, params = mlp.get_workload(1)

    exe = relay.vm.compile(mod, target, params=params)
    vm = profiler_vm.VirtualMachineProfiler(exe, dev)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = vm.profile(data, func_name="main")
    assert "fused_nn_softmax" in str(report)
    assert "Total" in str(report)

    f = StringIO(report.csv())
    reader = csv.reader(f, delimiter=",")
    # force parsing
    in_header = True
    for row in reader:
        if in_header:
            assert "Hash" in row
            in_header = False


@tvm.testing.parametrize_targets
def test_graph_executor(target, dev):
    mod, params = mlp.get_workload(1)

    exe = relay.build(mod, target, params=params)
    gr = debug_executor.create(exe.get_graph_json(), exe.lib, dev)

    data = np.random.rand(1, 1, 28, 28).astype("float32")
    report = gr.profile(data=data)
    assert "fused_nn_softmax" in str(report)
    assert "Total" in str(report)
    assert "Hash" in str(report)
