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

from tvm import relax, rpc
from tvm.contrib import utils
from tvm.relax.testing import nn
from tvm.script import relax as R


def get_exec(data_shape):
    builder = relax.BlockBuilder()
    weight1_np = np.random.randn(64, 64).astype("float32")
    weight2_np = np.random.randn(64, 64).astype("float32")

    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(data_shape[1], weight1_np.shape[0], bias=False),
            nn.ReLU(),
            nn.Linear(weight2_np.shape[0], weight2_np.shape[1], bias=False),
            nn.ReLU(),
        )
        data = nn.Placeholder(data_shape, name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    mod = builder.get()

    params = {"linear_weight": weight1_np, "linear_weight1": weight2_np}
    mod = relax.transform.BindParams("main", params)(mod)

    target = "llvm"
    return relax.build(mod, target)


def test_conv2d_cpu():
    data_np = np.random.randn(1, 64).astype("float32")
    ex = get_exec(data_np.shape)

    vm = relax.VirtualMachine(ex, tvm.cpu(), profile=True)
    report = vm.profile("main", tvm.nd.array(data_np))
    print(report)

    assert "Duration" in str(report)
    assert "matmul" in str(report)


def with_rpc(ex, f, data_np):
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    ex.export_library(path)

    server = rpc.Server("127.0.0.1")
    remote = rpc.connect(server.host, server.port, session_timeout=10)

    remote.upload(path)
    rexec = remote.load_module("vm_library.so")

    device = remote.cpu()

    vm = relax.VirtualMachine(rexec, device=device, profile=True)
    data = tvm.nd.array(data_np, device)

    f(vm, data)


def test_rpc():
    data_np = np.random.randn(1, 64).astype("float32")
    ex = get_exec(data_np.shape)

    def callback(vm, data):
        vm.profile("main", data)

        vm.set_input("main", data)
        report = vm.profile("main")

        assert "matmul" in str(report)
        print(report)

    with_rpc(ex, callback, data_np)


def test_tuple():
    @tvm.script.ir_module
    class NestedTuple:
        @R.function
        def main(
            x: R.Tensor((16,), "float32")
        ) -> R.Tuple(
            R.Tuple(
                R.Tensor((16,), "float32"),
                R.Tuple(
                    R.Tensor((16,), "float32"),
                ),
            ),
            R.Tensor((16,), "float32"),
        ):
            return ((x, (x,)), x)

    target = "llvm"
    ex = relax.build(NestedTuple, target)

    data_np = np.random.randn(16).astype("float32")

    def callback(vm, data):
        report = vm.profile("main", data)
        assert "vm.builtin.make_tuple" in str(report)

    with_rpc(ex, callback, data_np)


if __name__ == "__main__":
    tvm.testing.main()
