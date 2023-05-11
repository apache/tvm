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
"""Test flexible shape dispatch pass"""
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.testing.resnet import get_workload
from tvm.relay import vm
from tvm import runtime


def test_end_to_end():
    # Load a resnet model.
    mod, params = get_workload()
    # Apply flexible dispatch pass.
    mod = relay.transform.FlexibleShapeDispatch(axis=0, buckets=[1, 4], auto_pad=True)(mod)
    # Compile and confirm result supports multiple shapes.
    exe = relay.vm.compile(mod, "llvm", params=params)
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())

    # Evaluate various batch sizes
    batch_1 = np.random.normal(size=[1, 3, 224, 224]).astype("float32")
    assert list(vm.invoke("main", batch_1).shape) == [1, 1000]

    batch_4 = np.random.normal(size=[4, 3, 224, 224]).astype("float32")
    assert list(vm.invoke("main", batch_4).shape) == [4, 1000]

    # Apply autopadding to an input.
    batch_3 = np.random.normal(size=[3, 3, 224, 224]).astype("float32")
    assert list(vm.invoke("main", batch_3).shape) == [3, 1000]


def test_multiple_inputs():
    # Create a small relay module with multiple inputs to dispatch over.
    x = relay.var("x", shape=[10, 10], dtype="float32")
    w = relay.var("w", shape=[10, 10], dtype="float32")
    y = x + w
    mod = tvm.IRModule.from_expr(y)

    # Apply flexible dispatch to dim 1 for both inputs.
    mod = relay.transform.FlexibleShapeDispatch(axis=1, buckets=[5, 10], input_indices=[0, 1])(mod)

    # Compile and confirm that output shapes are correct.
    exe = relay.vm.compile(mod, "llvm")
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())

    x_w_5 = np.random.normal(size=[10, 5]).astype("float32")
    assert list(vm.invoke("main", x_w_5, x_w_5).shape) == [10, 5]

    x_w_10 = np.random.normal(size=[10, 10]).astype("float32")
    assert list(vm.invoke("main", x_w_10, x_w_10).shape) == [10, 10]


def test_fixed_output():
    # Test a graph where the output shape is not based on input dynamism.
    x = relay.var("x", shape=[10, 10], dtype="float32")
    w = relay.var("w", shape=[10, 10], dtype="float32")
    y = relay.nn.dense(x, w)
    mod = tvm.IRModule.from_expr(y)

    # Apply flexible dispatch to dimension 1 for both inputs.
    mod = relay.transform.FlexibleShapeDispatch(
        axis=1, buckets=[5, 7], input_indices=[0, 1], affects_output=False
    )(mod)

    # Compile and confirm that output shapes are correct.
    exe = relay.vm.compile(mod, "llvm")
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())

    x_w_5 = np.random.normal(size=[10, 5]).astype("float32")
    assert list(vm.invoke("main", x_w_5, x_w_5).shape) == [10, 10]

    x_w_7 = np.random.normal(size=[10, 7]).astype("float32")
    assert list(vm.invoke("main", x_w_7, x_w_7).shape) == [10, 10]

    return


def test_multiple_outputs():
    # Create a graph with multiple outputs and test that it works.
    x = relay.var("x", shape=[10, 10], dtype="float32")
    y = relay.split(x, 2, axis=1)
    mod = tvm.IRModule.from_expr(y.astuple())

    # Apply flexible dispatch to batch dimension.
    mod = relay.transform.FlexibleShapeDispatch(axis=0, buckets=[5, 10])(mod)

    # Compile and confirm that both outputs are correct.
    exe = relay.vm.compile(mod, "llvm")
    vm = runtime.vm.VirtualMachine(exe, tvm.cpu())

    x_5 = np.random.normal(size=[5, 10]).astype("float32")
    result_5 = vm.invoke("main", x_5)
    assert list(result_5[0].shape) == [5, 5]
    assert list(result_5[1].shape) == [5, 5]

    x_10 = np.random.normal(size=[10, 10]).astype("float32")
    result_10 = vm.invoke("main", x_10)
    assert list(result_10[0].shape) == [10, 5]
    assert list(result_10[1].shape) == [10, 5]


if __name__ == "__main__":
    tvm.testing.main()
