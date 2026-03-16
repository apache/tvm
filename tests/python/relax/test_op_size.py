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
from tvm import relax
from tvm.script import relax as R


def test_op_size():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((), "int64"):
            return R.size(x)

    x_np = np.random.rand(2, 3).astype("float32")
    x = tvm.runtime.tensor(x_np)

    target = tvm.target.Target("llvm")
    ex = relax.build(Module, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    res = vm["main"](x)
    assert res.numpy() == 6


def test_op_size_dynamic():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor((), "int64"):
            return R.size(x)

    x_np = np.random.rand(4, 5).astype("float32")
    x = tvm.runtime.tensor(x_np)

    target = tvm.target.Target("llvm")
    ex = relax.build(Module, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    res = vm["main"](x)
    assert res.numpy() == 20


if __name__ == "__main__":
    tvm.testing.main()
