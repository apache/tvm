#!/usr/bin/env python3
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
from tvm import relax, te
from tvm.runtime import Executable
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_compile_tir():
    """Test tvm.compile with TIR input."""
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    func = te.create_prim_func([A, B, C])

    # Test compile with PrimFunc
    exec_prim = tvm.compile(func)
    assert isinstance(exec_prim, Executable)

    # Test compile with IRModule containing PrimFunc
    mod = tvm.IRModule.from_expr(func)
    exec_mod = tvm.compile(mod)
    assert isinstance(exec_mod, Executable)

    # Verify the compiled module works
    dev = tvm.cpu(0)
    a_np = np.random.uniform(size=10).astype(np.float32)
    b_np = np.random.uniform(size=10).astype(np.float32)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros(10, dtype=np.float32), dev)

    exec_prim(a, b, c)
    np.testing.assert_allclose(c.numpy(), a_np + b_np)
    exec_mod(a, b, c)
    np.testing.assert_allclose(c.numpy(), a_np + b_np)


def test_compile_relax():
    """Test tvm.compile with Relax input."""

    # Define a simple Relax program
    @I.ir_module
    class MyModule:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")) -> R.Tensor:
            z = R.add(x, y)
            return z

    # Test compile with Relax module
    target = tvm.target.Target("llvm")
    exec_relax = tvm.compile(MyModule, target)
    assert isinstance(exec_relax, Executable)

    # Verify the compiled module works
    dev = tvm.cpu(0)
    x_np = np.random.uniform(size=(3, 4)).astype(np.float32)
    y_np = np.random.uniform(size=(3, 4)).astype(np.float32)
    x = tvm.nd.array(x_np, dev)
    y = tvm.nd.array(y_np, dev)

    vm = relax.VirtualMachine(exec_relax, dev)
    z = vm["main"](x, y)
    np.testing.assert_allclose(z.numpy(), x_np + y_np)


@tvm.testing.skip_if_32bit(reason="skipping test for i386.")
def test_compile_mixed_module():
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def add_one(X: T.Buffer((4,), "float32"), Y: T.Buffer((4,), "float32")):
            for i in range(4):
                Y[i] = X[i] + 1

        @R.function
        def main(x: R.Tensor((4,), "float32")):
            cls = MyModule
            with R.dataflow():
                y = R.call_tir(cls.add_one, [x], R.Tensor((4,), "float32"))
                return y

    # Test with custom pipeline
    target = tvm.target.Target("c")
    ex = tvm.compile(MyModule, target)
    assert isinstance(ex, Executable)

    dev = tvm.cpu(0)
    x = tvm.nd.array(np.array([1, 2, 3, 4], dtype=np.float32), dev)
    y = tvm.nd.array(np.zeros(4, dtype=np.float32), dev)
    # For tir function, we can directly call the function
    ex["add_one"](x, y)
    np.testing.assert_allclose(y.numpy(), x.numpy() + 1)
    # For relax function, we need to use the vm to call the function
    vm = relax.VirtualMachine(ex, dev)
    z = vm["main"](x)
    np.testing.assert_allclose(z.numpy(), x.numpy() + 1)


if __name__ == "__main__":
    tvm.testing.main()
