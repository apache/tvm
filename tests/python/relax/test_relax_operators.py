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

import sys
import tempfile

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm._ffi.base import TVMError
from tvm.script import relax as R


def run_cpu(mod, func_name, *input):
    target = tvm.target.Target("llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*input)


@tvm.script.ir_module
class ShapeOfTest:
    @R.function
    def get_shape(t: R.Tensor(ndim=-1, dtype="int32")) -> R.Shape(ndim=-1):
        return R.shape_of(t)

    @R.function
    def get_shape_const() -> R.Shape(ndim=-1):
        x: R.Tensor((), "int32") = R.const(1, dtype="int32")
        return R.shape_of(x)


def test_op_shape_of():
    const_shape = run_cpu(ShapeOfTest, "get_shape_const")
    assert const_shape == tvm.runtime.ShapeTuple([])

    scalar_shape = run_cpu(ShapeOfTest, "get_shape", tvm.nd.array(np.array(1, dtype="int32")))
    assert scalar_shape == tvm.runtime.ShapeTuple([])

    tensor_shape = run_cpu(
        ShapeOfTest, "get_shape", tvm.nd.array(np.zeros((1, 2, 3)).astype("int32"))
    )
    assert tensor_shape == tvm.runtime.ShapeTuple([1, 2, 3])


if __name__ == "__main__":
    tvm.testing.main()
