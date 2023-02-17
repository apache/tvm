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

# Example code on creating, compiling, and running an MLP model in relax


import tvm
from tvm import relax, tir, topi
import numpy as np


def build_mlp(data, weight):
    bb = relax.BlockBuilder()

    with bb.function("mlp", [data, weight]):
        gv0 = bb.emit_te(tvm.contrib.cblas.matmul, data, weight, transa=False, transb=False)
        gv1 = bb.emit_te(topi.nn.relu, gv0)
        bb.emit_func_output(gv1)

    mod = bb.get()
    return mod


if __name__ == "__main__":
    # symbolic dimensions
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    # create data and weight variables
    data = relax.Var("data", relax.TensorStructInfo([n, m], "float32"))
    weight = relax.Var("weight", relax.TensorStructInfo([m, n], "float32"))

    # construct a mlp model
    mod = build_mlp(data, weight)

    # build and create vm executor
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.vm.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # run the mlp model on relax vm
    data = tvm.nd.array(np.random.rand(16, 32).astype(np.float32))
    weight = tvm.nd.array(np.random.rand(32, 16).astype(np.float32))
    res = vm["mlp"](data, weight)
    print(res)
