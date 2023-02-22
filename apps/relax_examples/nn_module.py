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

# Example code on creating, compiling, and running a neural network with pytorch-like API


import tvm
from tvm.relay import Call
from tvm import relax, tir
from tvm.relax.testing import nn
from tvm.script import relax as R
import numpy as np


if __name__ == "__main__":
    builder = relax.BlockBuilder()

    # a symbolic variable to represent minibatch size
    n = tir.Var("n", "int64")
    input_size = 784
    hidden_sizes = [128, 32]
    output_size = 10

    # build a three linear-layer neural network for a classification task
    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
            nn.LogSoftmax(),
        )
        data = nn.Placeholder((n, input_size), name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    # get and print the IRmodule being built
    mod = builder.get()
    mod.show()

    # build the IRModule and create relax vm
    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    # init parameters
    params = nn.init_params(mod)

    # run the model on relax vm
    # the input data has a minibatch size of 3
    data = tvm.nd.array(np.random.rand(3, input_size).astype(np.float32))
    res = vm["main"](data, *params)
    print(res)
