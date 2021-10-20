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

import itertools

import numpy as np
import scipy.sparse as sp


import tvm
from tvm.ir import IRModule
from tvm import relay
from tvm.relay.data_dep_optimization import simplify_fc_transpose


def run_func(func, params, x):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(func, "llvm", params=params)

    from tvm.contrib import graph_executor

    dev = tvm.cpu(0)
    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    # set inputs
    m.set_input("data", tvm.nd.array(x.astype(dtype)))
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    return tvm_output.numpy()


def test_simplify_fc_transpose():
    data = relay.var("data", shape=(1, 32), dtype="float32")
    x = relay.nn.relu(data)
    w1 = relay.var("w1", shape=(32, 64), dtype="float32")
    y = relay.nn.dense(x, relay.transpose(w1, axes=[1, 0]))
    z = relay.nn.relu(y)
    w2 = relay.var("w2", shape=(64, 16), dtype="float32")
    zz = relay.nn.dense(z, relay.transpose(w2, axes=[1, 0]))
    func = relay.Function(relay.analysis.free_vars(zz), zz)
    params = {
        "w1": tvm.nd.array(np.random.uniform(-1, 1, (32, 64)).astype("float32")),
        "w2": tvm.nd.array(np.random.uniform(-1, 1, (64, 16)).astype("float32")),
    }
    x_np = np.random.randn(1, 32).astype("float32")
    old_result = run_func(func, params, x_np)

    new_func, new_params = simplify_fc_transpose.convert(func, params)
    new_result = run_func(new_func, new_params, x_np)
    np.testing.assert_allclose(old_result, new_result, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_simplify_fc_transpose()
