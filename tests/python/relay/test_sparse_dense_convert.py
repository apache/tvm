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


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype="float32"):
    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r:r+BS_R,c:c+BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.data.size >= nnz
    assert s.indices.shape == (num_blocks, )
    assert s.indptr.shape == (M // BS_R + 1, )
    return s

def run_func(func, params, x):
    with relay.build_config(opt_level=3):
        graph, lib, new_params = relay.build(func, "llvm", params=params)

    from tvm.contrib import graph_runtime
    ctx = tvm.cpu(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.set_input(**new_params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    return tvm_output.asnumpy()

def test_bsr_sparse_dense():
    data = relay.var("data", shape=(1, 128), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(768, 128), dtype="float32")
    y = relay.nn.dense(x, w)
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)

    params = {
        "weight": tvm.nd.array(random_bsr_matrix(768, 128, 32, 1, 0.1).todense())
    }

    x_np = np.random.randn(1, 128).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    sparse_func, params = relay.data_dep_optimization.bsr_dense.convert(func, params, (32, 1), 0.2)
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_bsr_sparse_dense()
