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
from tvm.topi.sparse.utils import random_bsr_matrix
from tvm.relay.build_module import bind_params_by_name


def run_func(func, params, x):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, new_params = relay.build(func, "llvm", params=params)

    from tvm.contrib import graph_executor

    dev = tvm.cpu(0)
    dtype = "float32"
    m = graph_executor.create(graph, lib, dev)
    # set inputs
    m.set_input("data", tvm.nd.array(x.astype(dtype)))
    m.set_input(**new_params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)
    return tvm_output.numpy()


def test_bsr_sparse_conv2d_nchw():
    data = relay.var("data", shape=(1, 64, 32, 32), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(128, 64, 1, 1), dtype="float32")
    y = relay.nn.conv2d(x, w, channels=128, kernel_size=1, data_layout="NCHW", kernel_layout="OIHW")
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)

    params = {
        "weight": tvm.nd.array(
            np.array(random_bsr_matrix(128, 64, 8, 1, 0.1, "float32").todense()).reshape(
                128, 64, 1, 1
            )
        )
    }

    x_np = np.random.randn(1, 64, 32, 32).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert(
        func, params, (8, 1), 0.2, "NCHW"
    )
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)


def test_bsr_sparse_conv2d_nhwc():
    data = relay.var("data", shape=(1, 32, 32, 64), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(1, 1, 64, 128), dtype="float32")
    y = relay.nn.conv2d(x, w, channels=128, kernel_size=1, data_layout="NHWC", kernel_layout="HWIO")
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)

    params = {
        "weight": tvm.nd.array(
            np.array(random_bsr_matrix(128, 64, 8, 1, 0.1, "float32").todense()).T.reshape(
                1, 1, 64, 128
            )
        )
    }

    x_np = np.random.randn(1, 32, 32, 64).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert(
        func, params, (8, 1), 0.2, "NHWC"
    )
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)


def test_bsr_sparse_conv2d_3x3_nchw():
    data = relay.var("data", shape=(1, 64, 32, 32), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(128, 64, 3, 3), dtype="float32")
    y = relay.nn.conv2d(
        x, w, channels=128, kernel_size=3, padding=1, data_layout="NCHW", kernel_layout="OIHW"
    )
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)

    params = {
        "weight": tvm.nd.array(
            np.array(random_bsr_matrix(128, 64 * 9, 16, 1, 0.1, "float32").todense()).reshape(
                128, 64, 3, 3
            )
        )
    }

    x_np = np.random.randn(1, 64, 32, 32).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    func = bind_params_by_name(func, params)
    sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert2(
        func, {}, (16, 1), 0.2, "NCHW", 3
    )
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)


def test_bsr_sparse_conv2d_3x3_nhwc():
    data = relay.var("data", shape=(1, 32, 32, 64), dtype="float32")
    x = relay.nn.relu(data)
    w = relay.var("weight", shape=(3, 3, 64, 128), dtype="float32")
    y = relay.nn.conv2d(
        x, w, channels=128, kernel_size=3, padding=1, data_layout="NHWC", kernel_layout="HWIO"
    )
    z = relay.nn.relu(y)
    func = relay.Function(relay.analysis.free_vars(z), z)

    params = {
        "weight": tvm.nd.array(
            np.array(random_bsr_matrix(128, 64 * 9, 16, 1, 0.1, "float32").todense()).T.reshape(
                3, 3, 64, 128
            )
        )
    }

    x_np = np.random.randn(1, 32, 32, 64).astype("float32")
    # dense output
    dense_output = run_func(func, params, x_np)
    # sparse
    func = bind_params_by_name(func, params)
    sparse_func, params = relay.data_dep_optimization.bsr_conv2d.convert2(
        func, {}, (16, 1), 0.2, "NHWC", 3
    )
    sparse_output = run_func(sparse_func, params, x_np)
    np.testing.assert_allclose(sparse_output, dense_output, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_bsr_sparse_conv2d_nhwc()
    test_bsr_sparse_conv2d_nchw()
    test_bsr_sparse_conv2d_3x3_nhwc()
    test_bsr_sparse_conv2d_3x3_nchw()
