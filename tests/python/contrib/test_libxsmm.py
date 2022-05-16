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
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.op.contrib import libxsmm
from tvm.relay.op.contrib.libxsmm import partition_for_libxsmm
from tvm.contrib import graph_runtime
from tvm import testing


def check_libxsmm_used(mod, libxsmm_enabled):
    num_libxsmm_subgraphs = sum(
        [1 if "libxsmm" in gv.name_hint else 0 for gv in mod.get_global_vars()]
    )
    assert num_libxsmm_subgraphs == 1 if libxsmm_enabled else num_libxsmm_subgraphs == 0


def run_and_verify(mod, input_dict, params, target, libxsmm_enabled=True):
    dev = tvm.cpu()
    result_dict = {}
    for r in ("origin", "libxsmm"):
        if r == "libxsmm":
            mod = partition_for_libxsmm(mod)
            check_libxsmm_used(mod, libxsmm_enabled)

        json, lib, param = relay.build(mod, target="llvm", params=params)
        runtime_module = tvm.contrib.graph_runtime.create(json, lib, device=dev)
        for k, v in input_dict.items():
            runtime_module.set_input(k, v)
        runtime_module.load_params(tvm.runtime.save_param_dict(param))
        runtime_module.run()
        result_dict[r] = runtime_module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(result_dict["origin"], result_dict["libxsmm"], rtol=1e-5, atol=1e-5)


def run_and_verify_func(config, dtype, libxsmm_enabled=True):
    f, input_shapes, param_list = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in param_list}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        for k, v in input_shapes.items()
        if k not in param_list
    }
    run_and_verify(f, input_dict, params, target="llvm", libxsmm_enabled=libxsmm_enabled)


def get_dense(x_shape, k_shape, dtype="float32"):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.dense(x, kernel, units=k_shape[0])
    dic = {"x": x_shape, "kernel": k_shape}
    param_list = ["kernel"]
    return out, dic, param_list


def get_dense_with_bias(x_shape, k_shape, dtype="float32"):
    dense, dic, param_list = get_dense(x_shape, k_shape, dtype)
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    out = relay.nn.bias_add(dense, bias)
    dic["bias"] = (k_shape[0],)
    param_list += ["bias"]

    return out, dic, param_list


def test_dense():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)

    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_list
    run_and_verify_func(config, dtype=dtype)


def test_dense_libxsmm_not_enabled():
    dtype = "float32"
    x_shape = (1024, 1024)
    k_shape = (1024, 1024)

    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense = tvm.IRModule.from_expr(dense)
    config = dense, dic, param_list
    run_and_verify_func(config, dtype=dtype, libxsmm_enabled=False)


def test_dense_with_bias():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias = tvm.IRModule.from_expr(dense_with_bias)
    config = dense_with_bias, dic, param_list
    run_and_verify_func(config, dtype=dtype)


def test_dense_with_bias_libxsmm_not_enabled():
    dtype = "float32"
    x_shape = (256, 512)
    k_shape = (1024, 512)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias = tvm.IRModule.from_expr(dense_with_bias)
    config = dense_with_bias, dic, param_list
    run_and_verify_func(config, dtype=dtype, libxsmm_enabled=False)


def test_dense_with_relu():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)
    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense_with_relu = relay.nn.relu(dense)
    dense_with_relu = tvm.IRModule.from_expr(dense_with_relu)
    config = dense_with_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype)


def test_dense_with_relu_libxsmm_not_enabled():
    dtype = "float32"
    x_shape = (257, 256)
    k_shape = (256, 256)
    dense, dic, param_list = get_dense(x_shape, k_shape, dtype=dtype)
    dense_with_relu = relay.nn.relu(dense)
    dense_with_relu = tvm.IRModule.from_expr(dense_with_relu)
    config = dense_with_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype, libxsmm_enabled=False)


def test_dense_with_bias_and_relu():
    dtype = "float32"
    x_shape = (16, 32)
    k_shape = (64, 32)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias_and_relu = relay.nn.relu(dense_with_bias)
    dense_with_bias_and_relu = tvm.IRModule.from_expr(dense_with_bias_and_relu)
    config = dense_with_bias_and_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype)


def test_dense_with_bias_and_relu_libxsmm_not_enabled():
    dtype = "float32"
    x_shape = (256, 257)
    k_shape = (256, 257)

    dense_with_bias, dic, param_list = get_dense_with_bias(x_shape, k_shape, dtype=dtype)
    dense_with_bias_and_relu = relay.nn.relu(dense_with_bias)
    dense_with_bias_and_relu = tvm.IRModule.from_expr(dense_with_bias_and_relu)
    config = dense_with_bias_and_relu, dic, param_list
    run_and_verify_func(config, dtype=dtype, libxsmm_enabled=False)


if __name__ == "__main__":
    pytest.main([__file__])
