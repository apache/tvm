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

import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor


@tvm.testing.requires_hexagon
def test_dense_u8u8i32_vrmpy(hexagon_session):
    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    target = tvm.target.Target(target_hexagon, host=target_hexagon)

    M = 128
    N = 768
    K = 768
    data_shape = (M, K)
    weight_shape = (N, K)

    dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)

    dense = relay.nn.dense(data, weight, out_dtype="int32")

    use_bias = False

    if dtype == "uint8":
        data_np = np.random.uniform(1, 255, size=data_shape).astype(dtype)
        weight_np = np.random.uniform(1, 255, size=weight_shape).astype(dtype)
    else:
        data_np = np.random.uniform(-128, 127, size=data_shape).astype(dtype)
        weight_np = np.random.uniform(-128, 127, size=weight_shape).astype(dtype)

    # data_np = np.ones(data_shape).astype(dtype) * 127
    # weight_np =  np.ones(weight_shape).astype(dtype) * 127

    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    params = {"weight": weight_np, "bias": bias_np}

    if use_bias:
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        out = relay.nn.bias_add(dense, bias)
    else:
        out = dense

    mod = tvm.IRModule.from_expr(out)

    with tvm.transform.PassContext(
        opt_level=3,
    ):
        lib = relay.build(mod, target=target, params=params)

    asm = lib.lib.get_source("asm")
#    assert "vrmpy" in asm

    rt_mod = hexagon_session.get_executor_from_factory(lib)

    rt_mod.set_input("data", data_np)

    rt_mod.run()

    out = rt_mod.get_output(0).numpy()
    ref = np.dot(data_np.astype("int32"), weight_np.transpose().astype("int32"))

    if use_bias:
        ref += bias_np

    np.testing.assert_equal(out, ref)
