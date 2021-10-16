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
import tvm
from tvm import relay
import numpy as np
from tvm.contrib.cutlass import profile_and_build


def get_mod(M, N, K):
    data = relay.var("data", shape=(M, K), dtype="float16")
    weight = relay.var("weight", shape=(N, K), dtype="float16")
    bias = relay.var("bias", shape=(N,), dtype="float16")
    gemm_out = relay.nn.dense(data, weight)
    # gemm_out = relay.nn.bias_add(gemm_out, bias)
    # gemm_out = relay.nn.relu(gemm_out)
    # gemm_out = relay.nn.dense(gemm_out, weight)
    out = gemm_out
    return tvm.IRModule.from_expr(out)


def get_ref_rt_mod(mod, params):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", params=params)
    ctx = tvm.gpu()
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
    return rt_mod, ctx


def get_output(rt_mod, x):
    rt_mod.set_input("data", x)
    rt_mod.run()
    return rt_mod.get_output(0).asnumpy()


def get_output_numpy(np_data, np_weight):
    np_out = np.dot(np_data, np_weight.T)
    # np_out = np_out + np_bias
    # np_out = np.dot(np_out, np_weight.T)
    # np_out = np_out * (np_out > 0)
    # np_out = np_out*(0.5+erf(np_out * np.sqrt(0.5)) * 0.5)
    return np_out


M = 1820
N = 768
K = 768

mod = get_mod(M, N, K)

np_data = np.random.uniform(-1, 1, (M, K)).astype("float16")
np_weight = np.random.uniform(-1, 1, (N, K)).astype("float16")
np_bias = np.random.uniform(-1, 1, (N,)).astype("float16")

tvm_weight = np_weight
tvm_bias = np_bias

params = {"weight": tvm_weight, "bias": tvm_bias}

rt_mod, ctx = profile_and_build(mod, params, "80")
rt_mod_ref, ctx = get_ref_rt_mod(get_mod(M, N, K), params)
x = tvm.nd.array(np_data, device=ctx)

out = get_output(rt_mod, x)
ref_out = get_output(rt_mod_ref, x)
np_out = get_output_numpy(np_data, np_weight)

np.testing.assert_allclose(out, ref_out, atol=1e-5, rtol=1e-5)
