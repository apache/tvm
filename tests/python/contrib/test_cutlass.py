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
import traceback
import tvm
from tvm import relay
import numpy as np
import time
from tvm.contrib.cutlass import profile_and_build


M = 1820
N = 768
K = 768


def get_mod():
    data = relay.var("data", shape=(M, K), dtype="float16")
    weight = relay.var("weight", shape=(N, K), dtype="float16")
    bias = relay.var("bias", shape=(N,), dtype="float16")
    gemm_out = relay.nn.dense(data, weight)
    # gemm_out = relay.nn.bias_add(gemm_out, bias)
    # gemm_out = relay.nn.relu(gemm_out)
    # gemm_out = relay.nn.dense(gemm_out, weight)
    out = gemm_out
    return tvm.IRModule.from_expr(out)


mod = get_mod()

np_data = np.random.uniform(-1, 1, (M, K)).astype("float16")
np_weight = np.random.uniform(-1, 1, (N, K)).astype("float16")
np_bias = np.random.uniform(-1, 1, (N,)).astype("float16")

tvm_data = np_data
tvm_weight = np_weight
tvm_bias = np_bias

params = {"weight": tvm_weight, "bias": tvm_bias}

print("compiling...")
sm = "86"
rt_mod, ctx = profile_and_build(mod, params, sm)
x = tvm.nd.array(tvm_data, device=ctx)
rt_mod.set_input("data", x)

print("Running for the first time...")
rt_mod.run()
y = rt_mod.get_output(0)

print("np computing...")
np_out = np.dot(np_data, np_weight.T)
# np_out = np_out + np_bias
# np_out = np.dot(np_out, np_weight.T)
# np_out = np_out * (np_out > 0)
# np_out = np_out*(0.5+erf(np_out * np.sqrt(0.5)) * 0.5)

try:
    np.testing.assert_allclose(y.asnumpy(), np_out, atol=1e-2, rtol=1e-2)
    print("Accuracy test passed...")
except:
    traceback.print_exc()
    print("Accuracy test failed...")


times = []
for i in range(100):
    start = time.time()
    rt_mod.run()
    ctx.sync()  # wait for the device to finish
    times.append(time.time() - start)
print("Latency:", 1000.0 * np.mean(times), "ms")
print("TFLOPS:", 2 * M * N * K / np.mean(times) / 1e12)
