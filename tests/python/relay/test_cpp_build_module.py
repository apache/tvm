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

import tvm
from tvm import relay
from tvm.contrib.nvcc import have_fp16

from tvm._ffi.function import _init_api
_init_api("tvm.relay.build_module")


def test_basic_build():
    tgt = "llvm"
    ctx = tvm.cpu()
    # func
    a = relay.var("a", dtype="float32", shape=(16, 8))
    b = relay.var("b", dtype="float32", shape=(8, 8))
    c = relay.var("c", dtype="float32", shape=(16, 8))
    x = relay.nn.dense(a, b)
    y = relay.nn.relu(x)
    z = y + c
    func = relay.Function([a, b, c], z)
    A = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), ctx=ctx)
    B = tvm.nd.array(np.random.uniform(-1, 1, (8, 8)).astype("float32"), ctx=ctx)
    C = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), ctx=ctx)
    params = {
        "b" : B,
        "c" : C
    }
    # build
    targets = {
        tvm.expr.IntImm("int32", ctx.device_type): tgt
    }
    g_json, mmod, params = relay.build(func, targets, "llvm", params=params)

    # test
    rt = tvm.contrib.graph_runtime.create(g_json, mmod, ctx)
    rt.set_input("a", A)
    rt.load_params(relay.save_param_dict(params))
    rt.run()
    out = rt.get_output(0)

    np.testing.assert_allclose(out.asnumpy(), np.maximum(np.dot(A.asnumpy(),
                                                                B.asnumpy().T),
                                                         0) + C.asnumpy(),
                               atol=1e-5, rtol=1e-5)


def test_fp16_build():
    dtype = "float16"

    if not tvm.module.enabled("cuda") or not tvm.gpu(0).exist:
        print("skip because cuda is not enabled..")
        return

    ctx = tvm.gpu(0)
    if dtype == "float16" and not have_fp16(ctx.compute_version):
        print("skip because gpu does not support fp16")
        return

    x = relay.var("x", dtype=dtype, shape=(4, 4))
    y = relay.var("y", dtype=dtype, shape=(4, 4))
    z = x + y
    func = relay.Function([x, y], z)
    X = tvm.nd.array(np.random.uniform(-1, 1, (4, 4)).astype(dtype), ctx=ctx)
    Y = tvm.nd.array(np.random.uniform(-1, 1, (4, 4)).astype(dtype), ctx=ctx)
    params = {
        "x": X,
        "y": Y,
    }

    # build
    g_json, mmod, params = relay.build(func, "cuda", params=params)

    # test
    rt = tvm.contrib.graph_runtime.create(g_json, mmod, ctx)
    rt.set_input("x", X)
    rt.set_input("y", Y)
    rt.load_params(relay.save_param_dict(params))
    rt.run()
    out = rt.get_output(0)

    np.testing.assert_allclose(out.asnumpy(), X.asnumpy() + Y.asnumpy(),
                               atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_basic_build()
    test_fp16_build()
