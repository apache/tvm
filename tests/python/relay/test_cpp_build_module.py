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
from tvm import te
from tvm import relay, runtime
from tvm.contrib.nvcc import have_fp16
import tvm.testing


def test_basic_build():
    tgt = "llvm"
    dev = tvm.cpu()
    # func
    a = relay.var("a", dtype="float32", shape=(16, 8))
    b = relay.var("b", dtype="float32", shape=(8, 8))
    c = relay.var("c", dtype="float32", shape=(16, 8))
    x = relay.nn.dense(a, b)
    y = relay.nn.relu(x)
    z = y + c
    func = relay.Function([a, b, c], z)
    A = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), device=dev)
    B = tvm.nd.array(np.random.uniform(-1, 1, (8, 8)).astype("float32"), device=dev)
    C = tvm.nd.array(np.random.uniform(-1, 1, (16, 8)).astype("float32"), device=dev)
    params = {"b": B, "c": C}
    # build
    targets = {tvm.tir.IntImm("int32", dev.device_type): tgt}
    mod = tvm.IRModule.from_expr(func)
    func_in_mod = mod["main"]
    assert mod["main"] == func_in_mod, "cannot compare function to itself"

    lib = relay.build(mod, targets, "llvm", params=params)
    assert mod["main"] == func_in_mod, "relay.build changed module in-place"

    # test
    rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    rt.set_input("a", A)
    rt.run()
    out = rt.get_output(0)

    np.testing.assert_allclose(
        out.numpy(),
        np.maximum(np.dot(A.numpy(), B.numpy().T), 0) + C.numpy(),
        atol=1e-5,
        rtol=1e-5,
    )


@tvm.testing.requires_cuda
def test_fp16_build():
    dtype = "float16"

    dev = tvm.cuda(0)
    if dtype == "float16" and not have_fp16(dev.compute_version):
        print("skip because gpu does not support fp16")
        return

    x = relay.var("x", dtype=dtype, shape=(4, 4))
    y = relay.var("y", dtype=dtype, shape=(4, 4))
    z = x + y
    func = relay.Function([x, y], z)
    X = tvm.nd.array(np.random.uniform(-1, 1, (4, 4)).astype(dtype), device=dev)
    Y = tvm.nd.array(np.random.uniform(-1, 1, (4, 4)).astype(dtype), device=dev)
    params = {
        "x": X,
        "y": Y,
    }

    # build
    g_json, mmod, params = relay.build(func, "cuda", params=params)

    # test
    rt = tvm.contrib.graph_executor.create(g_json, mmod, dev)
    rt.load_params(runtime.save_param_dict(params))
    rt.run()
    out = rt.get_output(0)

    np.testing.assert_allclose(out.numpy(), X.numpy() + Y.numpy(), atol=1e-5, rtol=1e-5)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_fp16_conversion(target, dev):
    if target == "cuda" and not have_fp16(dev.compute_version):
        print("skip because gpu does not support fp16")
        return

    n = 10

    for (src, dst) in [("float32", "float16"), ("float16", "float32")]:
        x = relay.var("x", relay.TensorType((n,), src))
        y = x.astype(dst)
        func = relay.Function([x], y)

        # init input
        X = tvm.nd.array(n * np.random.randn(n).astype(src) - n / 2)

        # build
        with tvm.transform.PassContext(opt_level=1):
            g_json, mmod, params = relay.build(tvm.IRModule.from_expr(func), target)

        # test
        rt = tvm.contrib.graph_executor.create(g_json, mmod, dev)
        rt.set_input("x", X)
        rt.run()
        out = rt.get_output(0)

        np.testing.assert_allclose(out.numpy(), X.numpy().astype(dst), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_basic_build()
    test_fp16_build()
    test_fp16_conversion()
