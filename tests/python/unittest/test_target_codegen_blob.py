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
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_runtime
import tvm
from tvm import te
import ctypes
import tvm.testing


@tvm.testing.uses_gpu
def test_synthetic():
    for device in ["llvm", "cuda"]:
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled..." % device)
            return

    input_shape = (1, 5, 23, 61)

    def verify(data):
        mod, params = relay.testing.synthetic.get_workload(input_shape=input_shape)
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)
        ctx = tvm.cpu()
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input("data", data)
        module.set_input(**graph_params)
        module.run()
        out = module.get_output(0).asnumpy()
        return out

    synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload(input_shape=input_shape)
    with tvm.transform.PassContext(opt_level=3):
        graph, synthetic_gpu_lib, graph_params = relay.build_module.build(
            synthetic_mod, "cuda", params=synthetic_params
        )

    from tvm.contrib import util

    temp = util.tempdir()
    path_lib = temp.relpath("deploy_lib.so")
    synthetic_gpu_lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))

    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    ctx = tvm.gpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.uses_gpu
def test_cuda_lib():
    ctx = tvm.gpu(0)
    for device in ["llvm", "cuda"]:
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled..." % device)
            return
    nn = 12
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=4)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))

    from tvm.contrib import util

    temp = util.tempdir()
    fn_add = tvm.build(s, [A, B], target="cuda", target_host="llvm", name="add")
    path_lib = temp.relpath("deploy_lib.so")
    fn_add.export_library(path_lib)
    m = tvm.runtime.load_module(path_lib)
    a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
    m["add"](a, b)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)


if __name__ == "__main__":
    test_synthetic()
    test_cuda_lib()
