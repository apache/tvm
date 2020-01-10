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
import ctypes

def test_resnet18():
    for device in ["llvm", "cuda"]:
        if not tvm.module.enabled(device):
            print("skip because %s is not enabled..." % device)
            return

    def verify(data):
        mod, params = relay.testing.resnet.get_workload(num_layers=18)
        with relay.build_config(opt_level=3):
            graph, lib, graph_params = relay.build_module.build(mod, "llvm", params=params)
        ctx = tvm.cpu()
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input("data", data)
        module.set_input(**graph_params)
        module.run()
        out = module.get_output(0).asnumpy()
        return out

    resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
    with relay.build_config(opt_level=3):
        graph, resnet18_gpu_lib, graph_params = relay.build_module.build(resnet18_mod, "cuda", params=resnet18_params)

    from tvm.contrib import util
    temp = util.tempdir()
    path_lib = temp.relpath("deploy_lib.so")
    resnet18_gpu_lib.export_library(path_lib)
    with open(temp.relpath("deploy_graph.json"), "w") as fo:
        fo.write(graph)
    with open(temp.relpath("deploy_param.params"), "wb") as fo:
        fo.write(relay.save_param_dict(graph_params))

    loaded_lib = tvm.module.load(path_lib)
    loaded_json = open(temp.relpath("deploy_graph.json")).read()
    loaded_params = bytearray(open(temp.relpath("deploy_param.params"), "rb").read())
    data = np.random.uniform(-1, 1, size=(1, 3, 224, 224)).astype("float32")
    ctx = tvm.gpu()
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).asnumpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


def test_system_lib():
    ctx = tvm.gpu(0)
    for device in ["llvm", "cuda"]:
        if not tvm.module.enabled(device):
            print("skip because %s is not enabled..." % device)
            return
    nn = 12
    n = tvm.convert(nn)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=4)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))

    from tvm.contrib import util
    temp = util.tempdir()
    fn_add = tvm.build(s, [A, B], target="cuda", target_host="llvm -system-lib", name="add")
    path_obj = temp.relpath("add.o")
    path_lib = temp.relpath("deploy_lib.so")
    fn_add.save(path_obj)
    fn_add.export_library(path_lib)
    # Load dll, will trigger system library registration
    dll = ctypes.CDLL(path_lib)
    # Load the system wide library
    m = tvm.module.system_lib()
    a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), ctx)
    m['add'](a, b)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)


if __name__ == "__main__":
    test_resnet18()
    test_system_lib()
