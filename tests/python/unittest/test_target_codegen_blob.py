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
from tvm.contrib import graph_executor
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
            lib = relay.build_module.build(mod, "llvm", params=params)
        dev = tvm.cpu()
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input("data", data)
        module.run()
        out = module.get_output(0).numpy()
        return out

    synthetic_mod, synthetic_params = relay.testing.synthetic.get_workload(input_shape=input_shape)
    with tvm.transform.PassContext(opt_level=3):
        synthetic_gpu_lib = relay.build_module.build(synthetic_mod, "cuda", params=synthetic_params)

    from tvm.contrib import utils

    temp = utils.tempdir()
    path_lib = temp.relpath("deploy_lib.so")
    synthetic_gpu_lib.export_library(path_lib)

    loaded_lib = tvm.runtime.load_module(path_lib)
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    dev = tvm.cuda()
    module = graph_executor.GraphModule(loaded_lib["default"](dev))
    module.set_input("data", data)
    module.run()
    out = module.get_output(0).numpy()

    tvm.testing.assert_allclose(out, verify(data), atol=1e-5)


@tvm.testing.uses_gpu
def test_cuda_lib():
    dev = tvm.cuda(0)
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

    from tvm.contrib import utils

    temp = utils.tempdir()
    fn_add = tvm.build(s, [A, B], target="cuda --host=llvm", name="add")
    path_lib = temp.relpath("deploy_lib.so")
    fn_add.export_library(path_lib)
    m = tvm.runtime.load_module(path_lib)
    a = tvm.nd.array(np.random.uniform(size=nn).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(nn, dtype=A.dtype), dev)
    m["add"](a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)


if __name__ == "__main__":
    test_synthetic()
    test_cuda_lib()
