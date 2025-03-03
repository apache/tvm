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
import tvm.testing
from tvm import relax
from tvm.script import tir as T, ir as I, relax as R

add_cuda_source = """
extern "C" __global__ void add_kernel(float* x, float* y, float* output, int n_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        output[i] = x[i] + y[i];
    }
}
"""


@tvm.testing.requires_cuda
def test_tir_call_source_kernel():
    @I.ir_module
    class Module:
        @T.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle) -> None:
            T.func_attr({"global_symbol": "add"})
            m = T.int64()
            x = T.match_buffer(x_handle, (m,), "float32")
            y = T.match_buffer(y_handle, (m,), "float32")
            output = T.match_buffer(output_handle, (m,), "float32")
            with T.block("root"):
                T.reads(x[0:m], y[0:m])
                T.writes(output[0:m])
                BLOCK_SIZE = T.meta_var(64)
                T.call_kernel(
                    add_cuda_source,
                    ((T.ceildiv(m, BLOCK_SIZE),), (BLOCK_SIZE,)),
                    x.data,
                    y.data,
                    output.data,
                    m,
                    kernel_name="add_kernel",
                )

        @R.function
        def main(x: R.Tensor(("m",), "float32"), y: R.Tensor(("m",), "float32")):
            m = T.int64()
            with R.dataflow():
                output = R.call_tir(Module.add, [x, y], relax.TensorStructInfo((m,), "float32"))
                R.output(output)
            return output

    @I.ir_module
    class Parsed:
        @T.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle):
            m = T.int64()
            x = T.match_buffer(x_handle, (m,))
            y = T.match_buffer(y_handle, (m,))
            output = T.match_buffer(output_handle, (m,))
            with T.block("root"):
                T.reads(x[0:m], y[0:m])
                T.writes(output[0:m])
                T.call_packed(
                    "add_kernel",
                    x.data,
                    y.data,
                    output.data,
                    m,
                    (m + T.int64(64) - T.int64(1)) // T.int64(64),
                    64,
                )

    tvm.ir.assert_structural_equal(Module["add"], Parsed["add"])
    assert len(Module.get_attr("external_mods")) == 1

    device = tvm.cuda(0)
    x_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    y_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    output_np = x_nd.numpy() + y_nd.numpy()

    with tvm.target.Target("cuda"):
        lib = relax.build(Module)
        output_nd = tvm.runtime.relax_vm.VirtualMachine(lib, device)["main"](x_nd, y_nd)
        tvm.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)
