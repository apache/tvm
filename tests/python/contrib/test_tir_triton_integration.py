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
# ruff: noqa: F401
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend import nn
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.testing import env

try:
    import triton
    import triton.language as tl
    from packaging import version
except ImportError:
    pytestmark = pytest.skip("Triton is not available", allow_module_level=True)
else:
    if version.parse(triton.__version__) < version.parse("3.3.0"):
        pytestmark = pytest.skip("Triton >= 3.3.0 is required", allow_module_level=True)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_tir_triton_integration():
    @triton.jit
    def add_kernel(
        x_ptr,  # *Pointer* to first input vector.
        y_ptr,  # *Pointer* to second input vector.
        output_ptr,  # *Pointer* to output vector.
        n_elements,  # Size of the vector.
        BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    ):
        """Triton vector add kernel from its tutorial."""
        pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    BLOCK_SIZE = 64

    @I.ir_module
    class Module:
        @Ts.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle) -> None:
            T.func_attr({"global_symbol": "add"})
            m = T.int64()
            x = T.match_buffer(x_handle, (m,), "float32")
            y = T.match_buffer(y_handle, (m,), "float32")
            output = T.match_buffer(output_handle, (m,), "float32")
            with Ts.sblock("root"):
                Ts.reads(x[0:m], y[0:m])
                Ts.writes(output[0:m])
                T.call_kernel(
                    add_kernel,
                    (T.ceildiv(m, BLOCK_SIZE),),
                    x.data,
                    y.data,
                    output.data,
                    m,
                    BLOCK_SIZE,
                    num_warps=8,
                )

        @R.function
        def main(x: R.Tensor(("m",), "float32"), y: R.Tensor(("m",), "float32")):
            m = T.int64()
            with R.dataflow():
                output = R.call_tir(Module.add, [x, y], relax.TensorType((m,), "float32"))
                R.output(output)
            return output

    # Triton omits constexpr arguments and appends global scratch, plus profile
    # scratch starting in 3.5. This kernel requires no scratch allocation.
    scratch_args = [tvm.tirx.reinterpret("handle", tvm.tirx.IntImm("uint64", 0))]
    if version.parse(triton.__version__) >= version.parse("3.5.0"):
        scratch_args.append(tvm.tirx.reinterpret("handle", tvm.tirx.IntImm("uint64", 0)))

    # The thread extent is 256 because the kernel is compiled with num_warps=8.
    @I.ir_module
    class Parsed:
        @Ts.prim_func
        def add(x_handle: T.handle, y_handle: T.handle, output_handle: T.handle):
            m = T.int64()
            x = T.match_buffer(x_handle, (m,))
            y = T.match_buffer(y_handle, (m,))
            output = T.match_buffer(output_handle, (m,))
            with Ts.sblock("root"):
                Ts.reads(x[0:m], y[0:m])
                Ts.writes(output[0:m])
                T.call_packed(
                    "add_kernel",
                    x.data,
                    y.data,
                    output.data,
                    m,
                    *scratch_args,
                    256,
                    (m + T.int64(64) - T.int64(1)) // T.int64(64),
                )

    tvm.ir.assert_structural_equal(Module["add"], Parsed["add"])
    assert len(Module.get_attr("external_mods")) == 1

    with tvm.target.Target("cuda"):
        lib = tvm.compile(Module)

    def run_and_check():
        device = tvm.cuda(0)
        x_nd = tvm.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        y_nd = tvm.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        output_np = x_nd.numpy() + y_nd.numpy()
        output_nd = tvm.runtime.vm.VirtualMachine(lib, device)["main"](x_nd, y_nd)
        tvm.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)

    tvm.testing.run_with_gpu_lock(run_and_check)
