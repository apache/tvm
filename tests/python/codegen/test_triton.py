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
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

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
def test_triton_call_kernel():
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

    add_m = T.dynamic("m", "int64")

    @I.ir_module
    class Module:
        @T.function
        def add(
            x: T.Tensor((add_m,), "float32"),
            y: T.Tensor((add_m,), "float32"),
            output: T.Tensor((add_m,), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add"})

            T.call_kernel(
                add_kernel,
                (T.ceildiv(add_m, BLOCK_SIZE),),
                x.data,
                y.data,
                output.data,
                add_m,
                BLOCK_SIZE,
                num_warps=8,
            )

    # Triton omits constexpr arguments and appends global scratch, plus profile
    # scratch starting in 3.5. This kernel requires no scratch allocation.
    scratch_args = [tvm_tirx.tirx.reinterpret(tvm_tirx.ir.IntImm("uint64", 0), ty="handle")]
    if version.parse(triton.__version__) >= version.parse("3.5.0"):
        scratch_args.append(tvm_tirx.tirx.reinterpret(tvm_tirx.ir.IntImm("uint64", 0), ty="handle"))

    # The thread extent is 256 because the kernel is compiled with num_warps=8.
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Parsed:
        @T.function
        def add(
            x: T.Tensor((m,), "float32"),
            y: T.Tensor((m,), "float32"),
            output: T.Tensor((m,), "float32"),
        ):
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

    tvm_tirx.ir.assert_structural_equal(Module["add"], Parsed["add"])
    assert len(Module.get_attr("external_mods")) == 1

    with tvm_tirx.target.Target("cuda"):
        lib = tvm_tirx.tirx.compile(Module)

    def run_and_check():
        device = tvm_tirx.cuda(0)
        x_nd = tvm_tirx.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        y_nd = tvm_tirx.runtime.tensor(np.random.rand(256).astype(np.float32), device)
        output_np = x_nd.numpy() + y_nd.numpy()
        output_nd = tvm_tirx.runtime.empty((256,), "float32", device)
        lib["add"](x_nd, y_nd, output_nd)
        tvm_tirx.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
