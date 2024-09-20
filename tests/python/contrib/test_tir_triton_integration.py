import numpy as np
import sys

import tvm
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.script import ir as I
from tvm import relax
from tvm.relax.frontend import nn
import tvm.testing
import pytest

try:
    import triton
    import triton.language as tl
except ImportError:
    pytestmark = pytest.skip("Triton is not available", allow_module_level=True)


@tvm.testing.requires_cuda
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
                T.call_triton(
                    add_kernel,
                    (T.ceildiv(m, BLOCK_SIZE),),
                    x.data,
                    y.data,
                    output.data,
                    m,
                    BLOCK_SIZE,
                )

        @R.function
        def main(x: R.Tensor(("m",), "float32"), y: R.Tensor(("m",), "float32")):
            m = T.int64()
            with R.dataflow():
                output = R.call_tir(Module.add, [x, y], relax.TensorStructInfo((m,), "float32"))
                R.output(output)
            return output

    device = tvm.cuda(0)
    x_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    y_nd = tvm.nd.array(np.random.rand(256).astype(np.float32), device)
    output_np = x_nd.numpy() + y_nd.numpy()

    with tvm.target.Target("cuda"):
        lib = relax.build(Module)
        output_nd = tvm.runtime.relax_vm.VirtualMachine(lib, device)["main"](x_nd, y_nd)
        tvm.testing.assert_allclose(output_nd.numpy(), output_np, rtol=1e-5)
