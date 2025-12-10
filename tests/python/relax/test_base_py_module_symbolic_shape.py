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

import tvm
from tvm.ir import IRModule
from tvm.relax.base_py_module import BasePyModule
from tvm import tir, relax
from tvm.script import ir as I, tir as T, relax as R


def _make_module():
    return IRModule({})


def test_infer_concrete_shape_from_numpy_input():
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    m = tir.Var("m", "int64")
    sym_shape = [n, m]

    x = np.zeros((3, 4), dtype="float32")
    inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x])
    assert inferred == [3, 4]


def test_infer_concrete_shape_all_concrete_dims():
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    shape = [tir.IntImm("int32", 5), 6]
    inferred = bpm._infer_concrete_shape_from_args(shape, in_args=[])
    assert inferred == [5, 6]


def test_infer_concrete_shape_error_when_uninferrable():
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    k = tir.Var("k", "int64")
    with pytest.raises(ValueError):
        bpm._infer_concrete_shape_from_args([k, 8], in_args=[])


@I.ir_module
class AddModuleSymbolic(BasePyModule):
    @T.prim_func
    def add_tir(var_x: T.handle, var_y: T.handle, var_out: T.handle):
        T.func_attr({"global_symbol": "add_tir"})
        n = T.int64()
        x = T.match_buffer(var_x, (n,), dtype="float32")
        y = T.match_buffer(var_y, (n,), dtype="float32")
        out = T.match_buffer(var_out, (n,), dtype="float32")

        for i in T.serial(n):
            out[i] = x[i] + y[i]

    @R.function
    def main_relax(
        x: R.Tensor(("n",), "float32"), y: R.Tensor(("n",), "float32")
    ) -> R.Tensor(("n",), "float32"):
        return R.add(x, y)


def test_base_py_module_relax_symbolic_end_to_end():
    bpm = AddModuleSymbolic(device=tvm.cpu(0), target="llvm")

    a = np.random.randn(5).astype("float32")
    b = np.random.randn(5).astype("float32")
    out = bpm.main_relax(a, b)
    assert isinstance(out, np.ndarray) or hasattr(out, "numpy")
    out_np = out if isinstance(out, np.ndarray) else out.numpy()
    tvm.testing.assert_allclose(out_np, a + b, rtol=1e-6, atol=1e-6)

    a7 = np.random.randn(7).astype("float32")
    b7 = np.random.randn(7).astype("float32")
    out2 = bpm.main_relax(a7, b7)
    out2_np = out2 if isinstance(out2, np.ndarray) else out2.numpy()
    tvm.testing.assert_allclose(out2_np, a7 + b7, rtol=1e-6, atol=1e-6)


def test_base_py_module_tir_symbolic_end_to_end():
    bpm = AddModuleSymbolic(device=tvm.cpu(0), target="llvm")

    a = np.random.randn(5).astype("float32")
    b = np.random.randn(5).astype("float32")

    n = tir.Var("n", "int64")
    out_sinfo = relax.TensorStructInfo((n,), "float32")

    out = bpm.call_tir("add_tir", [a, b], out_sinfo)
    out_np = out if isinstance(out, np.ndarray) else out.numpy()
    tvm.testing.assert_allclose(out_np, a + b, rtol=1e-6, atol=1e-6)


def test_infer_concrete_shape_multiple_symbolic_dims():
    """Test shape inference with multiple symbolic dimensions."""
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    m = tir.Var("m", "int64")
    k = tir.Var("k", "int64")
    sym_shape = [n, m, k]

    x = np.zeros((2, 3, 4), dtype="float32")
    inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x])
    assert inferred == [2, 3, 4]


def test_infer_concrete_shape_mixed_concrete_symbolic():
    """Test shape inference with mixed concrete and symbolic dimensions."""
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    sym_shape = [n, 5, 10]  # First dim is symbolic, others are concrete

    x = np.zeros((3, 5, 10), dtype="float32")
    inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x])
    assert inferred == [3, 5, 10]


def test_infer_concrete_shape_from_tvm_tensors():
    """Test shape inference from TVM tensors."""
    try:
        # Try to create TVM tensor using new API
        x_np = np.zeros((3, 4), dtype="float32")
        x_tvm = tvm.runtime.tensor(x_np)

        mod = _make_module()
        bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

        n = tir.Var("n", "int64")
        m = tir.Var("m", "int64")
        sym_shape = [n, m]

        inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x_tvm])
        assert inferred == [3, 4]
    except AttributeError:
        # Skip if tvm.runtime.tensor is not available
        pytest.skip("tvm.runtime.tensor not available")


def test_infer_concrete_shape_multiple_inputs():
    """Test shape inference when multiple inputs are available."""
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    m = tir.Var("m", "int64")
    sym_shape = [n, m]

    # Multiple inputs with different shapes - should use first matching one
    x1 = np.zeros((2, 3), dtype="float32")
    x2 = np.zeros((4, 5), dtype="float32")
    inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x1, x2])
    assert inferred == [2, 3]  # Should use first input


def test_infer_concrete_shape_wrong_ndim():
    """Test shape inference when input has wrong number of dimensions."""
    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    m = tir.Var("m", "int64")
    sym_shape = [n, m]  # 2D

    x = np.zeros((3,), dtype="float32")  # 1D - wrong ndim
    with pytest.raises(ValueError, match="Cannot infer concrete output shape"):
        bpm._infer_concrete_shape_from_args(sym_shape, [x])


@I.ir_module
class MatrixModuleSymbolic(BasePyModule):
    @T.prim_func
    def matmul_tir(var_a: T.handle, var_b: T.handle, var_c: T.handle):
        T.func_attr({"global_symbol": "matmul_tir"})
        m = T.int64()
        n = T.int64()
        k = T.int64()
        a = T.match_buffer(var_a, (m, k), dtype="float32")
        b = T.match_buffer(var_b, (k, n), dtype="float32")
        c = T.match_buffer(var_c, (m, n), dtype="float32")

        for i in T.serial(m):
            for j in T.serial(n):
                c[i, j] = 0.0
                for l in T.serial(k):
                    c[i, j] = c[i, j] + a[i, l] * b[l, j]

    @R.function
    def matmul_relax(
        a: R.Tensor(("m", "k"), "float32"), b: R.Tensor(("k", "n"), "float32")
    ) -> R.Tensor(("m", "n"), "float32"):
        return R.matmul(a, b)


def test_base_py_module_multiple_symbolic_dims():
    """Test BasePyModule with multiple symbolic dimensions."""
    bpm = MatrixModuleSymbolic(device=tvm.cpu(0), target="llvm")

    # Test Relax function with multiple symbolic dims
    a = np.random.randn(2, 3).astype("float32")
    b = np.random.randn(3, 4).astype("float32")
    out = bpm.matmul_relax(a, b)
    out_np = out if isinstance(out, np.ndarray) else out.numpy()
    expected = np.matmul(a, b)
    tvm.testing.assert_allclose(out_np, expected, rtol=1e-6, atol=1e-6)

    # Test TIR function with multiple symbolic dims
    # Use concrete shapes for TIR function to avoid constraint issues
    out_sinfo = relax.TensorStructInfo((2, 4), "float32")
    out_tir = bpm.call_tir("matmul_tir", [a, b], out_sinfo)
    out_tir_np = out_tir if isinstance(out_tir, np.ndarray) else out_tir.numpy()
    tvm.testing.assert_allclose(out_tir_np, expected, rtol=1e-6, atol=1e-6)


def test_base_py_module_call_dps_packed_symbolic():
    """Test call_dps_packed with symbolic shapes."""
    try:
        # Register a simple test function
        @tvm.register_global_func("test_add_packed")
        def test_add_packed(a, b, out):
            """Add two tensors element-wise."""
            a_np = a.numpy()
            b_np = b.numpy()
            result = a_np + b_np
            out[:] = result

        mod = _make_module()
        bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

        a = np.random.randn(5).astype("float32")
        b = np.random.randn(5).astype("float32")

        n = tir.Var("n", "int64")
        out_sinfo = relax.TensorStructInfo((n,), "float32")

        out = bpm.call_dps_packed("test_add_packed", [a, b], out_sinfo)
        out_np = out if isinstance(out, np.ndarray) else out.numpy()
        tvm.testing.assert_allclose(out_np, a + b, rtol=1e-6, atol=1e-6)

    except AttributeError as e:
        pytest.skip(f"call_dps_packed test requires register_global_func: {e}")


def test_base_py_module_call_dps_packed_multiple_args():
    """Test call_dps_packed with multiple arguments and symbolic shapes."""
    try:
        # Register a function that takes multiple arguments
        @tvm.register_global_func("test_matmul_packed")
        def test_matmul_packed(a, b, out):
            """Matrix multiplication."""
            a_np = a.numpy()
            b_np = b.numpy()
            result = np.matmul(a_np, b_np)
            out[:] = result

        mod = _make_module()
        bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

        a = np.random.randn(2, 3).astype("float32")
        b = np.random.randn(3, 4).astype("float32")

        out_sinfo = relax.TensorStructInfo((2, 4), "float32")

        out = bpm.call_dps_packed("test_matmul_packed", [a, b], out_sinfo)
        out_np = out if isinstance(out, np.ndarray) else out.numpy()
        expected = np.matmul(a, b)
        tvm.testing.assert_allclose(out_np, expected, rtol=1e-6, atol=1e-6)

    except AttributeError as e:
        pytest.skip(f"call_dps_packed test requires register_global_func: {e}")


def test_base_py_module_call_dps_packed_scalar_args():
    """Test call_dps_packed with scalar arguments and symbolic shapes."""
    try:
        # Register a function that takes scalar arguments
        @tvm.register_global_func("test_add_scalar_packed")
        def test_add_scalar_packed(x, scalar, out):
            """Add scalar to tensor."""
            x_np = x.numpy()
            if hasattr(scalar, "numpy"):
                scalar_val = scalar.numpy()
            else:
                scalar_val = scalar
            result = x_np + scalar_val
            out[:] = result

        mod = _make_module()
        bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

        x = np.random.randn(4).astype("float32")
        scalar = 2.5

        n = tir.Var("n", "int64")
        out_sinfo = relax.TensorStructInfo((n,), "float32")

        out = bpm.call_dps_packed("test_add_scalar_packed", [x, scalar], out_sinfo)
        out_np = out if isinstance(out, np.ndarray) else out.numpy()
        expected = x + scalar
        tvm.testing.assert_allclose(out_np, expected, rtol=1e-6, atol=1e-6)

    except AttributeError as e:
        pytest.skip(f"call_dps_packed test requires register_global_func: {e}")


def test_infer_concrete_shape_from_pytorch_tensors():
    """Test shape inference from PyTorch tensors (if available)."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    mod = _make_module()
    bpm = BasePyModule(mod, device=tvm.cpu(0), target="llvm")

    n = tir.Var("n", "int64")
    m = tir.Var("m", "int64")
    sym_shape = [n, m]

    x_torch = torch.zeros((3, 4), dtype=torch.float32)
    inferred = bpm._infer_concrete_shape_from_args(sym_shape, [x_torch])
    assert inferred == [3, 4]


def test_base_py_module_relax_with_pytorch_tensors():
    """Test Relax functions with PyTorch tensors and symbolic shapes."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    bpm = AddModuleSymbolic(device=tvm.cpu(0), target="llvm")

    a_torch = torch.randn(5, dtype=torch.float32)
    b_torch = torch.randn(5, dtype=torch.float32)

    out = bpm.main_relax(a_torch, b_torch)
    out_np = out if isinstance(out, np.ndarray) else out.numpy()
    expected = a_torch.numpy() + b_torch.numpy()
    tvm.testing.assert_allclose(out_np, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tvm.testing.main()
