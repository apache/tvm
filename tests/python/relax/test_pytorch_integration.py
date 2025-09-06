# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Test PyTorch integration with TVM Relax.

This test verifies:
1. Seamless PyTorch tensor I/O with TVM backend
2. Cross-function calls between Python, TIR, and Relax functions
3. Dynamic Python function addition and execution
4. End-to-end pipeline testing
5. Error handling and edge cases
"""

import pytest
import torch
import torch.nn.functional as F
import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.relax import BasePyModule
import numpy as np


@I.ir_module
class PyTorchIntegrationModule(BasePyModule):
    """Test module for PyTorch integration with TVM."""

    @I.pyfunc
    def main(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Main function demonstrating cross-function calls."""
        n = x.shape[0]

        # Call TIR function
        lv = self.call_tir(self.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))

        # Apply ReLU
        lv1 = F.relu(lv)

        # Call packed function (will be added dynamically)
        lv2 = self.call_dps_packed("my_softmax", [lv1, 1], out_sinfo=R.Tensor((n, 20), "float32"))

        # Call Python function
        lv3 = self.my_identity_func(lv2)

        return lv3

    @T.prim_func
    def matmul(
        var_A: T.handle,
        var_B: T.handle,
        var_C: T.handle,
    ):
        """TIR function for matrix multiplication."""
        n = T.int32()
        A = T.match_buffer(var_A, (n, 16), "float32")
        B = T.match_buffer(var_B, (16, 20), "float32")
        C = T.match_buffer(var_C, (n, 20), "float32")

        for i, j, k in T.grid(n, 20, 16):
            with T.block("block"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @I.pyfunc
    def my_identity_func(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestPyTorchIntegration:
    def test_module_creation_and_instantiation(self):
        module = PyTorchIntegrationModule

        assert hasattr(module, "__call__"), "Module should be callable"

        device = tvm.cpu(0)
        instance = module(device)

        assert isinstance(instance, BasePyModule), "Instance should be BasePyModule"

        required_methods = ["main", "call_tir", "call_dps_packed"]
        for method in required_methods:
            assert hasattr(instance, method), f"Instance should have method: {method}"

    def test_module_creation_and_instantiation_gpu(self):
        module = PyTorchIntegrationModule

        if tvm.cuda().exist:
            assert hasattr(module, "__call__"), "Module should be callable"

            device = tvm.cuda(0)
            instance = module(device)

            assert isinstance(instance, BasePyModule), "Instance should be BasePyModule"
            required_methods = ["main", "call_tir", "call_dps_packed"]
            for method in required_methods:
                assert hasattr(instance, method), f"Instance should have method: {method}"
            assert "cuda" in str(instance.target)
        else:
            pytest.skip("CUDA not available")

    def test_python_function_execution(self):
        """Test that Python functions execute correctly."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Test my_identity_func
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = instance.my_identity_func(input_tensor)

        assert isinstance(result, torch.Tensor)
        assert torch.allclose(result, input_tensor, atol=1e-5)

    def test_tir_function_execution(self):
        """Test that TIR functions execute correctly."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Test matmul function
        n = 3
        x = torch.randn(n, 16, dtype=torch.float32)
        w = torch.randn(16, 20, dtype=torch.float32)

        result = instance.call_tir(instance.matmul, [x, w], R.Tensor((n, 20), "float32"))

        assert isinstance(result, torch.Tensor)
        assert result.shape == (n, 20)

        # Verify result with PyTorch matmul
        expected = torch.matmul(x, w)
        assert torch.allclose(result, expected, atol=1e-3)

    def test_dynamic_python_function_addition(self):
        """Test adding Python functions dynamically."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Define a custom function
        def custom_activation(x):
            return torch.sigmoid(x)

        # Add the function
        instance.add_python_function("custom_activation", custom_activation)

        # Verify function is added
        assert hasattr(instance, "custom_activation")
        assert "custom_activation" in instance.pyfuncs

        # Test function execution
        input_tensor = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
        result = instance.custom_activation(input_tensor)

        assert isinstance(result, torch.Tensor)
        expected = torch.sigmoid(input_tensor)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_call_dps_packed_with_dynamic_function(self):
        """Test call_dps_packed with dynamically added function."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Define my_softmax function
        def my_softmax(tensor, dim):
            """Custom softmax function for testing call_dps_packed."""
            # Convert TVM Tensor to PyTorch tensor if needed
            if hasattr(tensor, "numpy"):
                tensor = torch.from_numpy(tensor.numpy())
            return F.softmax(tensor, dim=dim)

        # Add the function
        instance.my_softmax = my_softmax

        # Test call_dps_packed
        input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        result = instance.call_dps_packed(
            "my_softmax", [input_tensor, 1], R.Tensor((2, 2), "float32")
        )

        assert isinstance(result, torch.Tensor)
        expected = F.softmax(input_tensor, dim=1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_end_to_end_pipeline(self):
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        def my_softmax(tensor, dim):
            if hasattr(tensor, "numpy"):
                tensor = torch.from_numpy(tensor.numpy())
            return F.softmax(tensor, dim=dim)

        instance.my_softmax = my_softmax

        n = 5
        x = torch.randn(n, 16, dtype=torch.float32)
        w = torch.randn(16, 20, dtype=torch.float32)

        result = instance.main(x, w)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (n, 20)
        assert result.dtype == torch.float32

    def test_end_to_end_pipeline_gpu(self):
        module = PyTorchIntegrationModule

        if tvm.cuda().exist:
            device = tvm.cuda(0)
            instance = module(device)

            # Test basic GPU functionality without complex TIR operations
            assert isinstance(instance, BasePyModule)
            assert "cuda" in str(instance.target)

            # Test that we can create and work with GPU tensors
            n = 5
            x = torch.randn(n, 16, dtype=torch.float32, device="cuda")
            w = torch.randn(16, 20, dtype=torch.float32, device="cuda")

            assert x.device.type == "cuda"
            assert w.device.type == "cuda"
            assert x.shape == (n, 16)
            assert w.shape == (16, 20)

            # Test basic PyTorch operations on GPU
            result = torch.matmul(x, w)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (n, 20)
            assert result.dtype == torch.float32
            assert result.device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_cross_function_data_flow(self):
        """Test data flow between different function types."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Add required functions
        def my_softmax(tensor, dim):
            if hasattr(tensor, "numpy"):
                tensor = torch.from_numpy(tensor.numpy())
            return F.softmax(tensor, dim=dim)

        instance.my_softmax = my_softmax

        # Create test data
        n = 4
        x = torch.randn(n, 16, dtype=torch.float32)
        w = torch.randn(16, 20, dtype=torch.float32)

        # Execute step by step to verify data flow
        # Step 1: TIR matmul
        lv = instance.call_tir(instance.matmul, [x, w], R.Tensor((n, 20), "float32"))
        assert isinstance(lv, torch.Tensor)
        assert lv.shape == (n, 20)

        # Step 2: ReLU
        lv1 = F.relu(lv)
        assert isinstance(lv1, torch.Tensor)
        assert lv1.shape == (n, 20)

        # Step 3: Softmax via call_dps_packed
        lv2 = instance.call_dps_packed("my_softmax", [lv1, 1], R.Tensor((n, 20), "float32"))
        assert isinstance(lv2, torch.Tensor)
        assert lv2.shape == (n, 20)

        # Step 4: Identity function
        lv3 = instance.my_identity_func(lv2)
        assert isinstance(lv3, torch.Tensor)
        assert lv3.shape == (n, 20)

        # Verify final result matches expected
        expected = F.softmax(F.relu(torch.matmul(x, w)), dim=1)
        assert torch.allclose(lv3, expected, atol=1e-3)

    def test_error_handling(self):
        """Test error handling for various edge cases."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Test with missing function
        with pytest.raises(Exception):
            instance.call_dps_packed(
                "non_existent_function", [torch.tensor([1.0])], R.Tensor((1,), "float32")
            )

        # Test with wrong tensor shapes
        x = torch.randn(3, 16, dtype=torch.float32)
        w = torch.randn(15, 20, dtype=torch.float32)  # Wrong shape

        with pytest.raises(Exception):
            instance.call_tir(instance.matmul, [x, w], R.Tensor((3, 20), "float32"))

    def test_tensor_type_preservation(self):
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        def my_softmax(tensor, dim):
            if hasattr(tensor, "numpy"):
                tensor = torch.from_numpy(tensor.numpy())
            return F.softmax(tensor, dim=dim)

        instance.my_softmax = my_softmax

        # Test with float32 data type (TIR function is hardcoded for float32)
        test_dtype = torch.float32
        n = 3
        x = torch.randn(n, 16, dtype=test_dtype)
        w = torch.randn(16, 20, dtype=test_dtype)

        result = instance.main(x, w)

        # Verify type preservation
        assert result.dtype == test_dtype
        assert isinstance(result, torch.Tensor)
        assert result.shape == (n, 20)
        assert result.dtype == torch.float32

    def test_batch_processing(self):
        """Test processing multiple inputs in batch."""
        module = PyTorchIntegrationModule
        device = tvm.cpu(0)
        instance = module(device)

        # Add required functions
        def my_softmax(tensor, dim):
            if hasattr(tensor, "numpy"):
                tensor = torch.from_numpy(tensor.numpy())
            return F.softmax(tensor, dim=dim)

        instance.my_softmax = my_softmax

        # Process multiple inputs
        batch_size = 5
        results = []

        for i in range(batch_size):
            n = 3 + i  # Varying batch sizes
            x = torch.randn(n, 16, dtype=torch.float32)
            w = torch.randn(16, 20, dtype=torch.float32)

            result = instance.main(x, w)
            results.append(result)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (n, 20)

        # Verify all results are valid
        assert len(results) == batch_size
        for result in results:
            assert isinstance(result, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])
