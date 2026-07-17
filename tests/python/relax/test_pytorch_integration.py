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
5. Missing packed-function error handling
"""

import pytest
import torch
import torch.nn.functional as F

import tvm
from tvm.relax import BasePyModule
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class PyTorchIntegrationModule(BasePyModule):
    """Test module for PyTorch integration with TVM."""

    @I.pyfunc
    def main(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Main function demonstrating cross-function calls."""
        n = x.shape[0]

        # Call TIR function
        lv = self.call_tir(self.matmul, [x, w], out_ty=R.Tensor((n, 20), "float32"))

        # Apply ReLU
        lv1 = F.relu(lv)

        # Call packed function (will be added dynamically)
        lv2 = self.call_dps_packed("my_softmax", [lv1, 1], out_ty=R.Tensor((n, 20), "float32"))

        # Call Python function
        lv3 = self.my_identity_func(lv2)

        return lv3

    @T.prim_func(s_tir=True)
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
            with T.sblock("block"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @I.pyfunc
    def my_identity_func(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestPyTorchIntegration:
    def test_end_to_end_pipeline(self):
        instance = PyTorchIntegrationModule(tvm.cpu(0))

        def my_softmax(tensor, dim):
            return F.softmax(tensor, dim=dim)

        instance.add_python_function("my_softmax", my_softmax)
        assert "my_softmax" in instance.pyfuncs

        torch.manual_seed(0)
        w = torch.randn(16, 20, dtype=torch.float32)

        # Reuse the compiled module with two different symbolic extents.
        for n in (3, 5):
            x = torch.randn(n, 16, dtype=torch.float32)
            result = instance.main(x, w)
            expected = F.softmax(F.relu(torch.matmul(x, w)), dim=1)

            assert isinstance(result, torch.Tensor)
            assert result.shape == (n, 20)
            assert result.dtype == torch.float32
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-3)

    def test_missing_packed_function(self):
        instance = BasePyModule(tvm.IRModule({}), tvm.cpu(0))

        with pytest.raises(
            ValueError,
            match="Function 'non_existent_function' not found as a global function",
        ):
            instance.call_dps_packed(
                "non_existent_function",
                [torch.tensor([1.0])],
                R.Tensor((1,), "float32"),
            )


if __name__ == "__main__":
    pytest.main([__file__])
