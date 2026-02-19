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

import torch
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.torch import from_fx
from tvm.relax.dpl.pattern import make_fused_bias_activation_pattern
from tvm.script import ir as I
from tvm.script import relax as R


def test_conv2d_bias_relu_fusion():
    """Test PyTorch conv2d + bias + relu fusion with reshape pattern"""

    class Conv2dBiasRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 3, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    # Convert PyTorch model to Relax IR
    model = Conv2dBiasRelu()
    graph_model = torch.fx.symbolic_trace(model)
    input_info = [([1, 3, 10, 10], "float32")]

    with torch.no_grad():
        mod = from_fx(graph_model, input_info)

    # Apply fusion with modified pattern
    patterns = [
        (
            "conv2d_bias_activation_with_reshape",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu", allow_reshape=True
            ),
        )
    ]

    fused_mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

    # Verify fusion occurred
    fused_functions = [name for name in fused_mod.functions.keys() if "fused" in str(name)]

    assert len(fused_functions) == 1, "Expected exactly one fused function"

    # Verify the fused function contains all operations
    fused_func = fused_mod[fused_functions[0]]
    assert hasattr(fused_func, "attrs"), "Fused function should have attributes"
    assert "Composite" in fused_func.attrs, "Fused function should have Composite attribute"


def test_conv2d_bias_relu_fusion_comparison():
    """Compare fusion with and without allow_reshape option"""

    class Conv2dBiasRelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 3, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(self.conv(x))

    model = Conv2dBiasRelu()
    graph_model = torch.fx.symbolic_trace(model)
    input_info = [([1, 3, 10, 10], "float32")]

    with torch.no_grad():
        mod = from_fx(graph_model, input_info)

    # Test with allow_reshape=False
    old_patterns = [
        (
            "conv2d_bias_activation_old",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu", allow_reshape=False
            ),
        )
    ]

    old_fused_mod = relax.transform.FuseOpsByPattern(old_patterns, bind_constants=False)(mod)

    # Test with allow_reshape=True
    new_patterns = [
        (
            "conv2d_bias_activation_new",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu", allow_reshape=True
            ),
        )
    ]

    new_fused_mod = relax.transform.FuseOpsByPattern(new_patterns, bind_constants=False)(mod)

    # Both should create fused functions
    old_fused_functions = [name for name in old_fused_mod.functions.keys() if "fused" in str(name)]
    new_fused_functions = [name for name in new_fused_mod.functions.keys() if "fused" in str(name)]

    assert len(old_fused_functions) >= 1, "Old pattern should create at least one fused function"
    assert len(new_fused_functions) >= 1, "New pattern should create at least one fused function"


def test_conv2d_no_fusion_case():
    """Test case where fusion should not occur"""

    class Conv2dNoBias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 3, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = Conv2dNoBias()
    graph_model = torch.fx.symbolic_trace(model)
    input_info = [([1, 3, 10, 10], "float32")]

    with torch.no_grad():
        mod = from_fx(graph_model, input_info)

    # Apply fusion pattern
    patterns = [
        (
            "conv2d_bias_activation",
            make_fused_bias_activation_pattern(
                "relax.nn.conv2d", with_bias=True, activation="relax.nn.relu", allow_reshape=True
            ),
        )
    ]

    fused_mod = relax.transform.FuseOpsByPattern(patterns, bind_constants=False)(mod)

    # No fusion should occur
    fused_functions = [name for name in fused_mod.functions.keys() if "fused" in str(name)]

    assert len(fused_functions) == 0, "No fusion should occur for conv2d without bias and relu"


if __name__ == "__main__":
    test_conv2d_bias_relu_fusion()
    test_conv2d_bias_relu_fusion_comparison()
    test_conv2d_no_fusion_case()
