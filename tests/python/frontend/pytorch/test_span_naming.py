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
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, cell-var-from-loop, unnecessary-lambda
# pylint: disable=missing-function-docstring, redefined-builtin, use-implicit-booleaness-not-comparison
"""Tests to ensure span names are correctly populated when importing Pytorch"""
from torch import nn
import torch
import tvm


class NestedConvModule(nn.Module):
    """Module that performs Conv2d and relu activation"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class NestedFinalModule(nn.Module):
    """Simple module that adds 2 inputs"""

    def forward(self, x, y):
        return x + y


class SimpleTwoConvModule(nn.Module):
    """
    ML model that performs 2 convolutions and adds them together.
    All operations are inside nested modules to make scope names interesting.
    """

    def __init__(self):
        super().__init__()
        # First convolutional module
        self.image_block1 = NestedConvModule(in_channels=3, out_channels=64)
        # Second convolutional module
        self.image_block2 = NestedConvModule(in_channels=64, out_channels=64)
        self.final_block = NestedFinalModule()

    def forward(self, x):
        # Forward pass through the first convolutional module
        x1 = self.image_block1(x)
        # Forward pass through the second convolutional module
        x2 = self.image_block2(x1)
        # Add the outputs of the two convolutional modules
        return self.final_block(x1, x2)


def test_pytorch_scope_based_span_names():
    model = SimpleTwoConvModule()
    sample_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        traced_torch_model = torch.jit.trace(model, sample_input)
    import_input = [("model_input", (1, 3, 64, 64))]
    relay_model_ir, relay_model_params = tvm.relay.frontend.from_pytorch(
        traced_torch_model, import_input, preserve_pytorch_scopes=True
    )
    # If specified, we are preserving the pytorch named spans
    for block in [1, 2]:
        for key in ["weight", "bias"]:
            assert f"image_block{block}.conv.{key}" in relay_model_params.keys()
    # Manually check all span names since asserting structural equality is not sufficient
    current_call = relay_model_ir["main"].body
    assert current_call.op.name == "add"
    assert current_call.span is not None and current_call.span.source_name.name == "final_block"
    current_call = current_call.args[1]
    for block in [2, 1]:
        assert current_call.op.name == "nn.relu"
        assert (
            current_call.span is not None
            and current_call.span.source_name.name == f"image_block{block}.relu"
        )
        current_call = current_call.args[0]
        assert current_call.op.name == "nn.bias_add"
        assert (
            current_call.span is not None
            and current_call.span.source_name.name == f"image_block{block}.conv"
        )
        current_call = current_call.args[0]
        assert current_call.op.name == "nn.conv2d"
        assert (
            current_call.span is not None
            and current_call.span.source_name.name == f"image_block{block}.conv"
        )
        current_call = current_call.args[0]
