import torch.nn as nn
import torch
import pytest
import tvm


class NestedConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class SimpleTwoConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional module
        self.image_block1 = NestedConvModule(in_channels=3, out_channels=64)
        # Second convolutional module
        self.image_block2 = NestedConvModule(in_channels=64, out_channels=64)

    def forward(self, x):
        # Forward pass through the first convolutional module
        x1 = self.image_block1(x)
        # Forward pass through the second convolutional module
        x2 = self.image_block2(x1)
        # Add the outputs of the two convolutional modules
        output = x1 + x2
        return output


@pytest.fixture
def traced_model_and_inputs():
    model = SimpleTwoConvModule()
    sample_input = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        traced_torch_model = torch.jit.trace(model, sample_input)
    import_input = [("model_input", (1, 3, 64, 64))]
    return traced_torch_model, import_input


def test_default_span_names(traced_model_and_inputs):
    traced_torch_model, import_input = traced_model_and_inputs
    relay_model_ir, relay_model_params = tvm.relay.frontend.from_pytorch(
        traced_torch_model, import_input
    )
    # By default, we assign new names based on the op kind
    import pdb
    pdb.set_trace()


def test_pytorch_scope_based_span_names(traced_model_and_inputs):
    traced_torch_model, import_input = traced_model_and_inputs
    relay_model_ir, relay_model_params = tvm.relay.frontend.from_pytorch(
        traced_torch_model, import_input, preserve_pytorch_scopes=True
    )
    # If specified, we are preserving the pytorch named
    import pdb
    pdb.set_trace()
