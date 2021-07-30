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

import tvm
import tvm.testing
import numpy as np
import torch
import onnx
import io
import sys
import pytest

from tvm import relay
from tvm.contrib import graph_executor

from torch import nn

## Model parameters
model_feature_size = 16
model_hidden_size = 32
model_num_layers = 2
seqs_length = 2
projection_size = 20
batch_size = 2


def check_torch_version_for_proj_in_lstm():
    """
    proj_size parameter is supported in torch.nn.LSTM layer started from 1.8.0 torch version
    """
    me = False

    version = torch.__version__
    major, minor, micro = version.split(".")

    if int(major) > 1:
        me = True
    elif int(major) == 1:
        if int(minor) >= 8:
            me = True

    return me


class LSTM_Model(nn.Module):
    def __init__(
        self,
        device,
        batch_first=False,
        layer_num=1,
        bidirectional=False,
        proj_size=0,
        use_bias=True,
        rnd_weights_init=False,
    ):
        super().__init__()

        self.device = device
        self.batch_first = batch_first
        self.use_bias = use_bias

        if check_torch_version_for_proj_in_lstm():
            self.lstm = nn.LSTM(
                input_size=model_feature_size,
                hidden_size=model_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                proj_size=proj_size,
                batch_first=batch_first,
                bias=use_bias,
            ).to(device)
        else:
            if proj_size > 0:
                print(
                    "WARNING: projection is not supported for torch version less than 1.8.0! ",
                    "LSTM was constructed without projection!",
                )
                # sys.exit()
            self.lstm = nn.LSTM(
                input_size=model_feature_size,
                hidden_size=model_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                batch_first=batch_first,
                bias=use_bias,
            ).to(device)

        if rnd_weights_init:
            self.gen_rnd_weights()

    def forward(self, input, hidden_init=None):
        """
        Computes the output tensor after input inference along LSTM layer.

        :param input: batch of data as a tensor of shape (seqs_length, batch_size, model_feature_size) or (batch_size, seqs_length, model_feature_size) if self.batch_first = True
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the output tensor of shape (batch_size, model_hidden_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(input, hidden_init)

        return out

    def gen_rnd_weights(self):
        """
        Generate random weigths for the model with biases
        Without projection:
            For first weights group:
                Wi (4*model_hidden_size, model_feature_size)
                Wh (4*model_hidden_size, model_hidden_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
            For first bidirectional weights group:
                Wi (4*model_hidden_size, model_feature_size)
                Wh (4*model_hidden_size, model_hidden_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
            For other weights group:
                Wi (4*model_hidden_size, model_hidden_size)
                Wh (4*model_hidden_size, model_hidden_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
        With projection:
            For first weights group:
                Wi (4*model_hidden_size, model_feature_size)
                Wh (4*model_hidden_size, proj_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
                P  (proj_size, model_hidden_size)
            For first bidirectional weights group:
                Wi (4*model_hidden_size, model_feature_size)
                Wh (4*model_hidden_size, proj_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
                P  (proj_size, model_hidden_size)
            For other weights group:
                Wi (4*model_hidden_size, proj_size * num_directions)
                Wh (4*model_hidden_size, proj_size)
                Bi (4*model_hidden_size)
                Bh (4*model_hidden_size)
                P  (proj_size, model_hidden_size)
        For generation of random weigths for the model without biases Bi and Bh are skipped
        """
        for weight_group in self.lstm.all_weights:
            for weight in weight_group:
                weight.data = torch.rand(weight.shape)

    def get_dummy_input(self):
        shape = [seqs_length, batch_size, model_feature_size]
        if self.batch_first:
            shape = [batch_size, seqs_length, model_feature_size]
        res = torch.rand(shape)

        return res, shape


def compare(input, gold_data, rtol=1e-5, atol=1e-5):
    tvm.testing.assert_allclose(input, gold_data, rtol=rtol, atol=atol)


def check_lstm_with_type(
    lstm_type, target=tvm.target.Target("llvm -mcpu=core-avx2"), dev=tvm.cpu(0)
):
    has_proj = "p" in lstm_type

    device = torch.device("cpu")
    hidden_layers_num = 1
    model = None
    for batch_first in (True, False):
        for use_bias in (True, False):
            for rnd_weights in [True]:  # (True, False):
                if lstm_type == "uni":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                elif lstm_type == "b":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        bidirectional=True,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = 2
                elif lstm_type == "p":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        proj_size=projection_size,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                elif lstm_type == "s":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        layer_num=model_num_layers,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = model_num_layers
                elif lstm_type == "sb":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        bidirectional=True,
                        layer_num=model_num_layers,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = 2 * model_num_layers
                elif lstm_type == "sp":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        layer_num=model_num_layers,
                        proj_size=projection_size,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = model_num_layers
                elif lstm_type == "bp":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        bidirectional=True,
                        proj_size=projection_size,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = 2
                elif lstm_type == "sbp":
                    model = LSTM_Model(
                        device,
                        batch_first=batch_first,
                        bidirectional=True,
                        layer_num=model_num_layers,
                        proj_size=projection_size,
                        rnd_weights_init=rnd_weights,
                        use_bias=use_bias,
                    )
                    hidden_layers_num = 2 * model_num_layers
                else:
                    print("WARNING: LSTM type {} is not supported here!".format(lstm_type))
                    return

                model.eval()

                # Get golden output from original model
                input_hidden_shape = (hidden_layers_num, batch_size, model_hidden_size)
                input_hidden_shape_with_proj = (hidden_layers_num, batch_size, projection_size)
                dummy_input, input_shape = model.get_dummy_input()
                golden_output_batch = model.forward(dummy_input.to(device)).detach().cpu().numpy()

                dtype = "float32"
                h_zeros = np.zeros(input_hidden_shape, dtype=dtype)
                if has_proj:
                    h_zeros = np.zeros(input_hidden_shape_with_proj, dtype=dtype)
                c_zeros = np.zeros(input_hidden_shape, dtype=dtype)

                tvm_output = None
                for format in ["ts"]:  # ["ts", "onnx"]:
                    if format == "ts":
                        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
                        traced_script_module = torch.jit.trace(model, dummy_input).eval()

                        # Import model to Relay
                        shape_list = [("input", input_shape)]
                        mod, params = relay.frontend.from_pytorch(traced_script_module, shape_list)

                        # Model compilation by tvm
                        with tvm.transform.PassContext(opt_level=3):
                            lib = relay.build(mod, target=target, params=params)
                    elif format == "onnx":
                        if has_proj:
                            print(
                                "WARNING: torch.onnx.export does not support conversion LSTM with projection "
                                "from pytorch! TODO: waiting for the support and correct test after that."
                            )
                            continue
                        onnx_io = io.BytesIO()
                        with torch.no_grad():
                            h0 = torch.rand(input_hidden_shape)
                            if has_proj:
                                h0 = torch.rand(input_hidden_shape_with_proj)
                            c0 = torch.rand(input_hidden_shape)
                            input_names = ["input", "h0", "c0"]

                            # default export (without dynamic input)
                            torch.onnx.export(
                                model, (dummy_input, (h0, c0)), onnx_io, input_names=input_names
                            )
                        onnx_io.seek(0, 0)
                        onnx_model = onnx.load_model(onnx_io)

                        # Import model to Relay
                        shape_dict = {
                            "input": input_shape,
                            "h0": input_hidden_shape,
                            "c0": input_hidden_shape,
                        }
                        if has_proj:
                            shape_dict = {
                                "input": input_shape,
                                "h0": input_hidden_shape_with_proj,
                                "c0": input_hidden_shape,
                            }
                        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

                        # Model compilation by tvm
                        with tvm.transform.PassContext(opt_level=1):
                            lib = relay.build(mod, target=target, params=params)

                    # Inference of the model with given input data
                    m = graph_executor.GraphModule(lib["default"](dev))

                    # Set inputs
                    m.set_input(
                        input=tvm.nd.array(dummy_input.numpy().astype(dtype)),
                        h0=tvm.nd.array(h_zeros),
                        c0=tvm.nd.array(c_zeros),
                    )
                    # Execute
                    m.run()
                    # Get outputs (converted to numpy array)
                    tvm_output = m.get_output(0).numpy()

                    compare(tvm_output, golden_output_batch)


@tvm.testing.uses_gpu
def test_lstms():
    for target, dev in tvm.testing.enabled_targets():
        check_lstm_with_type("uni", target, dev)
        # check_lstm_with_type("p", target, dev)
        check_lstm_with_type("s", target, dev)
        check_lstm_with_type("b", target, dev)
        # check_lstm_with_type("bp", target, dev)
        # check_lstm_with_type("sp", target, dev)
        check_lstm_with_type("sb", target, dev)
        # check_lstm_with_type("sbp", target, dev)


if __name__ == "__main__":
    test_lstms()
