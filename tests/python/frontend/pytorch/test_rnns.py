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
import onnx
import io
import sys

from tvm import relay
from tvm.contrib import graph_executor

from torch import nn

## LSTM parameters
lstm_feature_size = 16
lstm_hidden_size = 32
lstm_projection_size = 20

## GRU parameters
gru_feature_size = 8
gru_hidden_size = 16

num_layers = 2
seqs_length = 2
batch_size = 2

##RNN parameters
rnn_feature_size = 8
rnn_hidden_size = 16


class RNN_Model(nn.Module):
    """
    It is base class for RNN layer classes.
    It contains some common fields and methods for child classes.
    """

    def __init__(
        self,
    ):
        super().__init__()

        # model is defined in child class
        self.model = None

    def forward(self, input, hidden_init=None):
        """
        Computes the output tensor after input inference along RNN layer.

        :param input: batch of data as a tensor of shape (seqs_length, batch_size, feature_size) or (batch_size, seqs_length, feature_size) if self.batch_first = True
        :param hidden_init: initial hidden state(s) of the RNN as a tensor(s) of shape (num_layers, batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the output tensor of shape (batch_size, hidden_size)
        """
        if self.model is None:
            raise NotImplementedError("self.model must be defined in subclasses!")
        out, _ = self.model(input, hidden_init)

        return out

    def gen_rnd_weights(self):
        """
        Generate random weigths for the model
        """
        if self.model is None:
            raise NotImplementedError("self.model must be defined in subclasses!")
        with torch.no_grad():
            for weight_group in self.model.all_weights:
                for weight in weight_group:
                    weight.data = torch.rand(weight.shape)

    def get_dummy_inputs(self):
        raise NotImplementedError("subclasses must override get_dummy_inputs()!")

    def get_input_names(self):
        raise NotImplementedError("subclasses must override get_input_names()!")

    def get_shape_desc(self, frontend_type):
        raise NotImplementedError("subclasses must override get_shape_desc(frontend_type)!")

    def get_tvm_inputs(self, dtype):
        raise NotImplementedError("subclasses must override get_tvm_inputs(dtype)!")


class RNN_Model_Impl(RNN_Model):
    def __init__(
        self,
        seq_len=seqs_length,
        batch_size=batch_size,
        feature_size=rnn_feature_size,
        hidden_size=rnn_hidden_size,
        batch_first=False,
        layer_num=1,
        bidirectional=False,
        use_bias=True,
        rnd_weights_init=False,
        nonlinearity="tanh",
        dropout=0.0,
    ):
        super().__init__()
        # Shapes
        self.shape = [seq_len, batch_size, feature_size]
        if batch_first:
            self.shape = [batch_size, seq_len, feature_size]
        layers_num = 2 * layer_num if bidirectional else layer_num
        self.h0_shape = [layers_num, batch_size, hidden_size]
        # Dummy inputs
        self.dummy_inputs = (torch.rand(self.shape), torch.zeros(self.h0_shape))

        self.model = nn.RNN(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            nonlinearity=nonlinearity,
            bias=use_bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        if rnd_weights_init:
            self.gen_rnd_weights()

    def gen_rnd_weights(self):
        super().gen_rnd_weights()

    def get_dummy_inputs(self):
        return self.dummy_inputs

    def get_input_names(self):
        return ["input", "h0"]

    def get_shape_desc(self, frontend_type):
        shape_desc = None
        if frontend_type == "pt":  # PyTorch
            shape_desc = [("input", self.shape)]
        elif frontend_type == "onnx":  # ONNX
            shape_desc = {
                "input": self.shape,
                "h0": self.h0_shape,
            }
        return shape_desc

    def get_tvm_inputs(self, dtype):
        return {
            "input": tvm.nd.array(self.dummy_inputs[0].numpy().astype(dtype)),
            "h0": tvm.nd.array(self.dummy_inputs[1].numpy().astype(dtype)),
        }


class GRU_Model(RNN_Model):
    def __init__(
        self,
        seq_len=seqs_length,
        batch_size=batch_size,
        feature_size=gru_feature_size,
        hidden_size=gru_hidden_size,
        batch_first=False,
        layer_num=1,
        bidirectional=False,
        use_bias=True,
        rnd_weights_init=False,
    ):
        super().__init__()

        # Shapes
        self.shape = [seq_len, batch_size, feature_size]
        if batch_first:
            self.shape = [batch_size, seq_len, feature_size]
        layers_num = 2 * layer_num if bidirectional else layer_num
        self.h0_shape = [layers_num, batch_size, hidden_size]
        # Dummy inputs
        self.dummy_inputs = (torch.rand(self.shape), torch.zeros(self.h0_shape))

        self.model = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            bidirectional=bidirectional,
            batch_first=batch_first,
            bias=use_bias,
        )

        if rnd_weights_init:
            self.gen_rnd_weights()

    def gen_rnd_weights(self):
        """
        Generate random weigths for the model with biases
        For first uni- and bidirectional weights group:
            Wi (3*hidden_size, feature_size)
            Wh (3*hidden_size, hidden_size)
            Bi (3*hidden_size)
            Bh (3*hidden_size)
        For other weights group:
            Wi (3*hidden_size, hidden_size)
            Wh (3*hidden_size, hidden_size)
            Bi (3*hidden_size)
            Bh (3*hidden_size)
        For generation of random weigths for the model without biases the Bi and Bh weights are skipped
        """
        super().gen_rnd_weights()

    def get_dummy_inputs(self):
        return self.dummy_inputs

    def get_input_names(self):
        return ["input", "h0"]

    def get_shape_desc(self, frontend_type):
        shape_desc = None
        if frontend_type == "pt":  # PyTorch
            shape_desc = [("input", self.shape)]
        elif frontend_type == "onnx":  # ONNX
            shape_desc = {
                "input": self.shape,
                "h0": self.h0_shape,
            }
        return shape_desc

    def get_tvm_inputs(self, dtype):
        return {
            "input": tvm.nd.array(self.dummy_inputs[0].numpy().astype(dtype)),
            "h0": tvm.nd.array(self.dummy_inputs[1].numpy().astype(dtype)),
        }


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


class LSTM_Model(RNN_Model):
    def __init__(
        self,
        seq_len=seqs_length,
        batch_size=batch_size,
        feature_size=lstm_feature_size,
        hidden_size=lstm_hidden_size,
        batch_first=False,
        layer_num=1,
        bidirectional=False,
        proj_size=0,
        use_bias=True,
        rnd_weights_init=False,
    ):
        super().__init__()

        # Shapes
        self.shape = [seq_len, batch_size, feature_size]
        if batch_first:
            self.shape = [batch_size, seq_len, feature_size]
        layers_num = 2 * layer_num if bidirectional else layer_num
        self.h0_shape = [layers_num, batch_size, hidden_size]
        if proj_size > 0:
            self.h0_shape = [layers_num, batch_size, proj_size]
        self.c0_shape = [layers_num, batch_size, hidden_size]
        # Dummy inputs
        self.dummy_inputs = (
            torch.rand(self.shape),
            (torch.zeros(self.h0_shape), torch.zeros(self.c0_shape)),
        )

        if check_torch_version_for_proj_in_lstm():
            self.model = nn.LSTM(
                input_size=lstm_feature_size,
                hidden_size=lstm_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                proj_size=proj_size,
                batch_first=batch_first,
                bias=use_bias,
            )
        else:
            if proj_size > 0:
                print(
                    "WARNING: projection is not supported for torch version less than 1.8.0! ",
                    "LSTM was constructed without projection!",
                )
                # sys.exit()
            self.model = nn.LSTM(
                input_size=lstm_feature_size,
                hidden_size=lstm_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                batch_first=batch_first,
                bias=use_bias,
            )

        if rnd_weights_init:
            self.gen_rnd_weights()

    def gen_rnd_weights(self):
        """
        Generate random weigths for the model with biases
        Without projection:
            For first weights group:
                Wi (4*lstm_hidden_size, lstm_feature_size)
                Wh (4*lstm_hidden_size, lstm_hidden_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
            For first bidirectional weights group:
                Wi (4*lstm_hidden_size, lstm_feature_size)
                Wh (4*lstm_hidden_size, lstm_hidden_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
            For other weights group:
                Wi (4*lstm_hidden_size, lstm_hidden_size)
                Wh (4*lstm_hidden_size, lstm_hidden_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
        With projection:
            For first weights group:
                Wi (4*lstm_hidden_size, lstm_feature_size)
                Wh (4*lstm_hidden_size, proj_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
                P  (proj_size, lstm_hidden_size)
            For first bidirectional weights group:
                Wi (4*lstm_hidden_size, lstm_feature_size)
                Wh (4*lstm_hidden_size, proj_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
                P  (proj_size, lstm_hidden_size)
            For other weights group:
                Wi (4*lstm_hidden_size, proj_size * num_directions)
                Wh (4*lstm_hidden_size, proj_size)
                Bi (4*lstm_hidden_size)
                Bh (4*lstm_hidden_size)
                P  (proj_size, lstm_hidden_size)
        For generation of random weigths for the model without biases Bi and Bh are skipped
        """
        super().gen_rnd_weights()

    def get_dummy_inputs(self):
        return self.dummy_inputs

    def get_input_names(self):
        return ["input", "h0", "c0"]

    def get_shape_desc(self, frontend_type):
        shape_desc = None
        if frontend_type == "pt":  # PyTorch
            shape_desc = [("input", self.shape)]
        elif frontend_type == "onnx":  # ONNX
            shape_desc = {
                "input": self.shape,
                "h0": self.h0_shape,
                "c0": self.c0_shape,
            }
        return shape_desc

    def get_tvm_inputs(self, dtype):
        return {
            "input": tvm.nd.array(self.dummy_inputs[0].numpy().astype(dtype)),
            "h0": tvm.nd.array(self.dummy_inputs[1][0].numpy().astype(dtype)),
            "c0": tvm.nd.array(self.dummy_inputs[1][1].numpy().astype(dtype)),
        }


def compare(input, gold_data, rtol=1e-5, atol=1e-5):
    tvm.testing.assert_allclose(input, gold_data, rtol=rtol, atol=atol)


def check_rnn(rnn_type, rnn_mod, target=tvm.target.Target("llvm -mcpu=core-avx2"), dev=tvm.cpu(0)):
    def get_model(
        rnn_type,
        rnn_mod,
        args,
    ):
        # Fill args
        if "b" in rnn_mod:
            args["bidirectional"] = True
        if "s" in rnn_mod:
            args["layer_num"] = num_layers
        if "tanh" in rnn_mod:
            args["nonlinearity"] = "tanh"
        if "relu" in rnn_mod:
            args["nonlinearity"] = "relu"

        if rnn_type == "GRU":
            RNN_Model_selector = GRU_Model
        elif rnn_type == "LSTM":
            RNN_Model_selector = LSTM_Model
            if "p" in rnn_mod:
                args["proj_size"] = lstm_projection_size
        elif rnn_type == "RNN":
            RNN_Model_selector = RNN_Model_Impl

        return RNN_Model_selector(**args)

    def get_onnx_model(model):
        onnx_io = io.BytesIO()
        with torch.no_grad():
            input_names = model.get_input_names()
            inputs = model.get_dummy_inputs()

            # default export (without dynamic input)
            torch.onnx.export(model, inputs, onnx_io, input_names=input_names)

        onnx_io.seek(0, 0)
        return onnx.load_model(onnx_io)

    model = None
    dtype = "float32"
    device = torch.device("cpu")
    for batch_first in (True, False):
        for use_bias in (True, False):
            for rnd_weights in [True]:  # (True, False):
                model_inputs = {
                    "batch_first": batch_first,
                    "use_bias": use_bias,
                    "rnd_weights_init": rnd_weights,
                }
                model = get_model(rnn_type, rnn_mod, model_inputs)
                model.to(device)
                model.eval()

                # Get golden output from original model
                dummy_inputs = model.get_dummy_inputs()
                golden_output = model.forward(dummy_inputs[0].to(device)).detach().cpu().numpy()

                tvm_output = None
                for format in ["pt"]:  # ["pt", "onnx"]:
                    shape_desc = model.get_shape_desc(format)
                    if format == "pt":
                        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
                        traced_script_module = torch.jit.trace(model, dummy_inputs[0]).eval()

                        # Import model to Relay
                        with tvm.testing.disable_span_filling():
                            mod, params = relay.frontend.from_pytorch(
                                traced_script_module, shape_desc
                            )
                        with tvm.testing.enable_span_filling():
                            mod_with_span, _ = relay.frontend.from_pytorch(
                                traced_script_module, shape_desc
                            )
                        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)
                    elif format == "onnx":
                        try:
                            onnx_model = get_onnx_model(model)
                        except:
                            print(
                                "WARNING: torch.onnx.export does not support conversion LSTM with projection "
                                "from pytorch! TODO: waiting for the support and correct test after that."
                            )
                            continue

                        # Import model to Relay
                        with tvm.testing.disable_span_filling():
                            mod, params = relay.frontend.from_onnx(onnx_model, shape_desc)
                        with tvm.testing.enable_span_filling():
                            mod_with_span, _ = relay.frontend.from_onnx(onnx_model, shape_desc)
                        assert tvm.ir.structural_equal(mod, mod_with_span, map_free_vars=True)

                    # Model compilation by tvm
                    with tvm.transform.PassContext(opt_level=3):
                        lib = relay.build(mod, target=target, params=params)

                    # Inference of the model with given input data
                    m = graph_executor.GraphModule(lib["default"](dev))

                    # Set inputs
                    tvm_inputs = model.get_tvm_inputs(dtype)
                    m.set_input(**tvm_inputs)
                    # Execute
                    m.run()
                    # Get outputs (converted to numpy array)
                    tvm_output = m.get_output(0).numpy()

                    compare(tvm_output, golden_output)


@tvm.testing.uses_gpu
def test_rnns():
    for target, dev in tvm.testing.enabled_targets():
        # RNN types: GRU, LSTM
        # GRU modifications: unidirectional, stacked, bidirectional, stacked bidirectional
        for mod_type in ["uni", "s", "b", "sb"]:
            check_rnn("GRU", mod_type, target, dev)
        # LSTM modifications: unidirectional, stacked, bidirectional, stacked bidirectional,
        # and all these types with projection ("p", "sp", "bp", "sbp")
        # The latter are skiped for test acceleration
        for mod_type in ["uni", "s", "b", "sb"]:
            check_rnn("LSTM", mod_type, target, dev)

        for mod_type in ["uni", "s", "b", "sb", "tanh", "relu"]:
            check_rnn("RNN", mod_type, target, dev)


if __name__ == "__main__":
    test_rnns()
