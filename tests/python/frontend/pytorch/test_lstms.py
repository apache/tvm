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
import numpy as np
import torch
import onnx
import argparse
import sys
import shutil

from tvm import relay
from tvm.contrib import graph_executor

from pathlib import Path
from torch import nn

## Model parameters
model_feature_size = 5
model_hidden_size = 10
model_num_layers = 2
seqs_length = 15
projection_size = 7
batch_size = 3


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
        rnd_weights_init=False,
    ):
        super().__init__()

        self.device = device
        self.batch_first = batch_first

        # Network defition
        if check_torch_version_for_proj_in_lstm():
            self.lstm = nn.LSTM(
                input_size=model_feature_size,
                hidden_size=model_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                proj_size=proj_size,
                batch_first=batch_first,
            ).to(device)
        else:
            if proj_size > 0:
                print("WARNING: projection is not supported for torch version less than 1.8.0!")
            self.lstm = nn.LSTM(
                input_size=model_feature_size,
                hidden_size=model_hidden_size,
                num_layers=layer_num,
                bidirectional=bidirectional,
                batch_first=batch_first,
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
        Generate random weigths for the model
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


def compare(input, gold_data, epsilon=1e-6):
    remain = np.abs(gold_data - input)
    err = np.max(remain)
    if err < epsilon:
        print("SUCCESS: RESULTS ARE THE SAME WITH MAX ERROR {} AND EPSILON {}".format(err, epsilon))
    else:
        print("WARNING: RESULTS ARE NOT THE SAME WITH ERROR {}".format(err))


if __name__ == "__main__":

    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="It constructs neural network which conists of LSTM layer only. "
        "But the layer can be different types and adjusted by special parameters (see https://pytorch.org/docs/1.8.0/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM). "
        "There are several types: unidirectional, bidirectional, projection, stacked, stacked bidirectional, stacked projection bidirectional",
        formatter_class=MyFormatter,
    )
    parser.add_argument(
        "-t",
        "--lstm_type",
        type=str,
        default="uni",
        help="Type of lstm layer for test. There are several options: uni (unidirectional), b (bidirectional), "
        "p (with projection), s (stacked), sb (stacked bidirectional), sp (stacked with projection), "
        "bp (bidirectional with projection), sbp (stacked bidirectional with projection)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="ts",
        help='Format of the model. There are two options: "ts"(TorchScript) and "onnx"(ONNX). The first one is used by default',
    )
    parser.add_argument(
        "-l",
        "--layer_num",
        type=int,
        default=model_num_layers,
        help="Number of LSTM layers. It is useded for stacked LSTM",
    )
    parser.add_argument(
        "-p",
        "--projection_size",
        type=int,
        default=projection_size,
        help="Projection size is used in LSTM with projection",
    )
    parser.add_argument(
        "-b",
        "--batch_first",
        action="store_true",
        default=False,
        help="Batch first parameter used for LSTM layer initialization",
    )
    parser.add_argument(
        "-w",
        "--rnd_weights",
        action="store_true",
        default=False,
        help="Generate random weights and biases for the model. NOTE: By default All the weights and biases are initialized from "
        "\mathcal{U}(-\sqrt{k}, \sqrt{k}), where k = \frac{1}{\text{hidden\_size}}",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to directory for saving of intermediate results. NOTE: At the end the directory is removed with all dependencies inside",
    )

    args = parser.parse_args()
    if not hasattr(args, "out_dir"):
        args.out_dir = Path.cwd().joinpath("output")
        args.out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cpu")
    hidden_layers_num = 1
    model = None
    if args.lstm_type == "uni":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            rnd_weights_init=args.rnd_weights,
        )
    elif args.lstm_type == "b":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            bidirectional=True,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = 2
    elif args.lstm_type == "p":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            proj_size=args.projection_size,
            rnd_weights_init=args.rnd_weights,
        )
    elif args.lstm_type == "s":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            layer_num=args.layer_num,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = args.layer_num
    elif args.lstm_type == "sb":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            bidirectional=True,
            layer_num=args.layer_num,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = 2 * args.layer_num
    elif args.lstm_type == "sp":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            layer_num=args.layer_num,
            proj_size=args.projection_size,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = args.layer_num
    elif args.lstm_type == "bp":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            bidirectional=True,
            proj_size=args.projection_size,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = 2 * args.layer_num
    elif args.lstm_type == "sbp":
        model = LSTM_Model(
            device,
            batch_first=args.batch_first,
            bidirectional=True,
            layer_num=args.layer_num,
            proj_size=args.projection_size,
            rnd_weights_init=args.rnd_weights,
        )
        hidden_layers_num = 2 * args.layer_num
    else:
        print("LSTM type {} is not supported here!".format(args.lstm_type))
        sys.exit()

    model.eval()

    # Get golden output from original model
    input_hidden_shape = (hidden_layers_num, batch_size, model_hidden_size)
    dummy_input, input_shape = model.get_dummy_input()
    golden_output_batch = model.forward(dummy_input.to(device)).detach().cpu().numpy()

    dtype = "float32"
    h_zeros = np.zeros(input_hidden_shape, dtype=dtype)
    c_zeros = np.zeros(input_hidden_shape, dtype=dtype)

    tvm_output = None
    if args.format == "ts":
        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module = torch.jit.trace(model, dummy_input).eval()

        # Import model to Relay
        shape_list = [("input", input_shape)]
        mod, params = relay.frontend.from_pytorch(traced_script_module, shape_list)

        # Model compilation by tvm
        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
    elif args.format == "onnx":
        onnx_fpath = args.out_dir.joinpath("model_{}.onnx".format(args.lstm_type))

        with torch.no_grad():
            h0 = torch.rand(input_hidden_shape)
            c0 = torch.rand(input_hidden_shape)
            input_names = ["input", "h0", "c0"]

            # default export (without dynamic input)
            torch.onnx.export(model, (dummy_input, (h0, c0)), onnx_fpath, input_names=input_names)

        onnx_model = onnx.load(onnx_fpath)

        # Import model to Relay
        shape_dict = {"input": input_shape, "h0": input_hidden_shape, "c0": input_hidden_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        # Model compilation by tvm
        target = "llvm"
        dev = tvm.cpu(0)
        with tvm.transform.PassContext(opt_level=1):
            lib = relay.build(mod, target=target, params=params)
    else:
        print("ERROR: {} format is unsupported".format(args.format))

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

    # Remove output directory with tmp files
    shutil.rmtree(args.out_dir)
