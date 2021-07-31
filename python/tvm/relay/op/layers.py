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

"""Intermediate representation of complicated layers unified for all frontends"""

from .tensor import sigmoid, tanh, concatenate
from .transform import split
from . import nn


def lstm_cell(
    input_seqs,
    hidden_state,
    cell_state,
    w_inp,
    w_hid,
    b_inp=None,
    b_hid=None,
    proj=None,
    p_i=None,
    p_f=None,
    p_o=None,
    f_act=sigmoid,
    g_act=tanh,
    h_act=tanh,
    backwards=False,
):
    """
    Common implementation of LSTM cell for all frontends of TVM
    TODO (vvchernov): currently it is used by onnx and pytorch.

    Parameters
    ----------
    input_seqs : List[relay.Expr]
        The sequence of input tensors
        Input tensor should be 2d while issue #8412 is not resolved
        Shape = (batch, feature_size)
    hidden_state : relay.Expr
        Hidden state. shape = (batch, hidden_size)
    cell_state : relay.Expr
        Cell state. shape = (batch, hidden_size)
    w_inp, w_hid : relay.Expr
        weight matrices. wi shape = (4 * hidden_size, feature_size)
        wh shape = (4 * hidden_size, hidden_size or proj_size)
        NOTE: wi = (w_ii|w_if|w_ig|w_io) for input, forget, cell and output gates.
        The order is important for correct LSTM calculation!
    b_inp, b_hid : relay.Expr
        bias matrices. The same order of internal parts as for weights. shape = (4 * hidden_size)
    proj : relay.Expr
        projection matrix. shape = (proj_size, hidden_size)
    p_i, p_f, p_o : relay.Expr
        peephole LSTM matrices. shape = (batch, hidden_size)
    f_act, g_act, h_act : relay.op
        activation funtions
    backwards : bool
        Flag for reverse pass of LSTM

    Returns
    -------
    result : List[relay.Expr], relay.Expr, relay.Expr
        The sequence of computed result, final hidden and cell state
    """

    outputs_list = []
    for x_t in input_seqs if not backwards else reversed(input_seqs):
        # x_t shape = (batch, feature size), step shape = (batch, feature size + hidden_size)
        step = concatenate([x_t, hidden_state], axis=1)
        cat_w = concatenate([w_inp, w_hid], axis=1)
        # Instead of nn.dense(x_t, w_inp) + nn.dense(hidden_state, w_hid)
        # the nn.dense(step, cat_w) is used
        # gates shape = (batch, 4 * hidden_size)
        gates = nn.dense(step, cat_w)
        # Add biases
        if b_inp is not None:
            gates += b_inp
        if b_hid is not None:
            gates += b_hid
        inp_gate, fgt_gate, cell_gate, otp_gate = split(gates, 4, axis=-1)  # (batch, hidden_size)

        if p_i is not None and p_f is not None:
            inp_gate = f_act(inp_gate + p_i * cell_state)
            fgt_gate = f_act(fgt_gate + p_f * cell_state)
        else:
            inp_gate = f_act(inp_gate)
            fgt_gate = f_act(fgt_gate)

        cell_gate = g_act(cell_gate)
        cell_state = fgt_gate * cell_state + inp_gate * cell_gate
        if p_o is not None:
            otp_gate = f_act(otp_gate + p_o * cell_state)
        else:
            otp_gate = f_act(otp_gate)

        hidden_state = otp_gate * h_act(cell_state)

        if proj is not None:
            hidden_state = nn.dense(hidden_state, proj)

        outputs_list.append(hidden_state)  # [seq_num, (batch, hidden_size)]

    return outputs_list, hidden_state, cell_state
