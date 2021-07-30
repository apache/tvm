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
    H_t,
    C_t,
    Wi,
    Wh,
    Bi=None,
    Bh=None,
    P=None,
    p_i=None,
    p_f=None,
    p_o=None,
    f_act=sigmoid,
    g_act=tanh,
    h_act=tanh,
    backwards=False,
):
    # Input hidden state shape = (batch, hidden_size)
    # Wi, Wh, Bi, Bh, proj matrix P, peephole matrices: p_i, p_f, p_o are expected.
    # Wi and Wh shoud exist the others can be None

    outputs_list = []
    for x_t in input_seqs if not backwards else reversed(input_seqs):
        # x_t shape = (batch, feature size), step shape = (batch, feature size + hidden_size)
        step = concatenate([x_t, H_t], axis=1)
        W = concatenate([Wi, Wh], axis=1)
        # Instead of nn.dense(x_t, weights[0]) + nn.dense(H_t, weights[1]) we have nn.dense(step, W)
        # gates shape = (batch, 4 * hidden_size)
        gates = nn.dense(step, W)
        # Add biases
        if Bi is not None:
            gates += Bi
        if Bh is not None:
            gates += Bh
        i, f, c, o = split(gates, 4, axis=-1)  # (batch, hidden_size)

        if p_i is not None and p_f is not None:
            i = f_act(i + p_i * C_t)
            f = f_act(f + p_f * C_t)
        else:
            i = f_act(i)
            f = f_act(f)

        c = g_act(c)
        C_t = f * C_t + i * c
        if p_o is not None:
            o = f_act(o + p_o * C_t)
        else:
            o = f_act(o)

        H_t = o * h_act(C_t)

        if P is not None:
            H_t = nn.dense(H_t, P)

        outputs_list.append(H_t)  # [seq_num, (batch, hidden_size)]

    return outputs_list, H_t, C_t
