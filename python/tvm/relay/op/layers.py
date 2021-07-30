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
    ht,
    ct,
    wi,
    wh,
    bi=None,
    bh=None,
    p=None,
    p_i=None,
    p_f=None,
    p_o=None,
    f_act=sigmoid,
    g_act=tanh,
    h_act=tanh,
    backwards=False,
):
    # Input hidden state shape = (batch, hidden_size)
    # wi, wh, bi, bh, proj matrix (p), peephole matrices: p_i, p_f, p_o are expected.
    # wi and wh shoud exist the others can be None

    outputs_list = []
    for x_t in input_seqs if not backwards else reversed(input_seqs):
        # x_t shape = (batch, feature size), step shape = (batch, feature size + hidden_size)
        step = concatenate([x_t, ht], axis=1)
        w = concatenate([wi, wh], axis=1)
        # Instead of nn.dense(x_t, weights[0]) + nn.dense(ht, weights[1]) we have nn.dense(step, W)
        # gates shape = (batch, 4 * hidden_size)
        gates = nn.dense(step, w)
        # Add biases
        if bi is not None:
            gates += bi
        if bh is not None:
            gates += bh
        ig, fg, cg, og = split(gates, 4, axis=-1)  # (batch, hidden_size)

        if p_i is not None and p_f is not None:
            ig = f_act(ig + p_i * ct)
            fg = f_act(fg + p_f * ct)
        else:
            ig = f_act(ig)
            fg = f_act(fg)

        cg = g_act(cg)
        ct = fg * ct + ig * cg
        if p_o is not None:
            og = f_act(og + p_o * ct)
        else:
            og = f_act(og)

        ht = og * h_act(ct)

        if p is not None:
            ht = nn.dense(ht, p)

        outputs_list.append(ht)  # [seq_num, (batch, hidden_size)]

    return outputs_list, ht, ct
