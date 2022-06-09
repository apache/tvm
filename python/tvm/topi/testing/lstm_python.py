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
# pylint: disable=invalid-name
"""LSTM reference implementation using numpy."""
import numpy as np


def lstm_python(
    Xs: np.array,
    Wi: np.array,
    Wh: np.array,
    Bi: np.array = None,
    Bh: np.array = None,
    h_init: np.array = None,
    c_init: np.array = None,
    proj: np.array = None,
    p_i: np.array = None,
    p_f: np.array = None,
    p_o: np.array = None,
    f_act: str = "sigmoid",
    g_act: str = "tanh",
    h_act: str = "tanh",
    reverse: bool = False,
    weight_layout: str = "IFGO",
):
    """LSTM reference implementation using numpy

    Parameters
    ----------
    Xs : np.array
        (seq_length, batch_size, in_dim)
    Wi : np.array
        (4 * hidden_dim, in_dim)
    Wh : np.array
        (4 * hidden_dim, out_dim) where out_dim = proj_dim if proj_dim > 0, else hidden_dim
    Bi : np.array, optional
        (4 * hidden_dim,), by default None
    Bh : np.array, optional
        (4 * hidden_dim,), by default None
    h_init : np.array, optional
        (batch_size, out_dim), by default None
    c_init : np.array, optional
        (batch_size, hidden_dim), by default None
    proj : np.array, optional
        (proj_dim, hidden_dim), by default None
    p_i, p_f, p_o: np.array, optional
        (batch_size, hidden_dim), by default None
    f_act, g_act, h_act: str, optional
        activations, by default "sigmoid", "tanh", "tanh"
    reverse : bool, optional
        process Xs in reverse, by default False
    weight_layout : str, optional
        Packed layout for weights and biases, by default "IFGO"
    """
    i_gate_idx = weight_layout.find("I")
    f_gate_idx = weight_layout.find("F")
    g_gate_idx = weight_layout.find("G")
    o_gate_idx = weight_layout.find("O")

    str2act = {"sigmoid": lambda x: 1 / (1 + np.exp(-x)), "tanh": np.tanh}

    f_act = str2act[f_act]
    g_act = str2act[g_act]
    h_act = str2act[h_act]

    S, B, F = Xs.shape
    H = Wi.shape[0] // 4
    O = Wh.shape[1]

    # make life a bit easier
    Wi = np.reshape(Wi, (4, H, F))
    Wh = np.reshape(Wh, (4, H, O))
    if Bi is not None:
        Bi = np.reshape(Bi, (4, H))
    if Bh is not None:
        Bh = np.reshape(Bh, (4, H))

    h0 = h_init if h_init is not None else np.zeros((B, O), "float32")
    c0 = c_init if c_init is not None else np.zeros((B, H), "float32")

    hs = [h0]
    cs = [c0]

    for t in range(S):
        x = Xs[S - t - 1 if reverse else t]
        xh = [np.matmul(x, Wi[g].T) for g in range(4)]
        if Bi is not None:
            xh = [xh[g] + Bi[g] for g in range(4)]

        hh = [np.matmul(hs[t], Wh[g].T) for g in range(4)]
        if Bh is not None:
            hh = [hh[g] + Bh[g] for g in range(4)]

        sums = [xh[g] + hh[g] for g in range(4)]

        if p_i is not None and p_f is not None:
            i_gate = f_act(sums[i_gate_idx] + p_i * cs[t])
            f_gate = f_act(sums[f_gate_idx] + p_f * cs[t])
        else:
            i_gate = f_act(sums[i_gate_idx])
            f_gate = f_act(sums[f_gate_idx])

        g_gate = g_act(sums[g_gate_idx])

        next_c = f_gate * cs[t] + i_gate * g_gate

        if p_o is not None:
            o_gate = f_act(sums[o_gate_idx] + p_o * next_c)
        else:
            o_gate = f_act(sums[o_gate_idx])

        next_h = o_gate * h_act(next_c)

        if proj is not None:
            next_h = np.matmul(next_h, proj.T)

        hs.append(next_h)
        cs.append(next_c)

    return np.stack(hs[1:], axis=0), np.stack(cs[1:], axis=0)
