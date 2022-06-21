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
"""Test code for LSTM."""
import numpy as np
from rsa import verify
import tvm
from tvm import te, topi
import tvm.testing
import tvm.topi.testing


def verify_lstm(
    target,
    dev,
    seq_len,
    batch_size,
    in_dim,
    hidden_dim,
    proj_dim=0,
    bias=True,
    zero_init=True,
    peephole=False,
    reverse=False,
    weight_layout="IFGO",
):
    out_dim = proj_dim if proj_dim > 0 else hidden_dim

    def rand(*shape):
        sqrt_k = np.sqrt(1 / hidden_dim)
        return np.random.uniform(-sqrt_k, sqrt_k, size=shape).astype("float32")

    def get_ref_data():
        Xs = np.random.normal(size=(seq_len, batch_size, in_dim)).astype("float32")
        Wi = rand(4 * hidden_dim, in_dim)
        Wh = rand(4 * hidden_dim, out_dim)
        Bi = None
        Bh = None
        h0 = None
        c0 = None
        proj = None
        p_i = None
        p_f = None
        p_o = None

        if bias:
            Bi = rand(4 * hidden_dim)
            Bh = rand(4 * hidden_dim)

        if not zero_init:
            h0 = np.random.normal(size=(batch_size, out_dim)).astype("float32")
            c0 = np.random.normal(size=(batch_size, hidden_dim)).astype("float32")

        if proj_dim > 0:
            proj = rand(proj_dim, hidden_dim)

        if peephole:
            p_i, p_f, p_o = [rand(batch_size, hidden_dim) for _ in range(3)]

        hs, cs = tvm.topi.testing.lstm_python(
            Xs,
            Wi,
            Wh,
            Bi=Bi,
            Bh=Bh,
            h_init=h0,
            c_init=c0,
            proj=proj,
            p_i=p_i,
            p_f=p_f,
            p_o=p_o,
            reverse=reverse,
            weight_layout=weight_layout,
        )

        return [Xs, Wi, Wh, Bi, Bh, h0, c0, proj, p_i, p_f, p_o], [hs, cs]

    args_np, (hs_np, cs_np) = get_ref_data()

    args = [te.placeholder(a.shape, "float32") if a is not None else a for a in args_np]
    real_args = [a for a in args if a is not None]

    hs, cs = topi.nn.lstm(*args, reverse=reverse, weight_layout=weight_layout)
    with tvm.target.Target(target):
        sch = topi.generic.schedule_lstm([hs, cs])
    func = tvm.build(sch, real_args + [hs, cs], target=target)

    args_nd = [tvm.nd.array(a, dev) for a in args_np if a is not None]
    hs_nd = tvm.nd.array(np.zeros((seq_len, batch_size, out_dim), "float32"), dev)
    cs_nd = tvm.nd.array(np.zeros((seq_len, batch_size, hidden_dim), "float32"), dev)
    func(*args_nd, hs_nd, cs_nd)

    tvm.testing.assert_allclose(hs_nd.numpy(), hs_np, rtol=1e-4)
    tvm.testing.assert_allclose(cs_nd.numpy(), cs_np, rtol=1e-4)


def test_lstm():
    verify_lstm(
        "llvm",
        tvm.cpu(0),
        1,
        1,
        1,
        1,
        0,
        True,
        True,
        False,
        False,
        "IFGO",
    )

    verify_lstm(
        "llvm",
        tvm.cpu(0),
        8,
        4,
        8,
        16,
        0,
        True,
        False,
        False,
        False,
        "IFGO",
    )


def test_lstm_proj():
    verify_lstm("llvm", tvm.cpu(0), 8, 4, 16, 32, 8, True, True, False, False, "IFGO")


def test_lstm_peephole():
    verify_lstm("llvm", tvm.cpu(0), 8, 4, 16, 32, 0, True, True, True, False, "IFGO")


def test_lstm_reverse():
    verify_lstm("llvm", tvm.cpu(0), 8, 4, 16, 32, 0, True, True, False, True, "IFGO")


def test_lstm_weight_layout_iofg():
    # IOFG is used by ONNX, while IFGO is used by PyTorch
    verify_lstm("llvm", tvm.cpu(0), 8, 4, 16, 32, 0, True, True, False, False, "IOFG")


def test_lstm_assorted():
    verify_lstm("llvm", tvm.cpu(0), 8, 4, 16, 32, 16, True, False, True, True, "OIGF")
