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
from tvm import te


def test_lstm_cell_inline():
    num_step = 128
    num_input = 256
    num_hidden = 1152
    batch_size = 4
    # Global transition matrix
    X = te.placeholder((num_step - 1, batch_size, num_input), name="X")
    Wi2h = te.placeholder((4, num_hidden, num_input), name="Wi2h")
    Wh2h = te.placeholder((4, num_hidden, num_hidden), name="Wh2h")
    # h: output hidden state, c: cell state.
    s_state_h = te.placeholder((num_step, batch_size, num_hidden))
    s_state_c = te.placeholder((num_step, batch_size, num_hidden))
    s_init_c = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_c")
    s_init_h = te.compute((1, batch_size, num_hidden), lambda *i: 0.0, name="init_h")
    # LSTM transition
    k = te.reduce_axis((0, num_input), name="ki2h")
    s_i2h = te.compute(
        (num_step, 4, batch_size, num_hidden),
        lambda t, x, i, j: te.sum(X[t - 1, i, k] * Wi2h[x, j, k], axis=k),
        name="s_i2h",
    )
    k = te.reduce_axis((0, num_hidden), name="ki2h")
    s_h2h = te.compute(
        (num_step, 4, batch_size, num_hidden),
        lambda t, x, i, j: te.sum(s_state_h[t - 1, i, k] * Wh2h[x, j, k], axis=k),
        name="s_h2h",
    )
    # Gate rules
    gates = te.compute(s_i2h.shape, lambda *i: s_i2h(*i) + s_h2h(*i), name="gates")
    gshape = (num_step, batch_size, num_hidden)
    in_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, 0, i, j]), name="in_gate")
    in_transform = te.compute(
        gshape, lambda t, i, j: te.tanh(gates[t, 1, i, j]), name="in_transform"
    )
    forget_gate = te.compute(
        gshape, lambda t, i, j: te.sigmoid(gates[t, 2, i, j]), name="forget_gate"
    )
    out_gate = te.compute(gshape, lambda t, i, j: te.sigmoid(gates[t, 3, i, j]), name="out_gate")
    next_c = te.compute(
        gshape,
        lambda t, i, j: forget_gate[t, i, j] * s_state_c[t - 1, i, j]
        + in_gate[t, i, j] * in_transform[t, i, j],
        name="next_c",
    )
    next_h = te.compute(
        gshape, lambda t, i, j: out_gate[t, i, j] * te.tanh(next_c[t, i, j]), name="next_h"
    )
    update_c = te.compute(gshape, lambda *i: next_c(*i), name="update_c")
    update_h = te.compute(gshape, lambda *i: next_h(*i), name="update_h")
    # schedule
    scan_h, scan_c = tvm.te.scan(
        [s_init_h, s_init_c],
        [update_h, update_c],
        [s_state_h, s_state_c],
        inputs=[X],
        name="lstm_scan",
    )
    # schedule
    s = te.create_schedule(scan_h.op)
    # Inline gate computations
    s[gates].compute_inline()
    s[in_gate].compute_inline()
    s[in_transform].compute_inline()
    s[forget_gate].compute_inline()
    s[out_gate].compute_inline()
    # verify we can lower correctly
    tvm.lower(s, [X, Wi2h, Wh2h, scan_h, scan_c])


if __name__ == "__main__":
    test_lstm_cell_inline()
