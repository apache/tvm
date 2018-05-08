"""TVM operator for LSTM cell"""
from __future__ import absolute_import

import tvm
import topi

@tvm.target.generic_func
def lstm(in_data, in_weight, in_bias, in_state, forget_bias=1.0):
    """LSTM block cell

    xh = [x, h_prev]
    [i, ci, f, o] = xh * w + b
    f = f + forget_bias

    if not use_peephole:
    wci = wcf = wco = 0

    i = sigmoid(cs_prev * wci + i)
    f = sigmoid(cs_prev * wcf + f)
    ci = tanh(ci)

    cs = ci .* i + cs_prev .* f
    cs = clip(cs, cell_clip)

    o = sigmoid(cs * wco + o)
    co = tanh(cs)
    h = co .* o

    Parameters
    ----------
    in_data : tvm.Tensor
        Input for an LSTM cell with shape [batch_size, input_size]

    in_weight : tvm.Tensor
        Hidden layer weight tensor with
        shape [input_size + num_hidden, 4*num_hidden]

    in_bias : float
        Hidden layer bias with shape [4*num_hidden]

    in_state : tvm.Tensor
        Input state (previous cell output state) with
        shape [2, batch_size, num_hidden]

    forget_bias : value
        Forget gate bias


    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    _, batch_size, num_hidden = in_state.shape
    state_c, state_h = topi.split(in_state, 2, axis=0)
    state_h = topi.reshape(state_h, (batch_size, num_hidden))
    state_c = topi.reshape(state_c, (batch_size, num_hidden))
    weight_h, hidden_layers = in_weight.shape
    num_hidden = hidden_layers / 4
    _xh = []
    _xh.append(in_data)
    _xh.append(state_h)
    ixh = topi.concatenate(_xh, axis=1)
    # LSTM transition
    k = tvm.reduce_axis((0, weight_h), name="ki2h")
    s_h2h = tvm.compute(
        (batch_size, hidden_layers),
        lambda i, j: tvm.sum(ixh[i, k] * in_weight[k, j], axis=k), name="s_h2h")
    states = tvm.compute((batch_size, hidden_layers),
                         lambda i, j: (s_h2h[i, j] + in_bias[j]),
                         name="states")
    gates = states
    gshape = (batch_size, num_hidden)
    in_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid(gates[i, j]),
                          name="in_gate")
    in_transform = tvm.compute(gshape, lambda i, j: tvm.tanh(gates[i, (1 * num_hidden) + j]),
                               name="in_transform")
    forget_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid((gates[i, (2 * num_hidden) + j])
                                                               + forget_bias), name="forget_gate")
    out_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid(gates[i, (3 * num_hidden) + j]),
                           name="out_gate")
    next_c = tvm.compute(gshape, lambda i, j: forget_gate[i, j] * state_c[i, j] + \
                                              in_gate[i, j] * in_transform[i, j], name="next_c")
    next_h = tvm.compute(gshape, lambda i, j: out_gate[i, j] * tvm.tanh(next_c[i, j]),
                         name="next_h")
    next_h = topi.reshape(next_h, (1, batch_size, num_hidden))
    next_c = topi.reshape(next_c, (1, batch_size, num_hidden))
    next_state = []
    next_state.append(next_c)
    next_state.append(next_h)
    return topi.concatenate(next_state, axis=0)
