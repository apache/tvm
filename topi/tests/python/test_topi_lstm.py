"""LSTM Topi testcase"""
import tvm
import topi
import numpy as np

def LSTM_layer_python(inputs, Wh2h_np, Bh2h_np, state_np,
            o_state_np_py, num_layers):
    batch_size, input_size = inputs.shape

    def _LSTM_cell(in_data, in_weight, in_bias, in_state, forget_bias=1.0):
        def _sigmoid(x, derivative=False):
            return x*(1-x) if derivative else 1/(1+np.exp(-x))

        dtype = "float32"
        _, batch_size, num_hidden = in_state.shape
        state_c, state_h = np.split(in_state, 2, axis=0)
        state_h = np.reshape(state_h, (batch_size, num_hidden))
        state_c = np.reshape(state_c, (batch_size, num_hidden))
        weight_h, hidden_layers = in_weight.shape
        num_hidden = hidden_layers / 4
        xh_concat = []
        xh_concat.append(in_data)
        xh_concat.append(state_h)
        xh = np.concatenate(xh_concat, axis=1)

        # LSTM transition
        s_h2h = np.zeros(shape=(batch_size, hidden_layers)).astype(dtype)
        for i in range(batch_size):
            for j in range(hidden_layers):
                sum = 0
                for k in range(weight_h):
                    sum += xh[i, k] * in_weight[k, j]
                s_h2h[i][j] = sum
                s_h2h[i][j] = s_h2h[i][j] + in_bias[j]

        gates = s_h2h
        gshape = (batch_size, num_hidden)
        in_gate = np.zeros(shape=gshape).astype(dtype)
        in_transform = np.zeros(shape=gshape).astype(dtype)
        forget_gate = np.zeros(shape=gshape).astype(dtype)
        out_gate = np.zeros(shape=gshape).astype(dtype)
        for i in range(batch_size):
            for j in range(num_hidden):
                in_gate[i][j] = _sigmoid(gates[i, j])
                in_transform[i][j] = np.tanh(gates[i, (1 * num_hidden) + j])
                forget_gate[i][j] = _sigmoid((gates[i, (2 * num_hidden) + j]) \
                        + forget_bias)
                out_gate[i][j] = _sigmoid(gates[i, (3 * num_hidden) + j])

        next_c = np.zeros(shape=gshape).astype(dtype)
        next_h = np.zeros(shape=gshape).astype(dtype)
        for i in range(batch_size):
            for j in range(num_hidden):
                next_c[i][j] = forget_gate[i, j] * state_c[i, j] + \
                                   in_gate[i, j] * in_transform[i, j]
                next_h[i][j] = out_gate[i, j] * np.tanh(next_c[i, j])

        next_h = np.reshape(next_h, (1, batch_size, num_hidden))
        next_c = np.reshape(next_c, (1, batch_size, num_hidden))
        next_state = []
        next_state.append(next_c)
        next_state.append(next_h)
        return np.concatenate(next_state, axis=0)

    for layer in range(num_layers):
        #single LSTM cell
        state_layer_np = state_np[layer, :, :, :]
        o_state_np_a = _LSTM_cell(inputs, Wh2h_np, Bh2h_np, state_layer_np)
        inputs = np.reshape(o_state_np_a[1, :, :],
                            (batch_size, input_size))
        o_state_np_py[layer, :, :, :] = o_state_np_a
    return o_state_np_py

def verify_lstm(batch_size, input_size, num_hidden,
            num_layers, num_times, forget_bias):
    hidden_layers = 4 * num_hidden
    Xi2h = tvm.placeholder((batch_size, input_size), name="Xi2h")
    Wh2h = tvm.placeholder((input_size + num_hidden, hidden_layers),
                           name="Wh2h")
    Bh2h = tvm.placeholder((hidden_layers, ), name="Bh2h")
    State = tvm.placeholder((2, batch_size, num_hidden), name="state")
    o_state = topi.nlp.lstm(Xi2h, Wh2h, Bh2h, State, forget_bias)

    ###########################################################################
    target='llvm'
    target_host='llvm'
    ctx = tvm.context(target, 0)
    ###########################################################################
    def check_device():
        print("Running on target: %s" % target)
        with tvm.target.create(target):
            s = topi.generic.schedule_lstm(o_state)
        flstm = tvm.build(s, [Xi2h, Wh2h, Bh2h, State, o_state],
                          target=target, target_host=target_host)
        Xi2h_np = np.full((batch_size, num_times, input_size),
                          1.0, dtype="float32")
        Wh2h_np = np.full((input_size + num_hidden, hidden_layers),
                          0.5, dtype="float32")
        Bh2h_np = np.full((hidden_layers), 0.0, dtype="float32")
        state_np = np.full((num_layers, 2, batch_size, num_hidden),
                          0.1, dtype="float32")
        state_np_py = np.full((num_layers, 2, batch_size, num_hidden),
                          0.1, dtype="float32")
        o_state_np = np.zeros((num_layers, 2,
                              batch_size, num_hidden)).astype("float32")
        o_state_np_py = np.zeros((num_layers, 2,
                              batch_size, num_hidden)).astype("float32")
        Wh2h_a = tvm.nd.array(Wh2h_np, ctx)
        Bh2h_a = tvm.nd.array(Bh2h_np, ctx)

        #execute LSTM cell layers
        def _LSTMCell_layer(inputs, Wh2h_a, Bh2h_a, state_np,
                    o_state_np, num_layers):
            for layer in range(num_layers):
                #single LSTM cell
                Xi2h_a = tvm.nd.array(inputs, ctx)
                state_layer_np = state_np[layer, :, :, :]
                state_a = tvm.nd.array(state_layer_np, ctx)
                o_state_layer_np = o_state_np[layer, :, :, :]
                o_state_a = tvm.nd.array(o_state_layer_np, ctx)
                flstm(Xi2h_a, Wh2h_a, Bh2h_a, state_a, o_state_a)
                o_state_np_a = o_state_a.asnumpy()
                inputs = np.reshape(o_state_np_a[1, :, :],
                                    (batch_size, input_size))
                o_state_np[layer, :, :, :] = o_state_np_a
            return o_state_np

        #Time steps execution on LSTMCell layers
        for t_step in range(num_times):
            output_state = _LSTMCell_layer(Xi2h_np[:, t_step, :],
                                           Wh2h_a, Bh2h_a, state_np,
                                           o_state_np, num_layers)
            output_state_py = LSTM_layer_python(Xi2h_np[:, t_step, :],
                                           Wh2h_np, Bh2h_np, state_np_py,
                                           o_state_np_py, num_layers)
            np.testing.assert_allclose(output_state, output_state_py, rtol=1e-5)
            state_np = output_state
            state_np_py = output_state_py
        ctx.sync()

    check_device()

def test_lstm():
    verify_lstm(1, 2, 2, 2, 2, 1.0)

if __name__ == "__main__":
    test_lstm()
