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

"""
Implementation of a Long Short-Term Memory (LSTM) cell.

Adapted from:
https://gist.github.com/merrymercy/5eb24e3b019f84200645bd001e9caae9
"""

from tvm import relay
from . import layers
from .init import create_workload


def lstm_cell(num_hidden, batch_size=1, dtype="float32", name=""):
    """Long-Short Term Memory (LSTM) network cell.

    Parameters
    ----------
    num_hidden : int
        Number of units in output symbol.

    batch_size : int
        Batch size (length of states).

    Returns
    -------
    result : tvm.relay.Function
        A Relay function that evaluates an LSTM cell.
        The function takes in a tensor of input data, a tuple of two
        states, and weights and biases for dense operations on the
        inputs and on the state. It returns a tuple with two members,
        an output tensor and a tuple of two new states.
    """
    builder = relay.ScopeBuilder()

    input_type = relay.TensorType((batch_size, num_hidden), dtype)
    weight_type = relay.TensorType((4 * num_hidden, num_hidden), dtype)
    bias_type = relay.TensorType((4 * num_hidden,), dtype)

    dense_type = relay.TensorType((batch_size, 4 * num_hidden), dtype)
    slice_type = relay.TupleType([input_type, input_type, input_type, input_type])
    ret_type = relay.TupleType([input_type, relay.TupleType([input_type, input_type])])

    inputs = relay.Var("inputs", input_type)
    states = relay.Var("states", relay.TupleType([input_type, input_type]))

    i2h_weight = relay.Var("i2h_weight", weight_type)
    i2h_bias = relay.Var("i2h_bias", bias_type)

    h2h_weight = relay.Var("h2h_weight", weight_type)
    h2h_bias = relay.Var("h2h_bias", bias_type)

    i2h = builder.let(
        ("i2h", dense_type),
        layers.dense_add_bias(
            data=inputs, units=num_hidden * 4, weight=i2h_weight, bias=i2h_bias, name=f"{name}i2h"
        ),
    )
    h2h = builder.let(
        ("h2h", dense_type),
        layers.dense_add_bias(
            data=relay.TupleGetItem(states, 0),
            units=num_hidden * 4,
            weight=h2h_weight,
            bias=h2h_bias,
            name=f"{name}h2h",
        ),
    )

    gates = builder.let(("gates", dense_type), relay.add(i2h, h2h))
    slice_gates = builder.let(
        ("slice_gates", slice_type), relay.split(gates, indices_or_sections=4, axis=1).astuple()
    )

    in_gate = builder.let(
        ("in_gate", input_type), relay.sigmoid(relay.TupleGetItem(slice_gates, 0))
    )
    forget_gate = builder.let(
        ("forget_gate", input_type), relay.sigmoid(relay.TupleGetItem(slice_gates, 1))
    )
    in_transform = builder.let(
        ("in_transform", input_type), relay.tanh(relay.TupleGetItem(slice_gates, 2))
    )
    out_gate = builder.let(
        ("out_gate", input_type), relay.sigmoid(relay.TupleGetItem(slice_gates, 3))
    )

    next_c = builder.let(
        ("next_c", input_type),
        relay.add(
            relay.multiply(forget_gate, relay.TupleGetItem(states, 1)),
            relay.multiply(in_gate, in_transform),
        ),
    )
    next_h = builder.let(("next_h", input_type), relay.multiply(out_gate, relay.tanh(next_c)))
    ret = builder.let(("ret", ret_type), relay.Tuple([next_h, relay.Tuple([next_h, next_c])]))
    builder.ret(ret)

    body = builder.get()

    return relay.Function(
        [inputs, states, i2h_weight, i2h_bias, h2h_weight, h2h_bias], body, ret_type
    )


def get_net(iterations, num_hidden, batch_size=1, dtype="float32"):
    """Constructs an unrolled RNN with LSTM cells"""
    input_type = relay.TensorType((batch_size, num_hidden), dtype)
    weight_type = relay.TensorType((4 * num_hidden, num_hidden), dtype)
    bias_type = relay.TensorType((4 * num_hidden,), dtype)

    state_type = relay.TupleType([input_type, input_type])
    cell_type = relay.TupleType([input_type, state_type])

    builder = relay.ScopeBuilder()

    zeros = builder.let(("zeros", input_type), relay.zeros((batch_size, num_hidden), dtype))
    init_states = builder.let(("init_states", state_type), relay.Tuple([zeros, zeros]))

    states = init_states
    out = None

    for i in range(iterations):
        inputs = relay.Var("data", input_type)
        i2h_weight = relay.Var(f"i2h_{i}_weight", weight_type)
        i2h_bias = relay.Var(f"i2h_{i}_bias", bias_type)
        h2h_weight = relay.Var(f"h2h_{i}_weight", weight_type)
        h2h_bias = relay.Var(f"h2h_{i}_bias", bias_type)

        cell_fn = lstm_cell(num_hidden, batch_size, dtype, f"lstm_{i}")

        call = builder.let(
            (f"call_{i}", cell_type),
            relay.Call(cell_fn, [inputs, states, i2h_weight, i2h_bias, h2h_weight, h2h_bias]),
        )
        new_out = builder.let((f"out_{i}", input_type), relay.TupleGetItem(call, 0))
        new_states = builder.let((f"states_{i}", state_type), relay.TupleGetItem(call, 1))
        states = new_states
        out = new_out

    builder.ret(out)
    body = builder.get()
    args = relay.analysis.free_vars(body)
    return relay.Function(args, body, input_type)


def get_workload(iterations, num_hidden, batch_size=1, dtype="float32"):
    """Get benchmark workload for an LSTM RNN.

    Parameters
    ----------
    iterations : int
        The number of iterations in the desired LSTM RNN.
    num_hidden : int
        The size of the hiddxen state
    batch_size : int, optional (default 1)
        The batch size used in the model
    dtype : str, optional (default "float32")
        The data type
    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a LSTM network.
    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(iterations, num_hidden, batch_size, dtype)
    return create_workload(net)
