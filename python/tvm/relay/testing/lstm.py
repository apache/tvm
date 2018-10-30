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
    inputs = relay.var("inputs")
    states = relay.var("states",
                       relay.TupleType([
                           relay.TensorType((batch_size, num_hidden), dtype),
                           relay.TensorType((batch_size, num_hidden), dtype)]))

    i2h = layers.dense_add_bias(data=inputs, units=num_hidden * 4,
                                name="%si2h" % name)
    h2h = layers.dense_add_bias(data=relay.TupleGetItem(states, 0),
                                units=num_hidden * 4,
                                name="%sh2h" % name)

    gates = relay.add(i2h, h2h)
    slice_gates = relay.split(gates, indices_or_sections=4)

    in_gate = relay.sigmoid(slice_gates[0])
    forget_gate = relay.sigmoid(slice_gates[1])
    in_transform = relay.tanh(slice_gates[2])
    out_gate = relay.sigmoid(slice_gates[3])
    next_c = relay.add(relay.multiply(forget_gate,
                                      relay.TupleGetItem(states, 1)),
                       relay.multiply(in_gate, in_transform))
    next_h = relay.multiply(out_gate, relay.tanh(next_c))
    ret = relay.Tuple([next_h, relay.Tuple([next_h, next_c])])

    args = relay.ir_pass.free_vars(ret)
    return relay.Function(args, ret)


def rnn_builder(iterations, cell_fn, init_states, out, forward):
    """Recursive builder of unrolled RNN: Returns let-chain of cell function calls.

    Parameters
    ----------
    iterations : int
        Number of iterations for the unrolled RNN.

    cell_fn : tvm.relay.Function
        Relay function implementing the RNN cell computation.

    init_states : tvm.relay.Expr
        A Relay tuple representing the state for the first iteration.

    out : tvm.relay.Var
        A Relay variable into which the last iteration should assign
        its output.

    forward : tvm.relay.Expr
        A Relay AST that uses the result of the last iteration's cell.

    Returns
    -------
    ret : tvm.relay.Expr
        A let-chain of cell function calls of the number of iterations,
        with the last iteration using the "forward" argument as the
        output of the let chain.
    """
    i = iterations
    inputs = relay.Var("inputs_%s" % i)
    i2h_weight = relay.Var("i2h_%s_weight" % i)
    i2h_bias = relay.Var("i2h_%i_bias" % i)
    h2h_weight = relay.Var("h2h_%s_weight" % i)
    h2h_bias = relay.Var("h2h_%s_bias" % i)

    # base case: 0 is the first iteration, so use initial state
    if i == 0:
        return relay.Let(out,
                         relay.Call(cell_fn,
                                    [inputs, init_states, i2h_weight,
                                     i2h_bias, h2h_weight, h2h_bias]),
                         forward)

    # otherwise: create the chain backwards and insert in the last iteration
    prev_out = relay.Var("out_%s" % (i - 1))
    call = relay.Let(out,
                     relay.Call(cell_fn,
                                [inputs,
                                 relay.TupleGetItem(prev_out, 1),
                                 i2h_bias, i2h_weight,
                                 h2h_bias, h2h_weight]),
                     forward)
    return rnn_builder(i - 1, cell_fn, init_states, prev_out, call)


def get_net(iterations, num_hidden, batch_size=1, dtype="float32"):
    '''Constructs an unrolled RNN with LSTM cells'''
    states = relay.Tuple([relay.zeros((batch_size, num_hidden), dtype),
                          relay.zeros((batch_size, num_hidden), dtype)])

    cell_fn = lstm_cell(num_hidden, batch_size, dtype, "lstm")

    out = relay.Var("lstm_out")
    get_value = relay.TupleGetItem(out, 0)
    unrolled = rnn_builder(iterations - 1, cell_fn, states,
                           out, get_value)

    args = relay.ir_pass.free_vars(unrolled)
    return relay.Function(args, unrolled)


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
    net : nnvm.symbol
        The computational graph
    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, num_hidden, batch_size, dtype)
    return create_workload(net)
