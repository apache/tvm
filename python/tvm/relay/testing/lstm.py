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

def lstm_cell(inputs, states, i2h_weight, h2h_weight,
              i2h_bias, h2h_bias, num_hidden):
    i2h = layers.dense_add_bias(data=inputs, weight=i2h_weight,
                                bias=i2h_bias, units=num_hidden * 4)
    h2h = layers.dense_add_bias(data=inputs, weight=h2h_weight,
                                bias=h2h_bias, units=num_hidden * 4)

    gates = relay.add(i2h, h2h)
    slice_gates = relay.split(gates, indices_or_sections=4)

    in_gate = relay.sigmoid(slice_gates[0])
    forget_gate = relay.sigmoid(slice_gates[1])
    in_transform = relay.tanh(slice_gates[2])
    out_gate = relay.sigmoid(slice_gates[3])
    next_c = relay.add(relay.mul(forget_gate, states[1]),
                       relay.mul(in_gate, in_transform))
    next_h = relay.mul(out_gate, relay.tanh(next_c))

    return relay.Tuple([next_h, relay.Tuple([next_h, next_c])])
