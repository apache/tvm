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
from tvm import relay
from tvm.relay.analysis import get_calibration_data

import numpy as np

def test_basic():
    # A module with two subgraphs
    mod = tvm.IRModule()

    x0 = relay.var('x0', shape=(8, 8))
    y0 = relay.var('y0', shape=(8, 8))
    z0 = x0 + y0
    f0 = relay.Function([x0, y0], z0)
    g0 = relay.GlobalVar("g0")
    mod[g0] = f0

    x1 = relay.var('x1', shape=(8, 8))
    y1 = relay.var('y1', shape=(8, 8))
    z1 = x1 - y1
    f1 = relay.Function([x1, y1], z1)
    g1 = relay.GlobalVar("g1")
    mod[g1] = f1

    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))
    z = relay.var('z', shape=(8, 8))
    c0 = relay.Call(g0, [x, y])
    c1 = relay.Call(g1, [c0, z])
    fm = relay.Function([x, y, z], c1)
    mod["main"] = fm

    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    z_data = np.random.rand(8, 8).astype('float32')
    data = get_calibration_data(mod, {"x": x_data, "y": y_data, "z": z_data})

    print(data)

test_basic()

