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

import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_runtime
from tvm.relay.analysis import pipeline_graph


def run_module(mod, ctx, dname, data):
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm")

    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(dname, data)
    m.set_input(**params)
    m.run()
    output = m.get_output(0).asnumpy()
    return output


def get_network():
    dshape = (3, 3)
    mvalue = np.full((1), 4).astype("float32")
    mmv = relay.Constant(tvm.nd.array(mvalue))
    mv = relay.Constant(tvm.nd.array(mvalue))
    mv2 = relay.Constant(tvm.nd.array(mvalue))
    mv3 = relay.Constant(tvm.nd.array(mvalue))
    data = relay.var("data", relay.TensorType(dshape, "float32"))
    net = relay.multiply(data, mv)
    net = relay.add(net, mv2)
    net = relay.subtract(net, mv3)
    net = relay.add(net, mv3)
    func = relay.Function([data], net)
    mod = tvm.IRModule.from_expr(func)
    return mod, dshape


mod, dshape = get_network()
pl = [0, 2, 3]
mods = pipeline_graph(mod["main"], pl)

data = np.full(dshape, 5).astype("float32")

ctx = tvm.cpu(0)
out = run_module(mod, ctx, "data", data)

o1 = run_module(mods[0], ctx, "data", data)
o2 = run_module(mods[1], ctx, "x", o1)
o3 = run_module(mods[2], ctx, "x", o2)
o4 = run_module(mods[3], ctx, "x", o3)

tvm.testing.assert_allclose(out, o4)
print("suc")
