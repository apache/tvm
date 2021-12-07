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

import logging
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.relay.debug import PassBisection
from tvm.relay import testing


def create_model():
    data = relay.var("data", relay.TensorType((1, 42, 42, 42), "float32"))
    simple_net = relay.nn.conv2d(
        data=data,
        weight=relay.var("weight"),
        kernel_size=(3, 3),
        channels=64,
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="float32",
    )

    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    return simple_net


def create_graph(net, bisect):
    net, params = testing.create_workload(net)
    with tvm.target.create("llvm"):

        logging.getLogger().setLevel(logging.ERROR)

        count_pass_inst = PassBisection(limit=bisect)
        with tvm.transform.PassContext(opt_level=3, instruments=[count_pass_inst]):
            graph, lib, params = relay.build_module.build(net, params=params)
    return graph


def compare_graphs(graph0, graph1):
    return graph0 == graph1


if __name__ == "__main__":
    net = create_model()
    graph0 = create_graph(net, bisect=0)  # No bissection
    graph1 = create_graph(net, bisect=1)  # Only 1 pass runs
    same = compare_graphs(graph0, graph1)
    assert not same, "Bissection did not work! Graphs are identical"
