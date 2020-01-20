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
from __future__ import absolute_import

from .base import *
from .hardware import *

from tvm import relay
from collections import OrderedDict

# make topology reference unknown

class Topology(object):
    def __init__(self):
        self.edge2cond = OrderedDict()
        self.node2cond = OrderedDict()

    def analyze(self, graph, hardware):
        def fvisit_analyze(node):
            def set_cond(node, cond):
                self.node2cond[node] = cond
                for src in node.args:
                    self.edge2cond[(src, node)] = cond

            if isinstance(node, (relay.Var, relay.Constant)):
                # mark variable as float
                self.node2cond[node] = False
                return

            if isinstance(node, relay.Call):
                # print(node.op.name)
                if not integer_constraints(hardware[node.op]):
                    # current op does not support integer computation 
                    set_cond(node, False)
                    return

                src_node_conds = [self.node2cond[src] for src in node.args]
                if not any(src_node_conds) and float_constraints(hardware[node.op]):
                    # all float input and current op support float computation
                    set_cond(node, False)
                else:
                    set_cond(node, True)
                return
        relay.analysis.post_order_visit(graph, fvisit_analyze)
        # print('analyzed condition')
        # print_infos(graph, self.node2cond, self.edge2cond)
        return



def analyze_topology(graph, hardware):
    topology = Topology()
    topology.analyze(graph, hardware)
    return topology
