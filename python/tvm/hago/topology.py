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
    def __init__(self, node_conds=[], edge_conds=[]):
        self.node_conds = node_conds
        self.edge_conds = edge_conds

    def analyze(self, graph, hardware):
        node2idx = build_node_index(graph)
        edge2idx = build_edge_index(graph)
        # expand condition list
        self.node_conds = [None] * len(node2idx)
        self.edge_conds = [None] * len(edge2idx)

        def fvisit_analyze(node):
            def set_cond(node, cond):
                nidx = node2idx[node]
                self.node_conds[nidx] = cond
                for src in node.args:
                    eidx = edge2idx[(src, node)]
                    self.edge_conds[eidx] = cond

            if isinstance(node, (relay.Var, relay.Constant)):
                # mark variable as float
                self.node_conds[node2idx[node]] = False
                return

            if isinstance(node, relay.Call):
                # print(node.op.name)
                if not integer_constraints(hardware[node.op]):
                    # current op does not support integer computation 
                    set_cond(node, False)
                    return

                src_node_conds = [self.node_conds[node2idx[src]] for src in node.args]
                if not any(src_node_conds) and float_constraints(hardware[node.op]):
                    # all float input and current op support float computation
                    set_cond(node, False)
                else:
                    set_cond(node, True)
                return
        relay.analysis.post_order_visit(graph, fvisit_analyze)

        print('analyzed condition')
        print('node_conds: {}'.format(self.node_conds))
        print('edge_conds: {}'.format(self.edge_conds))

        # check that all condition has been set properly
        for cond in self.node_conds + self.edge_conds:
            assert cond is not None
        return


def analyze_topology(graph, hardware):
    topology = Topology()
    topology.analyze(graph, hardware)
    return topology
