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
from collections import OrderedDict, namedtuple

# make topology reference unknown

class Topology(object):
    def __init__(self, graph):
        self.graph = graph
        self.hardware = None
        self._node2idx = self._build_node_index()
        self._edge2idx = self._build_edge_index()
        self._node2edges = self._build_node2edges()

    def is_quantized_node(self, node):
        assert self.hardware is not None
        return self._node_conds[self._node2idx[node]]

    def is_quantized_edge(self, edge):
        assert self.hardware is not None
        return self._edge_conds[self._edge2idx[edge]]

    def node2idx(self):
        return self._node2idx

    def edge2idx(self):
        return self._edge2idx

    def node2edges(self):
        return self._node2edges

    def analyze(self, hardware):
        self.hardware = hardware
        node2idx = self.node2idx()
        for node in node2idx:
            print(node_str(node))
        edge2idx = self.edge2idx()
        for edge in edge2idx:
            print(edge_str(edge))
        # expand condition list
        self._node_conds = [None] * len(node2idx)
        self._edge_conds = [None] * len(edge2idx)

        def fvisit_analyze(node):
            def set_cond(node, cond):
                nidx = node2idx[node]
                self._node_conds[nidx] = cond
                for edge in list_in_edges(node):
                    eidx = edge2idx[edge]
                    self._edge_conds[eidx] = cond

            if isinstance(node, (relay.Var, relay.Constant)):
                # mark variable as float
                self._node_conds[node2idx[node]] = False
                return

            if isinstance(node, relay.Call):
                # print(node.op.name)
                if not hardware.list_integer_descs(node.op):
                    # current op does not support integer computation 
                    set_cond(node, False)
                    return

                src_node_conds = [self._node_conds[node2idx[src]] for src in list_in_nodes(node)]
                if not any(src_node_conds) and hardware.list_float_descs(node.op):
                    # all float input and current op support float computation
                    set_cond(node, False)
                else:
                    set_cond(node, True)
                return
        relay.analysis.post_order_visit(self.graph, fvisit_analyze)

        print('analyzed condition')
        print('node_conds: {}'.format(self._node_conds))
        print('edge_conds: {}'.format(self._edge_conds))

        # check that all condition has been set properly
        for cond in self._node_conds + self._edge_conds:
            assert cond is not None

    def generate_search_space(self):
        assert self.hardware is not None
        hardware = self.hardware
        node2idx = self.node2idx()
        edge2idx = self.edge2idx()

        # generate a maximum bit list, whose order is same with edge2idx
        # but without non-quantized edges
        bits = []
        for node in node2idx:
            if self.is_quantized_node(node):
                for src_idx, src in enumerate(list_in_nodes(node)):
                    dst_can_consume = [desc.in_dtype(src_idx).bits for desc in hardware.list_integer_descs(node.op)]
                    if isinstance(src, (relay.Var, relay.Constant)):
                        src_can_produce = []
                    else:
                        src_can_produce = [desc.out_dtype(0).bits for desc in hardware.list_integer_descs(src.op)]
                    max_consume = max(dst_can_consume) if len(dst_can_consume) else None
                    max_produce = max(src_can_produce) if len(src_can_produce) else None
                    final_bit = min_with_none(max_consume, max_produce)
                    bits.append(final_bit)

        print('bit limit')
        print(bits)
        self.edge2bit = self.build_edge_info(bits)
        self.print_edge_info(self.edge2bit)

        choices = [list(reversed(range(4, bit + 1))) for bit in bits]
        # print('bit choices')
        # edge2choices = complete_dict(choices, topology.edge2cond)
        # print_edge_dict(graph, edge2choices)
        return choices

    def _build_node_index(self):
        node2idx = OrderedDict()
        def fvisit_build_index(node):
            if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
                node2idx[node] = fvisit_build_index.idx_cnt 
                fvisit_build_index.idx_cnt += 1
        fvisit_build_index.idx_cnt = 0
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        num_nodes = fvisit_build_index.idx_cnt
        return node2idx
    
    def _build_edge_index(self):
        edge2idx = OrderedDict() 
        def fvisit_build_index(node):
            if isinstance(node, relay.Call):
                for edge in list_in_edges(node):
                    edge2idx[edge] = fvisit_build_index.idx_cnt 
                    fvisit_build_index.idx_cnt += 1

        fvisit_build_index.idx_cnt = 0
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        num_edges = fvisit_build_index.idx_cnt
        return edge2idx

    def _build_node2edges(self):
        node2edges = defaultdict(list)
        def fvisit_build_index(node):
            if isinstance(node, relay.Call):
                for edge in list_in_edges(node):
                    node2edges[edge[0]].append(edge) 
        relay.analysis.post_order_visit(self.graph, fvisit_build_index)
        return node2edges

    def build_node_info(self, alist):
        ret = OrderedDict()
        cnt = 0
        node2idx = self.node2idx()
        for key, nidx in node2idx.items():
            val = None
            if self._node_conds[nidx]:
                val = alist[cnt]
                cnt += 1
            ret[key] = val
        assert cnt == len(alist)
        return ret
    
    def build_edge_info(self, alist):
        ret = OrderedDict()
        cnt = 0
        edge2idx = self.edge2idx()
        for key, eidx in edge2idx.items():
            val = None
            if self._edge_conds[eidx]:
                val = alist[cnt]
                cnt += 1
            ret[key] = val
        assert cnt == len(alist)
        return ret

    def print_node_info(self, node2info):
        node2idx = self.node2idx(self.graph)
        def fvisit_print(node):
            if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
                print('{}: {}'.format(node_str(node, node2idx), node2info[node]))
        relay.analysis.post_order_visit(graph, fvisit_print)
    
    
    def print_edge_info(self, edge2info):
        node2idx = self.node2idx()
        node2edges = self.node2edges()
        def fvisit_print(node):
            if isinstance(node, relay.Call):
                oedges = node2edges[node]
                out_infos = [edge2info[e] for e in oedges]
                print('--------')
                print('{}: {}'.format(node_str(node, node2idx), out_infos))
                for edge in list_in_edges(node):
                    info = edge2info[edge]
                    print('  {} : {}'.format(edge_str(edge, node2idx), info))
        relay.analysis.post_order_visit(self.graph, fvisit_print)


def analyze_topology(graph, hardware):
    topology = Topology(graph)
    topology.analyze(hardware)
    return topology
