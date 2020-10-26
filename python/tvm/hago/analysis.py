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
from .topology import Topology, NodeKind, analyze_topology
from .quantize import create_quantizer
from .record import Strategy
from ..contrib import graph_runtime
from .threshold import threshold_estimate

from tvm import relay
import logging
import numpy as np
from itertools import islice
from collections import OrderedDict

class Stats(object):
    def __init__(self, topology, data, nodes):
        """
        data: [num of intermediate data][number of batch][tensor]
        """
        self.topology = topology
        self.nodes = nodes
        self.node2kinds = topology.node2kind()
        self.node2layouts = topology.node2layout()
        self.node2channel_axis = topology.node2channel_axis()
        self.node2edges = topology.node2edges()
        self.data = []
        for idx, node in enumerate(nodes):
            batched_data = data[idx]
            node_kind = self.node2kinds[node]
            if node_kind in (NodeKind.Input, NodeKind.Activation):
                flatten_data = batched_data
            elif node_kind == NodeKind.Weight:
                flatten_data = batched_data[0]
            else:
                raise ValueError
            self.data.append(flatten_data)
        self._avg_range = None
        self._pot_range = None

    def __len__(self):
        return len(self.data)

    def data(self, idx):
        return self.data[idx]

    def _calculate_avg_range(self, arr):
        # TODO - For some reason, averaging across different batches gives better accuracy compared
        # to averaging across all the images. One reason might be that there might be an outlier
        # causing the avg numbers to go up, but that might be absent in averaging across batches.
        samples = len(arr)
        arr = np.concatenate(arr).reshape(samples, -1)
        avg_min = np.average(np.min(arr, axis=1))
        avg_max = np.average(np.max(arr, axis=1))
        arange = np.amax([np.abs(avg_min), np.abs(avg_max)])
        return arange

    @property
    def avg_range(self):
        if self._avg_range is None:
            self._avg_range = []
            for idx, arr in enumerate(self.data):
                node = self.nodes[idx]
                if self.node2kinds[node] in (NodeKind.Input, NodeKind.Activation):
                    arange = self._calculate_avg_range(arr)
                elif self.node2kinds[node] == NodeKind.Weight:
                    use_channel_quantized = current_qconfig().use_channel_quantize
                    out_edges = self.node2edges[node]
                    assert len(out_edges) == 1
                    op_node = out_edges[0][1]
                    if use_channel_quantized and op_node.op.name in ['nn.conv2d']:
                        # per channel scales
                        layout = self.node2layouts[node]
                        assert layout in ("OIHW", "HWIO")
                        axis = self.node2channel_axis[node]
                        arr = np.moveaxis(arr, axis, 0)
                        num_scales = arr.shape[0]
                        arr = np.reshape(arr, (num_scales, -1))
                        arange = np.amax(np.abs(arr), axis=1)
                    else:
                        arange = np.amax(np.abs(arr))
                else:
                    raise ValueError
                self._avg_range.append(arange)
        return self._avg_range

    def mean(self, idx):
        pass

    def variance(self, idx):
        pass

def collect_stats(graph, topology, dataset, ctx, target):
    assert isinstance(graph, relay.Function)
    assert graph == topology.graph
    logging.info("collecting statistics for calibration...")
    nodes = []
    def fvisit(node):
        if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
            nodes.append(node)

    relay.analysis.post_order_visit(graph, fvisit)
    out = relay.Tuple(nodes)
    func = relay.Function(graph.params, out)
    outputs = evaluate(func, dataset, ctx, target)
    stats = Stats(topology, outputs, nodes)
    logging.info("statistics collected")
    return stats


def compare(np_x, np_y):
    # compare two array in terms of statistic property
    print('max value : {:.4f}, {:.4f}'.format(np.max(np_x), np.max(np_y)))
    print('min value : {:.4f}, {:.4f}'.format(np.min(np_x), np.min(np_y)))
    print('max abs   : {:.4f}, {:.4f}'.format(np.max(np.abs(np_x)), np.max(np.abs(np_y))))
    print('mean      : {:.4f}, {:.4f}'.format(np.mean(np_x), np.mean(np_y)))
    print('var       : {:.4f}, {:.4f}'.format(np.var(np_x), np.var(np_y)))
    abs_err = np.abs(np_x - np_y)
    rel_err = (np_x - np_y) / np.max(np.abs(np_y))
    idx = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)
    print('maximum absolute error: {:.4f}({:.2f}%), compare {:.4f} with {:.4f}'
          .format(np.max(abs_err), rel_err[idx] * 100, np_x[idx], np_y[idx]))
    return rel_err[idx]


# TODO(ziheng) avoid recompute
def inspect_graph_statistic(func, hardware, strategy, dataset, ctx, target):
    print('inspect graph statistic')
    assert isinstance(func, relay.Function)
    assert tvm.ir.structural_hash(func) == strategy.model_hash
    topology = analyze_topology(func, hardware)
    topology.generate_search_space()
    edge2bit = topology.edge2bit
    print('origin bits: \n{}'.format(strategy.bits))

    def collect_bits(node):
        sub_bits = []
        def fvisit_bits(node):
            if isinstance(node, relay.Call):
                for src in node.args:
                    if (src, node) not in edge2bit:
                        raise ValueError 
                    bit = edge2bit[(src, node)]
                    if bit is not None:
                        sub_bits.append(bit)
        relay.analysis.post_order_visit(node, fvisit_bits)
        return sub_bits

    funcs = []
    def fvisit(node):
        if isinstance(node, relay.Call):
            funcs.append((relay.Function(func.params, node), collect_bits(node)))
    relay.analysis.post_order_visit(func, fvisit)

    data_batch = [dataset[0]]

    for graph, sub_bits in funcs:
        topology = analyze_topology(graph, hardware)
        node2idx = topology.node2idx()
        edge2idx = topology.edge2idx()
        num_nodes = len(node2idx)
        num_edges = len(edge2idx)
        print('num nodes: {}'.format(num_nodes))
        print('num edges: {}'.format(num_edges))

        # print('bits:')
        # print_edge_dict(graph, edge2bit)
        stats = collect_stats(graph, dataset, ctx, target)
        sub_tholds = threshold_estimate(graph, topology, stats, sub_bits)
        sub_model_hash = tvm.ir.structural_hash(graph)
        sub_strategy = Strategy(sub_model_hash, topology, sub_bits, sub_tholds)

        quantizer = create_quantizer(graph, hardware, sub_strategy)
        simulated_graph = quantizer.simulate()
        quantized_graph = quantizer.quantize()

        print('evaluate original graph')
        real_out = evaluate(graph, data_batch, ctx, target)[0][0]
        print('evaluate simulated graph')
        simulated_out = evaluate(simulated_graph, data_batch, ctx, target)[0][0]
        # print('evaluate quantized graph')
        quantized_out = evaluate(quantized_graph, data_batch, ctx, target)[0][0]
        print('compare real_out vs. simulated_out')
        rel_err = compare(real_out, simulated_out)
        if rel_err > 0.05:
            raise ValueError
        print('compare real_out vs. quantized_out')
        compare(real_out, quantized_out)
        if not np.allclose(simulated_out, quantized_out):
            print('compare simulated_out vs. quantized_out')
            compare(simulated_out, quantized_out)
            is_close = np.isclose(simulated_out, quantized_out)
            indexes = np.where(np.logical_not(is_close))
            print('num of mismatched items: {}'.format(len(indexes[0])))
            print('simulated out:\n{}'.format(simulated_out[indexes]))
            print('quantized out:\n{}'.format(quantized_out[indexes]))
            print('\nsimulated graph')
            print(simulated_graph)
            print('\nquantized graph')
            print(quantized_graph)
            # raise ValueError
        print('\n\n')
