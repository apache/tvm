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
from . import _quantize
from .topology import Topology, analyze_topology
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
    def __init__(self, data):
        """
        data: intermediate data * number_of_batches
        """
        self.data = data
        self.range = []
        self.power_of_two_range = []
        for idx in range(len(data)):
            arr = np.concatenate(self.data[idx]).reshape(-1)
            arange = np.amax(np.abs(arr))
            power_of_two_range = 2**np.math.ceil(np.math.log(arange, 2)) if arange > 0 else 1.0
            self.range.append(arange)
            self.power_of_two_range.append(power_of_two_range)

    def __len__(self):
        return len(self.data)

    def data(self, idx):
        return self.data[idx] 

    def mean(self, idx):
        pass

    def variance(self, idx):
        pass


def evaluate(func, dataset, ctx, target):
    # list of array: (num_outputs, num_batch, arr)
    with relay.transform.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(func, target=target)
    runtime = graph_runtime.create(graph, lib, ctx)
    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]

    for batch_id, batch in enumerate(dataset):
        runtime.set_input('data', batch['data'])
        runtime.run()
        for i in range(num_outputs):
            output = runtime.get_output(i).asnumpy()
            outputs[i].append(output)
    return outputs


def collect_stats(graph, dataset, ctx, target):
    assert isinstance(graph, relay.Function)
    logging.info("collecting statistics for calibration...")
    outputs = []
    def fvisit(node):
        if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
            outputs.append(node)
    relay.analysis.post_order_visit(graph, fvisit)
    out = relay.Tuple(outputs)
    func = relay.Function(graph.params, out)
    outputs = evaluate(func, dataset, ctx, target)
    stats = Stats(outputs)
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


# def construct_subtopology(graph, topology):
#     node2idx = build_node_index(graph)
#     edge2idx = build_node_index(graph)
#     num_nodes = len(node2idx)
#     num_edges = len(edge2idx)
#     new_topo = Topology()
#     new_topo.node_conds = topology.node_conds[:num_nodes]
#     new_topo.edge_conds = topology.edge_conds[:num_edges]
#     return new_topo


# TODO(ziheng) avoid recompute
def inspect_graph_statistic(func, hardware, strategy, dataset, ctx, target):
    print('inspect graph statistic')
    assert isinstance(func, relay.Function)
    assert tvm.ir.structural_hash(func) == strategy.model_hash
    topology = analyze_topology(func, hardware)
    edge2bit = build_edge_dict(func, strategy.bits, topology.edge_conds)
    bits = [edge2bit[key] for key in edge2bit]
    print('origin bits: \n{}'.format(strategy.bits))
    print_edge_dict(func, edge2bit)

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
        print(graph)
        node2idx = build_node_index(graph)
        edge2idx = build_edge_index(graph)
        num_nodes = len(node2idx)
        num_edges = len(edge2idx)
        print('num nodes: {}'.format(num_nodes))
        print('num edges: {}'.format(num_edges))
        topology = analyze_topology(graph, hardware)

        # print('bits:')
        # print_edge_dict(graph, edge2bit)
        stats = collect_stats(graph, dataset, ctx, target)
        sub_tholds = threshold_estimate(graph, topology, stats, sub_bits)
        sub_model_hash = tvm.ir.structural_hash(graph)
        sub_strategy = Strategy(sub_model_hash, topology, sub_bits, sub_tholds)

        quantizer = create_quantizer(graph, hardware, sub_strategy)
        simulated_graph = quantizer.simulate()
        # quantized_graph = quantizer.quantize()

        print('evaluate original graph')
        real_out = evaluate(graph, data_batch, ctx, target)[0][0]
        print('evaluate simulated graph')
        simulated_out = evaluate(simulated_graph, data_batch, ctx, target)[0][0]
        # print('evaluate quantized graph')
        # quantized_out = evaluate(quantized_graph, data_batch, ctx, target)[0][0]
        print('compare real_out vs. simulated_out')
        rel_err = compare(real_out, simulated_out)
        if rel_err > 0.05:
            raise ValueError
        # print('compare real_out vs. quantized_out')
        # compare(real_out, quantized_out)
        # if not np.allclose(simulated_out, quantized_out):
        #     print('compare simulated_out vs. quantized_out')
        #     compare(simulated_out, quantized_out)
        #     is_close = np.isclose(simulated_out, quantized_out)
        #     indexes = np.where(np.logical_not(is_close))
        #     print('num of mismatched items: {}'.format(len(indexes[0])))
        #     print('simulated out:\n{}'.format(simulated_out[indexes]))
        #     print('quantized out:\n{}'.format(quantized_out[indexes]))
        #     print('\nsimulated graph')
        #     print(simulated_graph)
        #     print('\nquantized graph')
        #     print(quantized_graph)
        #     # raise ValueError
        print('\n\n')
