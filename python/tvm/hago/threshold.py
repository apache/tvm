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
"""Find optimal scale for quantization by minimizing KL-divergence"""
from .base import *
from . import analysis
from .. import relay


def threshold_rectify(graph, topology, bits, thresholds):
    # print('bits')
    # print(bits)
    edge2idx = build_edge_index(graph)
    node2idx = build_node_index(graph)
    edge2bit = build_edge_dict(graph, bits, topology.edge_conds)
    node2edges = build_node2edges(graph)
    # print('num_nodes: {}'.format(num_nodes))
    # print('num_node2edge: {}'.format(len(node2edges)))
    # for node in node2edges:
    #     print('{}: {}'.format(node_str(node), len(node2edges[node])))
    # for node in node2idx:
    #     if node not in node2edges:
    #         print('{} not existed.'.format(node_str(node)))

    assert len(thresholds) == len(node2idx)

    def fvisit_rectify(node):
        if isinstance(node, relay.Call):
            if not topology.node_conds[node2idx[node]]:
                return
            frectify = node.op.get_attr('FHagoRectify')
            if frectify is not None:
                output_edges = node2edges[node]
                print('---------')
                print(node.op.name)
                input_bits = [edge2bit[(src, node)] for src in node.args]
                output_bits = [edge2bit[edge] for edge in output_edges]
                input_tholds = [thresholds[node2idx[src]] for src in node.args]
                output_tholds = [thresholds[node2idx[node]]] * len(output_edges)

                tholds = frectify(input_bits, output_bits, input_tholds, output_tholds)
                assert len(tholds) == (len(input_tholds) + len(output_tholds))
                for i, src in enumerate(node.args):
                    thresholds[node2idx[src]] = tholds[i].value
                # TODO(ziheng) rectify output thresholds
    relay.analysis.post_order_visit(graph, fvisit_rectify)
    # print_scale_info(graph, bits, thresholds)
    return thresholds


def threshold_estimate(graph, topology, bits, dataset=None, rectify=True):
    print('calculating threshold...')
    cfg = current_qconfig()
    stats = analysis.collect_stats(graph, dataset)
    print('threshold method:')
    print(cfg.threshold_estimate_method)

    if cfg.threshold_estimate_method == 'global_scale':
        thresholds = [cfg.global_scale for _ in exprs]
    elif cfg.threshold_estimate_method == 'max_range':
        thresholds = [stats.range(i) for i in range(len(stats))]
    elif cfg.threshold_estimate_method == 'power_of_two_range':
        thresholds = [stats.power2_range(i) for i in range(len(stats))]
    else:
        raise ValueError

    print('before rectify, thresholds: {}'.format(thresholds))
    if rectify:
        thresholds = threshold_rectify(graph, topology, bits, thresholds)
    print('after rectify, thresholds: {}'.format(thresholds))
    return thresholds
