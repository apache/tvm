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
from .kl_divergence import _find_scale_by_kl

def threshold_rectify(graph, topology, bits, thresholds):
    # print('bits')
    # print(bits)
    edge2idx = topology.edge2idx()
    node2idx = topology.node2idx()
    node2edges = topology.node2edges()
    edge2bit = topology.build_edge_info(bits)
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
            if not topology.is_quantized_node(node):
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


def _round2pot(x):
    pot = 2**np.math.ceil(np.math.log(x, 2)) if x > 0 else 1.0
    return pot

def threshold_estimate(graph, topology, stats, bits=None):
    print('calculating threshold...')
    cfg = current_qconfig()
    print('threshold method:')
    print(cfg.threshold_estimate_method)
    if cfg.threshold_estimate_method == 'global_scale':
        thresholds = [cfg.global_scale for _ in exprs]
    elif cfg.threshold_estimate_method == 'avg_range':
        thresholds = stats.avg_range
    elif cfg.threshold_estimate_method == 'kl_estimate':
        thresholds = []
        for idx in range(len(stats.data)):
          samples = len(stats.data[idx])
          arr = np.concatenate(stats.data[idx]).reshape(samples, -1)
          thresholds.append(_find_scale_by_kl(arr))
    elif cfg.threshold_estimate_method.startswith('quantile_range:'):
        quantile = float(cfg.threshold_estimate_method[len('quantile_range:'):])
        assert(0 <= quantile and quantile <= 1, "quantile range must be in the range of [0, 1]")
        thresholds = [np.quantile(arr, quantile) for arr in stats.data]
    else:
        raise ValueError

    if cfg.round_scale_to_pot:
        thresholds = [_round2pot(x) for x in thresholds]

    print('thresholds: {}'.format(thresholds))
    return thresholds
