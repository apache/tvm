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
from __future__ import absolute_import
import numpy as np
import multiprocessing as mp
import logging
try:
    import scipy
except ImportError:
    scipy = None

import tvm

from .base import *
from . import quantize as qtz
from . import analysis
from .. import relay


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


def _find_scale_by_kl(arr,
                      quantized_dtype='int8',
                      num_bins=8001,
                      num_quantized_bins=255):
    """Given a tensor, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)

    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    if min_val >= 0 and quantized_dtype in ['uint8']:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2,
                   num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = scipy.stats.entropy(p, q)

    min_divergence_idx = np.argmin(divergence)
    opt_th = thresholds[min_divergence_idx]
    return opt_th


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

    if cfg.threshold_estimate_strategy == 'global_scale':
        thresholds = [cfg.global_scale for _ in exprs]
    elif cfg.threshold_estimate_strategy == 'max_range':
        thresholds = [stats.range(i) for i in range(len(stats))]
    elif cfg.threshold_estimate_strategy == 'power2_range':
        thresholds = [stats.power2_range(i) for i in range(len(stats))]
    else:
        raise ValueError

    print('before rectify, thresholds: {}'.format(thresholds))
    if rectify:
        thresholds = threshold_rectify(graph, topology, bits, thresholds)
    print('after rectify, thresholds: {}'.format(thresholds))
    return thresholds
