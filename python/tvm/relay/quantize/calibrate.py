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

from . import _quantize
from . import quantize
from .. import op as _op
from .. import expr as _expr
from .. import module as _module
from .. import analysis as _analysis
from .. import transform as _transform
from .. import build_module as _build_module
from ...contrib import graph_runtime


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


class Stats(object):
    def __init__(self, raw_data):
        """
        raw_data: intermediate data * number_of_batches
        """
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def data(self, idx):
        return self.raw_data[idx] 

    def range(self, idx, power_of_2=False):
        arr = np.concatenate(self.raw_data[idx]).reshape(-1)
        arange = np.max(np.abs(arr))
        if power_of_2:
            return math.floor(math.log(arange))
        return arange

    def mean(self, idx):
        pass

    def variance(self, idx):
        pass


def collect_stats(mod, dataset):
    """Given an annotated graph, create a profile graph to collect profile data from the
    calibration dataset. This pass collects simulated_quantize op input into a tuple.
    Simulated_quantize ops are rewritten to identity mode. The tuple is the output of the profile
    graph.

    Parameters
    ----------
    graph: Function
        The simulation graph after annotation.

    Returns
    -------
    ret: Function
        The profile graph which outputs a tuple of profile data.
    """
    logging.info("collecting statistics for calibration...")
    if isinstance(mod, tvm.relay.Module):
        func = mod['main']
    else:
        func = mod
    func = _quantize.CreateStatsCollector(func)
    with _transform.build_config(opt_level=2):
        graph, lib, params = _build_module.build(func, target="llvm")
    outputs = []
    runtime = graph_runtime.create(graph, lib, tvm.cpu())
    runtime.set_input(**params)

    num_outputs = runtime.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]

    for batch_id, batch in enumerate(dataset):
        runtime.set_input('data', tvm.nd.array(batch['data']))
        runtime.run()
        for i in range(num_outputs):
            output = runtime.get_output(i).asnumpy()
            outputs[i].append(output)
    return Stats(outputs)
    # for i in range(num_outputs):
    #     outputs[i] = np.concatenate(outputs[i]).reshape(-1)
    # return outputs


def _kl_scale(stats):
    assert scipy is not None, "scipy need to be installed for \
    utilizing kl calibration during quantization"
    with mp.Pool() as pool:
        logging.info("finding threshold with kl for calibration...")
        scales = list(pool.map(_find_scale_by_kl, stats))

    def func(sq_call):
        scale = scales[func.scale_idx]
        func.scale_idx += 1
        return scale
    func.scale_idx = 0

    return func


def set_params(mod, input_scale_func, weight_scale_func):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    cfg = quantize.current_qconfig()
    const_params = {}

    def visit_func(expr):
        # TODO(ziheng) memorize, e.g. two sq share the same scales
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            sq = expr
            _, ndom_scale, nclip_min, nclip_max = sq.args
            attrs = sq.attrs
            kind = attrs.kind
            nbit = cfg.get_nbit_by_kind(kind)
            valid_bit = nbit - attrs.sign

            # set scale
            if kind == quantize.QAnnotateKind.WEIGHT:
                assert isinstance(sq.args[0], _expr.Constant)
                scale = weight_scale_func(sq)
            else:
                scale = input_scale_func(sq)

            def _make_const(val):
                return _expr.const(val, 'float32')

            valid_range = 2**valid_bit
            const_params[ndom_scale] = _make_const(scale / valid_range)
            const_params[nclip_min] = _make_const(- (valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    func = mod['main']
    _analysis.post_order_visit(func, visit_func)
    func = _expr.bind(func, const_params)
    return _module.Module.from_expr(func)


# weight scale functions
def _power2_scale(sq_call):
    """calculate weight scale with nearest mode-2 scale"""
    var = sq_call.args[0]
    assert isinstance(var, _expr.Constant)
    val = np.amax(np.abs(var.data.asnumpy()))
    return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

def _max_scale(sq_call):
    """calculate weight scale with maximum absolute value"""
    var = sq_call.args[0]
    assert isinstance(var, _expr.Constant)
    val = np.amax(np.abs(var.data.asnumpy()))
    return val


# input scale functions
def _global_scale(sq_call):
    cfg = quantize.current_qconfig()
    return cfg.global_scale


def calibrate(dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    graph: Function
        The simulation graph after annotation.

    mod: tvm.relay.Module
        The module where calibration happens on.

    ctx: tvm.relay.PassContext
        The pass context used for calibration.

    weight_scales: 'power2' or 'max'.
        The way to calculate scales for weights (annotated with QAnnotateKind.WEIGHT).
        power2: Find the maximum of the absolute value of the tensor, and then round up to power
        of two.
        max: Find the maximum of the absolute value of the tensor.

    scales: List[float]
        Pre-calculated scales for input and activations. Length and the order of elements of the
        scales list should match the output tuple of the profile graph created by collect_stats.

    Returns
    -------
    ret: Function
        The graph after calibration
    """
    def wrapped_func(mod, ctx):
        """make transform.module pass happy"""
        cfg = quantize.current_qconfig()

        if cfg.calibrate_mode == 'kl':
            stats = collect_stats(mod, dataset)
            input_scale_func = _kl_scale(stats)
        elif cfg.calibrate_mode == 'global_scale':
            input_scale_func = _global_scale

        return set_params(mod, input_scale_func, _power2_scale)
    return wrapped_func
