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
"""Find scales for quantization on the dataset."""
from __future__ import absolute_import
import logging
import multiprocessing as mp
import numpy as np
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
from .kl_divergence import _find_scale_by_kl


def _kl_scale(mod, dataset, split_by=-1):
    logging.info("collecting statistics for calibration...")
    func = _quantize.CreateStatsCollector(mod['main'])
    if tvm.target.current_target():
        target = tvm.target.current_target()
        ctx = tvm.context(target.target_name)
    else:
        target = 'llvm'
        ctx = tvm.context(target)

    with _transform.build_config(opt_level=3):
        graph, lib, params = _build_module.build(func, target=target)

    runtime = graph_runtime.create(graph, lib, ctx)
    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()

    scales = []
    split_by = num_outputs if split_by == -1 else split_by
    for i in range(0, num_outputs, split_by):
        outputs = [[] for i in range(min(split_by, num_outputs - i))]
        for batch in dataset:
            runtime.set_input(**batch)
            runtime.run()
            for j in range(i, min(i+split_by, num_outputs)):
                outputs[j-i].append(runtime.get_output(j).asnumpy())
        samples = [np.concatenate(output).reshape(-1) for output in outputs]

        with mp.Pool() as pool:
            logging.info("finding threshold with kl for calibration...")
            scales += list(pool.map(_find_scale_by_kl, samples))

    def func(_):
        scale = scales[func.scale_idx]
        func.scale_idx += 1
        return scale
    func.scale_idx = 0

    return func


def _set_params(mod, input_scale_func, weight_scale_func):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    cfg = quantize.current_qconfig()
    const_params = {}

    def visit_func(expr):
        '''visitor function for traverse'''
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            nbit = cfg.get_nbit_by_kind(kind)
            valid_bit = nbit - attrs.sign

            # set scale
            if kind == quantize.QAnnotateKind.WEIGHT:
                assert isinstance(expr.args[0], _expr.Constant)
                scale = weight_scale_func(expr)
            else:
                scale = input_scale_func(expr)

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
def _power2_scale(sq_call):  # pylint: disable=unused-argument
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
def _global_scale(sq_call): # pylint: disable=unused-argument
    cfg = quantize.current_qconfig()
    return cfg.global_scale


def calibrate(dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    dataset: Optional[Iterable[NDArray]]
        The calibration dataset.

    Returns
    -------
    ret: Function
        The module pass function.
    """
    def wrapped_func(mod, ctx): # pylint: disable=unused-argument
        """make transform.module pass happy"""
        cfg = quantize.current_qconfig()

        if cfg.calibrate_mode == 'kl_divergence':
            cfg = quantize.current_qconfig()
            input_scale_func = _kl_scale(mod, dataset, cfg.calibrate_split_by)
        elif cfg.calibrate_mode == 'global_scale':
            input_scale_func = _global_scale
        else:
            raise ValueError("Unknown calibrate mode {}".format(cfg.calibrate_mode))

        if cfg.weight_scale == 'max':
            weight_scale_func = _max_scale
        elif cfg.weight_scale == 'power2':
            weight_scale_func = _power2_scale
        else:
            raise ValueError("Unknown weight scale mode {}".format(cfg.weight_scale))

        return _set_params(mod, input_scale_func, weight_scale_func)
    return wrapped_func
