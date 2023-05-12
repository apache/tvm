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
import tvm.driver
from tvm.ir import IRModule

from . import _quantize
from . import quantize
from .. import op as _op
from .. import expr as _expr
from .. import analysis as _analysis
from .. import build_module as _build_module
from ...contrib import graph_executor
from .kl_divergence import _find_scale_by_kl


def _get_profile_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateStatsCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime


def collect_stats(mod, dataset, chunk_by=-1):
    """Given an annotated graph, create a profile graph to collect profile data from the
    calibration dataset. This pass collects simulated_quantize op input into a tuple.
    Simulated_quantize ops are rewritten to identity mode. The tuple is the output of the profile
    graph.

    Parameters
    ----------
    mod: Module
        The simulation graph after annotation.

    dataset: Iterable[NDArray]
        The calibration dataset.

    chunk_by: optional, int
        The size of chunk to be returned in one iteration. It is meant to be
        used for reducing memory usage. If not specified, return samples for
        all layers in one chunk.

    Returns
    -------
    ret: Iterable[list of ndarray]
        List of output data of each layer, chunked by the chunk_by parameter
    """
    logging.info("collecting statistics for calibration...")
    runtime = _get_profile_runtime(mod)
    num_outputs = runtime.get_num_outputs()
    chunk_by = num_outputs if chunk_by == -1 else chunk_by

    for i in range(0, num_outputs, chunk_by):
        outputs = [[] for i in range(min(chunk_by, num_outputs - i))]
        for batch in dataset:
            runtime.set_input(**batch)
            runtime.run()
            for j in range(i, min(i + chunk_by, num_outputs)):
                outputs[j - i].append(runtime.get_output(j).numpy())
        yield [np.concatenate(output).reshape(-1) for output in outputs]


def _kl_scale(mod, dataset):
    cfg = quantize.current_qconfig()
    chunk_by = cfg.calibrate_chunk_by
    scales = []
    for samples in collect_stats(mod, dataset, chunk_by):
        logging.info("finding threshold with kl for calibration...")
        with mp.Pool() as pool:
            scales += list(pool.map(_find_scale_by_kl, samples))

    def func(_):
        scale = scales[func.scale_idx]
        func.scale_idx += 1
        return scale

    func.scale_idx = 0

    return func


def _find_scale_by_percentile(arr, percentile=0.99999):
    assert isinstance(arr, np.ndarray)
    x = np.abs(arr)
    max_k = int(x.size * percentile)
    return np.partition(x, max_k)[max_k]


def _percentile_scale(mod, dataset):
    cfg = quantize.current_qconfig()
    chunk_by = cfg.calibrate_chunk_by
    scales = []
    for samples in collect_stats(mod, dataset, chunk_by):
        logging.info("finding threshold with percentile for calibration...")
        with mp.Pool() as pool:
            scales += list(pool.map(_find_scale_by_percentile, samples))

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
        """visitor function for traverse"""
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
                return _expr.const(val, "float32")

            valid_range = 2**valid_bit
            const_params[ndom_scale] = _make_const(scale / valid_range)
            const_params[nclip_min] = _make_const(-(valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)
    main_func = _expr.bind(main_func, const_params)
    func_dict = {}
    for global_var, func in mod.functions.items():
        if global_var.name_hint != "main":
            func_dict[global_var] = func
    return IRModule.from_expr(main_func, func_dict)


# weight scale functions
def _power2_scale(sq_call):  # pylint: disable=unused-argument
    """calculate weight scale with nearest mode-2 scale"""
    var = sq_call.args[0]
    assert isinstance(var, _expr.Constant)
    val = np.amax(np.abs(var.data.numpy()))
    return 2 ** np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0


def _max_scale(sq_call):
    """calculate weight scale with maximum absolute value"""
    var = sq_call.args[0]
    assert isinstance(var, _expr.Constant)
    val = np.amax(np.abs(var.data.numpy()))
    return val


# input scale functions
def _global_scale(sq_call):  # pylint: disable=unused-argument
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

    def wrapped_func(mod, _):
        """make transform.module pass happy"""
        cfg = quantize.current_qconfig()

        if cfg.calibrate_mode == "kl_divergence":
            input_scale_func = _kl_scale(mod, dataset)
        elif cfg.calibrate_mode == "global_scale":
            input_scale_func = _global_scale
        elif cfg.calibrate_mode == "percentile":
            input_scale_func = _percentile_scale(mod, dataset)
        else:
            raise ValueError(f"Unknown calibrate mode {cfg.calibrate_mode}")

        if cfg.weight_scale == "max":
            weight_scale_func = _max_scale
        elif cfg.weight_scale == "power2":
            weight_scale_func = _power2_scale
        else:
            raise ValueError(f"Unknown weight scale mode {cfg.weight_scale}")

        return _set_params(mod, input_scale_func, weight_scale_func)

    return wrapped_func
