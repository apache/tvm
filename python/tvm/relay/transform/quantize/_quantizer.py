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
"""Quantizes functions by inserting qnn.quantize and qnn.dequantize ops."""
from typing import List

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import ffi as pattern_ffi
from tvm.relay.dataflow_pattern import _DFPatternCallback
from tvm.relay.transform.quantize import QuantizerPattern
from tvm.relay.frontend.common import infer_type

from . import _ffi as ffi


class Quantizer:
    """Class that inserts quantize and dequantizes around patterns. It also constructs
    important structures used by the QuantizationCalibrator.

    Parameters
    ----------
    func : relay.Function
        Funtion we are quantizing.

    params : dict of str to NDArray
        Parameters you would pass into relay.build or relay.build_module. We need params
        so that we can run parts of the graph during calibration.

    patterns : List[QuantizerPattern]
        A list of all the patterns that we are going to quantize using this Quantizer.

    skip_first : bool
        If True, we do not quantize the first quantizable pattern in the function. If False,
        we will quantize it.

    skip_last : bool
        If True, we do not quantize the last quantizable pattern in the function. If False,
        we will quantize it.
    """

    def __init__(
        self, func, params, patterns: List[QuantizerPattern], skip_first=True, skip_last=False
    ):
        self.patterns = patterns
        self.original_func = prerequisite_optimize(func, params)

        # num_orig_outputs is -1 if output is not a Tuple, else is length of tuple
        if isinstance(self.original_func.body, tvm.relay.expr.Tuple):
            self.num_orig_outputs = len(self.original_func.body)
        else:
            self.num_orig_outputs = -1

        # Partition the func into sub functions containing the patterns we want to quantize
        partitioned_func = self.original_func
        for q_pattern in self.patterns:
            partitioned_func = q_pattern.pattern.partition(partitioned_func)

        # Get rid of first and last par
        partitioned_func = skip_partitions(partitioned_func, skip_first, skip_last)
        # Add outputs necessary for calibration
        tuple_subgraph_func = partition_outputs(partitioned_func)

        # Lower partitioned funcs and store in a mod
        self.tuple_subgraph_func = lower_partitions(tuple_subgraph_func)

        # Rewrite the multi-output graph to be quantized, and lower partitioned funcs
        outs = rewrite_partitions(self.patterns, tuple_subgraph_func)
        q_tuple_subgraph_func = outs["new_out"]

        # Information about each partition used for calibration
        self.partition_infos = outs["infos_"]

        # Lower quantized partitions and store in a mod
        self.q_tuple_subgraph_func = lower_partitions(q_tuple_subgraph_func)

        # Create a function containing just the quantized original graph
        quantized_func = self.q_tuple_subgraph_func
        if self.num_orig_outputs == -1:
            self.quantized_func = relay.Function(
                self.q_tuple_subgraph_func.params, quantized_func.body.fields[0]
            )
        else:
            tuple_body = relay.Tuple(quantized_func.body.fields[self.num_orig_outputs])
            self.quantized_func = relay.Function(self.q_tuple_subgraph_func.params, tuple_body)


def prerequisite_optimize(func, params=None):
    """Prerequisite optimization passes for quantization. Perform "DynamicToStatic",
    "SimplifyInference", "FoldConstant", "FoldScaleAxis" before quantization.

    Parameters
    ---------
    params : dict of str to NDArray
        Parameters to use during calibration.

    Returns
    -------
    preopt_func : relay.Function
        The original function with optimizations needed before quantization applied.
    """
    optimize = tvm.transform.Sequential(
        [
            relay.transform.DynamicToStatic(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.FoldConstant(),
            relay.transform.EliminateCommonSubexpr(),
        ]
    )

    if params is not None:
        func = relay.build_module.bind_params_by_name(func, params)

    mod = tvm.ir.IRModule.from_expr(func)

    with relay.build_config(opt_level=3):
        mod = optimize(mod)

    return mod["main"]


def partition_outputs(expr):
    return ffi.partition_outputs(expr)


def rewrite_partitions(callbacks, expr):
    return ffi.rewrite_partitions(
        [
            _DFPatternCallback(callback.pattern, callback.callback, callback.require_type)
            for callback in callbacks
        ],
        infer_type(expr),
    )


def lower_partitions(expr):
    return ffi.lower_partitions(expr)


def skip_partitions(expr, skip_first=True, skip_last=True):
    return ffi.skip_partitions(expr, skip_first, skip_last)
