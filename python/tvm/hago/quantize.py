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
#pylint: disable=unused-argument
"""Automatic quantization toolkit."""
from __future__ import absolute_import

from . import _quantize
from .. import relay
from .base import *
from .hardware import *

import tvm
import sys
import numpy as np
from collections import namedtuple

SimulatedQuantizeParams = namedtuple("SimulatedQuantizeParams", ['out_scale',
                                                                 'in_scale',
                                                                 'clip_min',
                                                                 'clip_max',
                                                                 'out_dtype',
                                                                 'in_dtype',
                                                                 ])


def calculate_params(graph, hardware, topology, bits, thresholds):
    """calculate parameters of simulated quantize op from bits and thresholds"""
    # integer_range = 2 ^ (bit - sign_bit) 
    # scale = threshold / integer_range
    # clip_min = - (integer_range - 1)
    # clip_max =   (integer_range - 1)

    sign_bit = 1

    edge2idx  = build_edge_index(graph)
    node2idx  = build_node_index(graph)
    node2edges = build_node2edges(graph)
    edge2bit = complete_dict(bits, topology.edge2cond)
    assert len(thresholds) == len(node2idx)

    prov_dtypes, req_dtypes = infer_quantized_dtypes(graph, hardware, topology, edge2bit)
    assert len(prov_dtypes) == len(node2idx)
    assert len(req_dtypes) == len(edge2idx)

    op_params = []
    def infer_scale_for_node(node):
        if isinstance(node, (relay.Var, relay.Constant)):
            scale = 1.0
            return scale
        assert isinstance(node, relay.Call)
        finfer_scale = node.op.get_attr('FHagoInferScale')
        assert finfer_scale
        input_scales = [op_params[edge2idx[(src, node)]].out_scale for src in node.args]
        scale = finfer_scale(input_scales)
        return scale

    def fvisit(node):
        if isinstance(node, relay.Call):
            for src in node.args:
                eidx = edge2idx[(src, node)]
                in_scale = infer_scale_for_node(src)
                in_dtype = prov_dtypes[node2idx[src]]
                out_dtype = req_dtypes[eidx]

                if 'float' in str(out_dtype):
                    # dequantize
                    out_scale = 1.0
                    clip_min = -sys.float_info.max
                    clip_max = sys.float_info.max
                else:
                    bit = edge2bit[(src, node)]
                    integer_range = 2 ** (bit - sign_bit)
                    thold = thresholds[node2idx[src]]
                    out_scale = thold / integer_range 
                    clip_min = - float(integer_range - 1)
                    clip_max =   float(integer_range - 1)

                print("{}[{}] -> {}[{}]".format(node_str(src), in_dtype, node_str(node), out_dtype))
                param = SimulatedQuantizeParams(out_scale, in_scale, clip_min, clip_max,
                                                out_dtype, in_dtype)
                op_params.append(param)
    relay.analysis.post_order_visit(graph, fvisit)
    return op_params


def infer_quantized_dtypes(graph, hardware, topology, edge2bit):
    def select_constraint(in_bits, out_bits, hardware, node):
        # assume constraints have been sorted
        # need to handle None
        for cstr in integer_constraints(hardware[node.op]):
            selected = True
            for dtype, bit in zip(cstr.idtypes, in_bits):
                if bit is None:
                    continue
                if bit > dtype.bits:
                    selected = False
                    break
            for dtype, bit in zip(cstr.odtypes, out_bits):
                if bit is None:
                    continue
                if bit > dtype.bits:
                    selected = False
                    break

            if selected:
                return cstr
        raise ValueError("No feasible constraint")
        return None

    def assign_dtype(dtypes, idx, dtype):
        if dtypes[idx] is not None:
            assert dtypes[idx] == dtype, "previous dtype: {}, current dtype: {}".format(dtypes[idx], dtype)
        dtypes[idx] = dtype

    node2idx = build_node_index(graph)
    edge2idx = build_edge_index(graph)
    node2edges = build_node2edges(graph)

    # provided output data type
    prov_dtypes = [None] * len(node2idx)
    # data type requirement from succeeded ops
    req_dtypes = [None] * len(edge2idx)

    # TODO(ziheng) consider datatype casting instead of only bit requirement
    def fvisit(node):
        if isinstance(node, (relay.Var, relay.Constant)):
            prov_dtypes[node2idx[node]] = TVMType("float32")
        if isinstance(node, relay.Call):
            print(node.op.name)
            if not topology.node2cond[node]:
                # use float computation
                prov_dtypes[node2idx[node]] = TVMType('float32')
                for src in node.args:
                    eidx = edge2idx[(src, node)]
                    req_dtypes[eidx] = TVMType('float32')
                return

            # prepare in_bits, out_bits
            in_bits = [edge2bit[(src, node)] for src in node.args]
            # op can be referred multiple times
            out_bits = [edge2bit[edge] for edge in node2edges[node]]
            print('in bits: {}'.format(in_bits))
            print('out bits: {}'.format(out_bits))
            cstr = select_constraint(in_bits, out_bits, hardware, node)
            print('select {0}'.format(cstr))
            assert len(cstr.odtypes) == 1 
            assign_dtype(prov_dtypes, node2idx[node], cstr.odtype(0))

            for src, dtype in zip(node.args, cstr.idtypes):
                idx = edge2idx[(src, node)]
                assign_dtype(req_dtypes, idx, dtype)
    relay.analysis.post_order_visit(graph, fvisit)

    return prov_dtypes, req_dtypes 


def prerequisite_optimize(mod, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = relay.transform.Sequential([relay.transform.SimplifyInference(),
                                           relay.transform.FoldConstant(),
                                           relay.transform.FoldScaleAxis(),
                                           relay.transform.CanonicalizeOps(),
                                           relay.transform.FoldConstant()])

    if params:
        mod['main'] = bind_params(mod['main'], params)

    with relay.transform.PassContext(opt_level=3):
        mod = optimize(mod)
    return mod


def simulate(graph, op_params):
    class Simulator(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()

        def create_simulated_graph(self, graph, op_params):
            self._op_params = op_params
            self._edge2idx = build_edge_index(graph)
            return self.visit(graph)

        def visit_call(self, node):
            new_node = super().visit_call(node)
            new_args = []
            for idx, src in enumerate(node.args):
                param = self._op_params[self._edge2idx[(src, node)]]
                new_arg = _quantize.simulated_quantize(new_node.args[idx],
                                                       tvm.relay.const(param.out_scale),
                                                       tvm.relay.const(param.in_scale),
                                                       tvm.relay.const(param.clip_min),
                                                       tvm.relay.const(param.clip_max),
                                                       param.out_dtype,
                                                       param.in_dtype,
                                                       True,
                                                       "round")
                new_args.append(new_arg)
            return relay.Call(new_node.op, new_args, new_node.attrs)

    simulated_graph = Simulator().create_simulated_graph(graph, op_params)
    # print('before simulating')
    # print(graph)
    # print('creare simulated graph')
    # print(simulated_graph)
    return simulated_graph


def realize(graph, bits, thresholds):
    op_params = calculate_params(graph, bits, thresholds, hw_desc)

    class Realizer(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()

        def visit(self, e):
            new_e = super().visit(e)
            if isinstance(e, op.simulated_quantze):
                new_e = quantize(new_e)
            elif registered(e):
                frealize(e)

    print(graph)
    return graph


def quantize(graph, bits, thresholds, hw_desc):
    cfg = current_qconfig()
    op_params = calculate_params(graph, bits, thresholds, hw_desc)
    simulated_graph = simulate(graph, op_params)
    if cfg.do_simulation:
        return simulated_graph
    else:
        qgraph = realize(simulated_graph)
        return qgraph
        raise ValueError
