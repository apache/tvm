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


def calculate_params(graph, bits, thresholds, hw_desc):
    """calculate parameters of simulated quantize op from bits and thresholds"""
    # integer_range = 2 ^ (bit - sign_bit) 
    # scale = threshold / integer_range
    # clip_min = - (integer_range - 1)
    # clip_max =   (integer_range - 1)

    sign_bit = 1
    # print('check threshold type')
    # for thold in thresholds:
    #     if not isinstance(thold, float):
    #         print(type(thold))

    edge2idx, num_edges  = build_edge_index(graph)
    node2idx, num_nodes  = build_node_index(graph)
    node2edges = build_node2edges(graph)
    assert len(bits) == num_edges
    assert len(thresholds) == num_nodes

    prov_dtypes, req_dtypes = infer_quantized_dtypes(graph, bits, hw_desc)
    assert len(prov_dtypes) == num_nodes
    assert len(req_dtypes) == num_edges
    print('provided dtypes:')
    print(prov_dtypes)
    def fvisit_print(node):
        if isinstance(node, (relay.Var, relay.Constant, relay.Call)):
            print("{}: {}".format(node_str(node), prov_dtypes[node2idx[node]]))
    relay.analysis.post_order_visit(graph, fvisit_print)

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
                thold = thresholds[node2idx[src]]
                eidx = edge2idx[(src, node)]
                bit = bits[eidx]
                integer_range = 2 ** (bit - sign_bit)
                out_scale = thold / integer_range 
                in_scale = infer_scale_for_node(src)
                clip_min = - float(integer_range - 1)
                clip_max =   float(integer_range - 1)
                out_dtype = req_dtypes[eidx]
                in_dtype = prov_dtypes[node2idx[src]]

                print("{} -> {}: in_dtype={}, out_dtype={}".format(node_str(src), node_str(node), in_dtype, out_dtype))
                param = SimulatedQuantizeParams(out_scale, in_scale, clip_min, clip_max,
                                                out_dtype, in_dtype)
                op_params.append(param)
    relay.analysis.post_order_visit(graph, fvisit)

    return op_params


def infer_quantized_dtypes(graph, bits, hw_desc):
    def select_constraint(input_bits, output_bits, constraints):
        # assume constraints have been sorted
        for cstr in constraints:
            selected = True
            for dtype, bit in zip(cstr.idtypes, input_bits):
                if bit > dtype.bits:
                    selected = False
                    break
            for dtype, bit in zip(cstr.odtypes, output_bits):
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

    node2idx, num_nodes = build_node_index(graph)
    edge2idx, num_edges = build_edge_index(graph)
    node2edges = build_node2edges(graph)

    assert len(bits) == num_edges
    # provided output data type
    prov_dtypes = [None] * num_nodes
    # data type requirement from succeeded ops
    req_dtypes = [None] * num_edges

    print_bits_info(graph, bits)

    # fill prov_dtypes and req_dtypes according to the constraints
    def fvisit_infer_dtype(node):
        if isinstance(node, (relay.Var, relay.Constant)):
            prov_dtypes[node2idx[node]] = "float32"

        if isinstance(node, relay.Call) and node.op.name in hw_desc.ops:
            # print(node.op.name)
            input_bits = [bits[edge2idx[(src, node)]] for src in node.args]
            # op can be referred multiple times
            output_bits = [bits[edge2idx[edge]] for edge in node2edges[node]]
            # print('input_bits: {}'.format(input_bits))
            # print('output_bits: {}'.format(output_bits))
            # assume all op has only one output
            if output_bits != []:
                # handle output node, which does not have output edge/bit 
                output_bits = [max(output_bits)]  # select the biggest bit

            cstr = select_constraint(input_bits, output_bits, hw_desc[node.op.name])
            print('select {0}'.format(cstr))
            assert len(cstr.odtypes) == 1 
            assign_dtype(prov_dtypes, node2idx[node], cstr.odtype(0))

            for src, dtype in zip(node.args, cstr.idtypes):
                idx = edge2idx[(src, node)]
                assign_dtype(req_dtypes, idx, dtype)
    relay.analysis.post_order_visit(graph, fvisit_infer_dtype)

    # # (TODO) ziheng 
    # # fill prov_dtypes and req_dtypes according each other
    # def fvisi_infer(node):
    #     out_edges = node2edges[node]
    #     reqs = req_dtypes[edge2idx[edge] for edge in out_edges]

    # relay.analysis.post_order_visit(graph, fvisit_infer_dtype)


    # since prov_dtype has been only used for checking overflow,
    # so if it is not specified by constraint, we make it as int32.
    for idx, dtype in enumerate(prov_dtypes):
        if dtype is None:
            prov_dtypes[idx] = DType('int32')

    # take the nearest feasible data type for req_dtypes
    for idx, bit in enumerate(bits):
        if req_dtypes[idx] is not None:
            continue
        if bit <= 8:
            req_dtypes[idx] = DType('int8')
        elif bit > 8 and bit <= 16:
            req_dtypes[idx] = DType('int16')
        elif bit > 16 and bit <= 32:
            req_dtypes[idx] = DType('int32')
        else:
            raise ValueError
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
            self._edge2idx, _ = build_edge_index(graph)
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
