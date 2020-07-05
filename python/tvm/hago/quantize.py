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
import math
import numpy as np
import logging
from collections import namedtuple

SimulatedQuantizeParams = namedtuple("SimulatedQuantizeParams", ['in_scale',
                                                                 'out_scale',
                                                                 'clip_min',
                                                                 'clip_max',
                                                                 'in_dtype',
                                                                 'out_dtype',
                                                                 ])

def select_constraint(graph, hardware, topology, bits):
    def select(node, in_bits, hardware):
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

            if selected:
                return cstr
        raise ValueError("No feasible constraint")
        return None

    print('\nselect constraints')
    node2idx = build_node_index(graph)
    edge2bit = build_edge_dict(graph, bits, topology.edge_conds)
    constraints = [None] * len(node2idx)

    def fvisit(node):
        if isinstance(node, relay.Call):
            if not topology.node_conds[node2idx[node]]:
                return
            # prepare in_bits
            print('---------')
            print(node_str(node, node2idx))
            in_bits = [edge2bit[(src, node)] for src in node.args]
            print('  in bits: {}'.format(in_bits))
            cstr = select(node, in_bits, hardware)
            print('  {0}'.format(cstr))
            assert len(cstr.odtypes) == 1 
            constraints[node2idx[node]] = cstr
    relay.analysis.post_order_visit(graph, fvisit)
    return constraints  


def infer_quantized_dtypes(graph, constraints):
    def assign_dtype(dtypes, idx, dtype):
        if dtypes[idx] is not None:
            assert dtypes[idx] == dtype, "previous dtype: {}, current dtype: {}".format(dtypes[idx], dtype)
        dtypes[idx] = dtype

    node2idx = build_node_index(graph)
    edge2idx = build_edge_index(graph)

    # provided output data type
    prov_dtypes = [None] * len(node2idx)
    # data type requirement from succeeded ops
    req_dtypes = [None] * len(edge2idx)

    # TODO(ziheng) consider datatype casting instead of only bit requirement
    def fvisit(node):
        if isinstance(node, (relay.Var, relay.Constant)):
            prov_dtypes[node2idx[node]] = DataType("float32")
        if isinstance(node, relay.Call):
            cstr = constraints[node2idx[node]]
            if cstr is None:
                # use float computation
                prov_dtypes[node2idx[node]] = DataType('float32')
                for src in node.args:
                    eidx = edge2idx[(src, node)]
                    req_dtypes[eidx] = DataType('float32')
                return

            assign_dtype(prov_dtypes, node2idx[node], cstr.odtype(0))

            for src, dtype in zip(node.args, cstr.idtypes):
                idx = edge2idx[(src, node)]
                assign_dtype(req_dtypes, idx, dtype)
    relay.analysis.post_order_visit(graph, fvisit)
    return prov_dtypes, req_dtypes 




def prerequisite_optimize(func, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = tvm.transform.Sequential([relay.transform.SimplifyInference(),
                                         relay.transform.FoldConstant(),
                                         relay.transform.FoldScaleAxis(),
                                         relay.transform.CanonicalizeOps(),
                                         relay.transform.FoldConstant()])

    if params:
        func = relay.build_module.bind_params_by_name(func, params)

    with tvm.transform.PassContext(opt_level=3):
        mod = tvm.IRModule.from_expr(func)
        mod = optimize(mod)
    return mod['main']


class Simulator(tvm.relay.ExprMutator):
    def __init__(self, graph, topology, constraints):
        """generate simulate graph"""
        super().__init__()
        self.graph = graph
        self.topology = topology
        self.constraints = constraints

        self._name_cnt = 0
        self._node2idx = build_node_index(graph)
        self._edge2idx = build_edge_index(graph)
        self._prov_dtypes, self._req_dtypes = infer_quantized_dtypes(graph, constraints)

        self.internal_param_nodes = []
        self.output_param_nodes = []
        self.simulated_graph = self.visit(graph)
        self._runtime = None


    def eval(self, bits, thresholds, dataset, ctx, target):
        """compile simulated model and run it on the dataset"""
        if self._runtime is None:
            # print(self.simulated_graph)
            self._runtime = relay.create_executor("graph", ctx=ctx, target=target).evaluate(self.simulated_graph)

        # prepare parameters
        internal_params, output_params = self.calculate_params(bits, thresholds)
        param_map = {} 
        for nodes, p in zip(self.internal_param_nodes, internal_params):
            vals = [p.in_scale, p.out_scale, p.clip_min, p.clip_max]
            for node, val in zip(nodes, vals):
                param_map[node.name_hint] = tvm.nd.array(np.array(val, 'float32'))
        for nodes, p in zip(self.output_param_nodes, output_params):
            vals = [p.in_scale, p.out_scale, p.clip_min, p.clip_max]
            for node, val in zip(nodes, vals):
                param_map[node.name_hint] = tvm.nd.array(np.array(val, 'float32'))

        outputs = []
        for batch_id, batch in enumerate(dataset):
            out = self._runtime(batch['data'], **param_map).asnumpy()
            outputs.append(out)
        return outputs


    def visit_call(self, node):
        new_node = super().visit_call(node)
        new_args = []
        for idx, src in enumerate(node.args):
            # sq's input dtype is the predecessor op's output dtype (provided dtype)
            in_dtype = self._prov_dtypes[self._node2idx[src]]
            # sq's output dtype is the successor op's input dtype (required dtype)
            out_dtype = self._req_dtypes[self._edge2idx[(src, node)]]

            in_scale = relay.var('in_scale' + str(self._name_cnt), 'float32')
            out_scale = relay.var('out_scale' + str(self._name_cnt), 'float32')
            clip_min = relay.var('clip_min' + str(self._name_cnt), 'float32')
            clip_max = relay.var('clip_max' + str(self._name_cnt), 'float32')
            self._name_cnt += 1
            self.internal_param_nodes.append((in_scale, out_scale, clip_min, clip_max))
            new_arg = _quantize.simulated_quantize(new_node.args[idx],
                                                   in_scale,
                                                   out_scale,
                                                   clip_min,
                                                   clip_max,
                                                   in_dtype,
                                                   out_dtype,
                                                   True,
                                                   "round")
            new_args.append(new_arg)
        new_node = relay.Call(new_node.op, new_args, new_node.attrs)
        return new_node

    def visit_function(self, fn):
        # dequantize output
        new_fn = super().visit_function(fn)
        assert isinstance(new_fn.body, relay.Call)
        in_dtype = self._prov_dtypes[self._node2idx[fn.body]]

        in_scale = relay.var('in_scale' + str(self._name_cnt), 'float32')
        out_scale = relay.var('out_scale' + str(self._name_cnt), 'float32')
        clip_min = relay.var('clip_min' + str(self._name_cnt), 'float32')
        clip_max = relay.var('clip_max' + str(self._name_cnt), 'float32')
        self._name_cnt += 1
        self.output_param_nodes.append((in_scale, out_scale, clip_min, clip_max))
        new_body = _quantize.simulated_quantize(new_fn.body,
                                                in_scale,
                                                out_scale,
                                                clip_min,
                                                clip_max,
                                                in_dtype,
                                                DataType('float32'),
                                                True,
                                                "round")

        new_params = relay.analysis.free_vars(new_body)
        return relay.Function(
            new_params,
            new_body,
            new_fn.ret_type,
            new_fn.type_params,
            new_fn.attrs)

    def calculate_params(self, bits, thresholds):
        """calculate parameters of simulated quantize op from bits and thresholds"""
        graph, topology, constraints = self.graph, self.topology, self.constraints
        sign_bit = 1
        edge2idx  = build_edge_index(graph)
        node2idx  = build_node_index(graph)
        assert len(thresholds) == len(node2idx)
        edge2bit = build_edge_dict(graph, bits, topology.edge_conds)

        # graph, topology, bits, constraints
        prov_dtypes, req_dtypes = infer_quantized_dtypes(graph, constraints)

        # num of edges + number of output edges
        internal_params, output_params = [], []
        def infer_scale_for_node(node):
            if isinstance(node, (relay.Var, relay.Constant)):
                scale = 1.0
                return scale
            assert isinstance(node, relay.Call)
            finfer_scale = node.op.get_attr('FHagoInferScale')
            assert finfer_scale
            input_scales = [internal_params[edge2idx[(src, node)]].out_scale for src in node.args]
            scale = finfer_scale(input_scales)
            return scale

        print('\ncalculate parameters')
        def fvisit(node):
            if isinstance(node, relay.Call):
                for src in node.args:
                    eidx = edge2idx[(src, node)]
                    in_scale = infer_scale_for_node(src)
                    in_dtype = prov_dtypes[node2idx[src]]
                    out_dtype = req_dtypes[eidx]

                    print('---------')
                    print(edge_str((src, node), node2idx))
                    if 'float' in str(out_dtype):
                        # dequantize
                        out_scale = 1.0
                        clip_min = float('nan')
                        clip_max = float('nan')
                        print('  not quantized'.format(edge_str((src, node))))
                    else:
                        bit = edge2bit[(src, node)]
                        integer_range = 2 ** (bit - sign_bit)
                        thold = thresholds[node2idx[src]]
                        out_scale = thold / integer_range 
                        clip_min = - (integer_range - 1)
                        clip_max =    integer_range - 1
                        print('  bit={}, threshold={}'.format(bit, thold))

                    # print("{}[{}] -> {}[{}]".format(node_str(src), in_dtype, node_str(node), out_dtype))
                    param = SimulatedQuantizeParams(in_scale, out_scale, clip_min, clip_max,
                                                    in_dtype, out_dtype)
                    print('  {}'.format(param))
                    in_cond, in_expo = exponent_based_two(param.in_scale)
                    assert in_cond, "scale={}, expo={}\nparam\{}".format(param.in_scale, in_expo, param)
                    out_cond, out_expo = exponent_based_two(param.out_scale)
                    assert out_cond, "scale={}, expo={}\nparam={}".format(param.out_scale, out_expo, param)
                    internal_params.append(param)
                return
            if isinstance(node, relay.Function):
                # handle output of function 
                assert isinstance(node.body, relay.Call) 
                node = node.body
                print('---------')
                print("{} -> OUT".format(node_str(node, node2idx)))
                in_scale = infer_scale_for_node(node)
                in_dtype = prov_dtypes[node2idx[node]]
                out_dtype = DataType('float32')
                out_scale = 1.0
                param = SimulatedQuantizeParams(in_scale, out_scale, float('nan'), float('nan'),
                                                in_dtype, out_dtype)
                print('  {}'.format(param))
                output_params.append(param)
                return
        relay.analysis.post_order_visit(graph, fvisit)

        if current_qconfig().threshold_estimate_method == 'power_of_two_range':
            # check all scale need to be power of 2
            print('check scale to be power of two...')
            params = internal_params + output_params
            for param in params:
                in_cond, in_expo = exponent_based_two(param.in_scale)
                assert in_cond, "scale={}, expo={}\nparam\{}".format(param.in_scale, in_expo, param)
                out_cond, out_expo = exponent_based_two(param.out_scale)
                assert out_cond, "scale={}, expo={}\nparam={}".format(param.out_scale, out_expo, param)

        return internal_params, output_params


    def bind_simulated_graph(self, bits, thresholds):
        # prepare parameters
        internal_params, output_params = self.calculate_params(bits, thresholds)
        param_map = {} 
        for nodes, p in zip(self.internal_param_nodes, internal_params):
            vals = [p.in_scale, p.out_scale, p.clip_min, p.clip_max]
            for node, val in zip(nodes, vals):
                param_map[node.name_hint] = tvm.nd.array(np.array(val, 'float32'))
        for nodes, p in zip(self.output_param_nodes, output_params):
            vals = [p.in_scale, p.out_scale, p.clip_min, p.clip_max]
            for node, val in zip(nodes, vals):
                param_map[node.name_hint] = tvm.nd.array(np.array(val, 'float32'))
        binded_simulated_graph = relay.build_module.bind_params_by_name(self.simulated_graph, param_map)
        return binded_simulated_graph


class Realizer(tvm.relay.ExprMutator):
    def __init__(self, original_graph, simulated_graph, constraints):
        super().__init__()
        self._original_graph = original_graph
        self._simulated_graph = simulated_graph
        self._constraints = constraints
        self._node2idx = build_node_index(original_graph)
        self._snode2idx = build_node_index(simulated_graph)  # for printing debug info
        self._snode2node = build_node_mapping(simulated_graph, original_graph)

    def realize(self):
        return self.visit(self._simulated_graph)

    def visit_call(self, node):
        new_node = super().visit_call(node)
        if node.op.name == "hago.simulated_quantize":
            print('---------')
            print('simulated_quantize({})'.format(node_str(node.args[0], self._snode2idx)))
            new_node = self._realize_simulated_quantize(new_node)
            return new_node
        nidx = self._node2idx[self._snode2node[node]]
        cstr = self._constraints[nidx]
        frealize = node.op.get_attr("FHagoRealize")
        if frealize and cstr is not None:
            in_dtypes = list(map(str, cstr.idtypes))
            out_dtypes = list(map(str, cstr.odtypes))
            new_node = frealize(new_node, in_dtypes, out_dtypes)
        return new_node

    def _realize_simulated_quantize(self, node):
        data, in_scale, out_scale, clip_min, clip_max = node.args
        attrs = node.attrs
        in_scale = to_scalar(in_scale)
        out_scale = to_scalar(out_scale)
        clip_min = to_scalar(clip_min)
        clip_max = to_scalar(clip_max)
        in_dtype = attrs.in_dtype
        out_dtype = attrs.out_dtype
        print('  in_scale: {}'.format(in_scale))
        print('  out_scale: {}'.format(out_scale))
    
        if in_dtype == 'float32' and out_dtype == 'float32':
            # do nothing
            return data
        elif out_dtype == 'float32':
            # dequantize
            assert out_scale == 1.0
            data = relay.cast(data, out_dtype)
            data = relay.multiply(data, relay.const(in_scale, out_dtype))
            return data
        elif in_dtype == 'float32':
            # quantize
            assert in_scale == 1.0
            data = relay.divide(data, relay.const(out_scale, in_dtype))
            data = relay.round(data)
            print('  clip min: {}'.format(clip_min))
            print('  clip max: {}'.format(clip_max))
            data = relay.clip(data, clip_min, clip_max)
            data = relay.cast(data, out_dtype)
            return data
        elif in_scale == out_scale and in_dtype == out_dtype:
            # do nothing
            # TODO(ziheng) whether to clip?
            return data
        else:
            # requantize
            dtype = in_dtype
            if TVMType(out_dtype).bits > TVMType(in_dtype).bits:
                # pre-casting
                data = relay.cast(data, out_dtype)
                dtype = out_dtype
            data = self._transform_scale(data, in_scale, out_scale, dtype)
            print('  clip min: {}'.format(clip_min))
            print('  clip max: {}'.format(clip_max))
            data = relay.clip(data, clip_min, clip_max)
            if dtype != out_dtype:
                data = relay.cast(data, out_dtype)
            return data

    def _transform_scale(self, data, in_scale, out_scale, dtype):
        """calculate `data * in_scale / out_scale`"""
        if math.isclose(in_scale, out_scale):
            return data

        def use_shift(in_val, out_val):
            # whether to use shift, consider floating point numeric error
            in_cond, in_exp = exponent_based_two(in_val) 
            out_cond, out_exp = exponent_based_two(out_val) 
            if in_cond and out_cond:
                return True, in_exp - out_exp 
            return exponent_based_two(in_val / out_val)

        factor = in_scale / out_scale
        do_shift, shift_factor = use_shift(in_scale, out_scale)
        print('  factor: {}'.format(factor))
        print('  shift_factor: {}'.format(shift_factor))
        print('  rounded shift_factor: {}'.format(round(shift_factor)))
        if do_shift:
            if shift_factor > 0:
                print('  use left shift')
                out = relay.left_shift(data, relay.const(round(shift_factor), dtype))
            else:
                print('  use right shift')
                # TODO(ziheng) statistic bias
                # add bias for rounding
                shift_factor = - round(shift_factor)
                out = data + relay.const(2**(shift_factor - 1), dtype)
                out = relay.right_shift(out, relay.const(shift_factor, dtype))
        elif math.isclose(factor, round(factor)):
            print('  use integer multiply')
            # TODO(ziheng) overflow risk
            out = relay.multiply(data, relay.const(factor, dtype))
        else:
            print('  use float multiply')
            # TODO(ziheng) rounding
            out = relay.cast(data, "float32")
            out = relay.multiply(out, relay.const(factor, 'float32'))
            out = relay.round(out)
            out = relay.cast(out, dtype)
            raise ValueError
        return out

    
class Quantizer(object):
    def __init__(self, graph, hardware, topology, bits, thresholds):
        self.original_graph = graph
        self.simulated_graph = None
        self.quantized_graph = None

        self.hardware = hardware
        self.topology = topology
        self.bits = bits
        self.thresholds = thresholds

    def simulate(self):
        self.constraints = select_constraint(self.original_graph, self.hardware, self.topology, self.bits)
        self._simulator = Simulator(self.original_graph, self.topology, self.constraints)
        self.simulated_graph = self._simulator.bind_simulated_graph(self.bits, self.thresholds)
        return self.simulated_graph

    def quantize(self):
        if self.simulated_graph is None:
            self.simulate()
        self._realizer = Realizer(self.original_graph, self.simulated_graph, self.constraints)
        self.quantized_graph = self._realizer.realize()
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(self.quantized_graph))
        self.quantized_graph = mod['main']
        return self.quantized_graph

def create_quantizer(graph, hardware, strategy):
    # check model hash
    model_hash = tvm.ir.structural_hash(graph)
    assert model_hash == strategy.model_hash
    return Quantizer(graph, hardware, strategy.topology, strategy.bits, strategy.thresholds)
