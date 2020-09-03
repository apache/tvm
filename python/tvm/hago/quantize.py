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
from .topology import Topology

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

def select_desc(graph, hardware, topology, bits):
    def select(node, in_bits, hardware):
        # assume descriptors have been sorted
        # need to handle None
        for desc in hardware.list_integer_descs(node.op):
            selected = True
            in_dtypes = desc.in_dtypes
            if not isinstance(in_dtypes, list):
                in_dtypes = [in_dtypes] * len(in_bits)
            for dtype, bit in zip(in_dtypes, in_bits):
                if bit is None:
                    continue
                if bit > dtype.bits:
                    selected = False
                    break

            if selected:
                return desc
        raise ValueError("No feasible constraint")
        return None

    print('\nselect descriptor')
    node2idx = topology.node2idx()
    edge2bit = topology.build_edge_info(bits)
    descs = [None] * len(node2idx)

    def fvisit(node):
        if isinstance(node, relay.Call):
            if not topology.is_quantized_node(node):
                return
            # prepare in_bits
            print('---------')
            print(node_str(node, node2idx))
            in_bits = [edge2bit[edge] for edge in list_in_edges(node)]
            for bit in in_bits:
                assert bit is not None
            print('  in bits: {}'.format(in_bits))
            desc = select(node, in_bits, hardware)
            print('  {0}'.format(desc))
            descs[node2idx[node]] = desc
    relay.analysis.post_order_visit(graph, fvisit)
    return descs  


def infer_quantized_dtypes(topology, constraints):
    def assign_dtype(dtypes, idx, dtype):
        if dtypes[idx] is not None:
            assert dtypes[idx] == dtype, "previous dtype: {}, current dtype: {}".format(dtypes[idx], dtype)
        dtypes[idx] = dtype

    node2idx = topology.node2idx()
    edge2idx = topology.edge2idx()

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
                for edge in list_in_edges(node):
                    eidx = edge2idx[edge]
                    req_dtypes[eidx] = DataType('float32')
            else:
                assign_dtype(prov_dtypes, node2idx[node], cstr.out_dtype(0))

                edges = list(list_in_edges(node))
                in_dtypes = cstr.in_dtypes
                if not isinstance(in_dtypes, list):
                    in_dtypes = [in_dtypes] * len(edges)

                for edge, dtype in zip(edges, in_dtypes):
                    eidx = edge2idx[edge]
                    assign_dtype(req_dtypes, eidx, dtype)
    relay.analysis.post_order_visit(topology.graph, fvisit)
    return prov_dtypes, req_dtypes 


class CalibrationDataset(object):
    def __init__(self, batches):
        assert isinstance(batches, list) and isinstance(batches[0], dict)
        self.batches = batches

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= len(self.batches):
            raise StopIteration
        ret = self.batches[self._counter]
        self._counter += 1
        return ret


def prerequisite_optimize(func, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """

    optimize = tvm.transform.Sequential([relay.transform.CanonicalizeOps(),
                                         relay.transform.SimplifyInference(),
                                         relay.transform.FoldConstant(),
                                         relay.transform.FoldScaleAxis(),
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
        self._node2idx = self.topology.node2idx()
        self._edge2idx = self.topology.edge2idx()
        self._prov_dtypes, self._req_dtypes = infer_quantized_dtypes(topology, constraints)

        self.internal_param_nodes = []
        self.output_param_nodes = []
        self.simulated_graph = self.visit(graph)
        self._runtime = None


    def eval(self, bits, thresholds, dataset, ctx, target):
        """compile simulated model and run it on the dataset"""
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

        if self._runtime is None:
            # print(self.simulated_graph)
            self._runtime = relay.create_executor("graph", ctx=ctx, target=target).evaluate(self.simulated_graph)

        # prepare runtime
        input_keys = [str(param.name_hint) for param in self.simulated_graph.params]
        outputs = []
        for batch_id, batch in enumerate(dataset):
            inputs = {}
            for key in input_keys:
                if key in param_map:
                    inputs[key] = param_map[key]
                else:
                    assert key in batch
                    inputs[key] = batch[key]
            out = self._runtime(**inputs)
            outputs.append(out)
        return outputs

    def create_simulated_quantize(self, input_node, in_dtype, out_dtype):
        in_scale = relay.var('in_scale' + str(self._name_cnt), 'float32')
        out_scale = relay.var('out_scale' + str(self._name_cnt), 'float32')
        clip_min = relay.var('clip_min' + str(self._name_cnt), 'float32')
        clip_max = relay.var('clip_max' + str(self._name_cnt), 'float32')
        self._name_cnt += 1
        self.internal_param_nodes.append((in_scale, out_scale, clip_min, clip_max))
        new_node = _quantize.simulated_quantize(input_node,
                                               in_scale,
                                               out_scale,
                                               clip_min,
                                               clip_max,
                                               in_dtype,
                                               out_dtype,
                                               True,
                                               "round")
        return new_node

    def _get_dtype(self, src, dst):
        # sq's input dtype is the predecessor op's output dtype (provided dtype)
        in_dtype = self._prov_dtypes[self._node2idx[src]]
        # sq's output dtype is the successor op's input dtype (required dtype)
        out_dtype = self._req_dtypes[self._edge2idx[(src, dst)]]
        return in_dtype, out_dtype


    def visit_call(self, node):
        new_node = super().visit_call(node)
        old_args = list_in_nodes(node)
        new_args = list_in_nodes(new_node)
        old2new = {}
        for old_arg, new_arg in zip(old_args, new_args):
            old2new[old_arg] = new_arg

        sim_args = []
        for idx, old_arg in enumerate(node.args):
            if isinstance(old_arg, (relay.Var, relay.Constant, relay.Call)):
                in_dtype, out_dtype = self._get_dtype(old_arg, node)
                new_arg = old2new[old_arg]
                sim_arg = self.create_simulated_quantize(new_arg, in_dtype, out_dtype)
                sim_args.append(sim_arg)

            elif isinstance(old_arg, relay.Tuple):
                sim_arg = []
                for src in old_arg:
                    in_dtype, out_dtype = self._get_dtype(src, node)
                    new_arg = old2new[src]
                    sim_arg.append(self.create_simulated_quantize(new_arg, in_dtype, out_dtype))
                sim_args.append(relay.Tuple(sim_arg))
            else:
                raise ValueError
        new_node = relay.Call(new_node.op, sim_args, new_node.attrs)
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
        edge2idx  = self._edge2idx 
        node2idx  = self._node2idx
        assert len(thresholds) == len(node2idx)
        edge2bit = topology.build_edge_info(bits)

        # graph, topology, bits, constraints
        prov_dtypes, req_dtypes = infer_quantized_dtypes(topology, constraints)

        # num of edges + number of output edges
        internal_params, output_params = [], []
        def infer_scale_for_node(node):
            if not topology.is_quantized_node(node) or \
                isinstance(node, (relay.Var, relay.Constant)):
                scale = 1.0
                return scale
            assert isinstance(node, relay.Call)
            finfer_scale = node.op.get_attr('FHagoInferScale')
            assert finfer_scale, "no FHagoInferScale for {}".format(node.op.name)
            input_scales = [internal_params[edge2idx[edge]].out_scale for edge in list_in_edges(node)]
            scale = finfer_scale(input_scales)
            return scale

        print('\ncalculate parameters')
        def fvisit(node):
            if isinstance(node, relay.Call):
                for edge in list_in_edges(node):
                    src, _ = edge
                    eidx = edge2idx[edge]
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
        self._node2idx = Topology(original_graph).node2idx()
        self._snode2idx = Topology(simulated_graph).node2idx()  # for printing debug info
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
            in_dtypes = [str(cstr.in_dtype(i)) for i in range(node.op.num_inputs)] 
            out_dtypes = [str(cstr.out_dtype(0))]
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
            if DataType(out_dtype).bits > DataType(in_dtype).bits:
                # pre-casting
                data = relay.cast(data, out_dtype)
                dtype = out_dtype

            return relay.qnn.op.requantize(data,
                                           input_scale=relay.const(in_scale, 'float32'),
                                           input_zero_point=relay.const(0, 'int32'),
                                           output_scale=relay.const(out_scale, 'float32'),
                                           output_zero_point=relay.const(0, 'int32'),
                                           out_dtype=out_dtype)
            # TODO - Look if something is missing
            # print('  clip min: {}'.format(clip_min))
            # print('  clip max: {}'.format(clip_max))
            # data = self._transform_scale(data, in_scale, out_scale, dtype)
            # data = relay.clip(data, clip_min, clip_max)
            # if dtype != out_dtype:
            #     data = relay.cast(data, out_dtype)
            # return data

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
        self.constraints = select_desc(self.original_graph, self.hardware, self.topology, self.bits)
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
