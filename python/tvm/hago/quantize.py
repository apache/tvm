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

from . import _ffi_api
from .. import relay
from .base import *
from .hardware import *
from .topology import Topology, analyze_topology

import tvm
from tvm.tir import expr
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


def simulated_quantize(data,
                       in_scale,
                       out_scale,
                       clip_min,
                       clip_max,
                       in_dtype,
                       out_dtype,
                       axis=None):
    in_scale = relay.const(in_scale, dtype='float32')
    out_scale = relay.const(out_scale, dtype='float32')
    clip_min = relay.const(clip_min, dtype='float32')
    clip_max = relay.const(clip_max, dtype='float32')
    new_node = _ffi_api.simulated_quantize(data, in_scale, out_scale,
        clip_min, clip_max, in_dtype, out_dtype, True, "round", axis)
    return new_node


def select_desc(graph, hardware, topology, bits):
    def select(node, in_bits, hardware):
        # assume descriptors have been sorted
        # need to handle None
        for desc in hardware.list_integer_descs(node):
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
        self._node2kinds = self.topology.node2kind()
        self._node2layouts = self.topology.node2layout()
        self._node2channel_axis = self.topology.node2channel_axis()
        self._edge2idx = self.topology.edge2idx()
        self._prov_dtypes, self._req_dtypes = infer_quantized_dtypes(topology, constraints)

        self.internal_param_nodes = []
        self.output_param_nodes = []
        self.scale_shape = self.topology.infer_scale_shape()
        self.simulated_graph = self.visit(graph)
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(self.simulated_graph))
        self.simulated_graph = mod['main']
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

    def create_simulated_quantize(self, input_node,
        in_dtype, out_dtype, axis, in_scale_shape=(), out_scale_shape=()):
        in_scale = relay.var('in_scale' + str(self._name_cnt), shape=in_scale_shape)
        out_scale = relay.var('out_scale' + str(self._name_cnt), shape=out_scale_shape)
        clip_min = relay.var('clip_min' + str(self._name_cnt))
        clip_max = relay.var('clip_max' + str(self._name_cnt))
        self._name_cnt += 1
        self.internal_param_nodes.append((in_scale, out_scale, clip_min, clip_max))
        new_node = _ffi_api.simulated_quantize(input_node,
                                               in_scale,
                                               out_scale,
                                               clip_min,
                                               clip_max,
                                               in_dtype,
                                               out_dtype,
                                               True,
                                               "round",
                                               axis)
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
                eidx = self._edge2idx[(old_arg, node)]
                iscale_shape, oscale_shape = self.scale_shape[eidx]

                axis = None
                qconfig = current_qconfig()
                use_channel_quantize = qconfig.use_channel_quantize
                assert not (oscale_shape != () and iscale_shape != ()), \
                        "Both input and output scale cannot be tensors simultaneously"
                if oscale_shape != ():
                    # This is the constant node. Find the 'O' dimension.
                    assert use_channel_quantize
                    assert isinstance(old_arg, relay.Constant)
                    assert node.op.name in qconfig.per_channel_ops()
                    const_node_layout = self._node2layouts[old_arg]
                    assert const_node_layout in ('OIHW', 'HWIO')
                    axis = self._node2channel_axis[old_arg]
                elif iscale_shape != ():
                    # This is coming from the per channel quantized operator output, like conv2d
                    # output is per-channel quantized.
                    assert use_channel_quantize
                    assert old_arg.op.name in qconfig.per_channel_ops()
                    data_layout = self._node2layouts[old_arg]
                    assert data_layout in ('NCHW', 'NHWC')
                    axis = self._node2channel_axis[old_arg]

                sim_arg = self.create_simulated_quantize(new_arg,
                                                         in_dtype,
                                                         out_dtype,
                                                         axis,
                                                         iscale_shape,
                                                         oscale_shape)
                sim_args.append(sim_arg)

            elif isinstance(old_arg, relay.Tuple):
                sim_arg = []
                for src in old_arg:
                    in_dtype, out_dtype = self._get_dtype(src, node)
                    new_arg = old2new[src]
                    eidx = self._edge2idx[(src, node)]
                    iscale_shape, oscale_shape = self.scale_shape[eidx]
                    assert not (oscale_shape != () and iscale_shape != ()), \
                            "Both input and output scale cannot be tensors simultaneously"

                    axis = None
                    qconfig = current_qconfig()
                    use_channel_quantize = qconfig.use_channel_quantize
                    assert oscale_shape == (), "Tuple inputs can't have tensor output scale"
                    if iscale_shape != ():
                        # This is coming from the per channel quantized operator output, like conv2d
                        # output is per-channel quantized.
                        assert use_channel_quantize
                        assert src.op.name in qconfig.per_channel_ops()
                        data_layout = self._node2layouts[src]
                        assert data_layout in ('NCHW', 'NHWC')
                        axis = self._node2channel_axis[src]

                    sim_arg.append(self.create_simulated_quantize(new_arg,
                                                                  in_dtype,
                                                                  out_dtype,
                                                                  axis,
                                                                  iscale_shape,
                                                                  oscale_shape))
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

        in_scale_shape = ()
        for src in new_fn.body.args:
            if isinstance(src, relay.Call) and src.op.name == "nn.simulated_quantize":
                out_scale_node = src.args[2]
                assert isinstance(out_scale_node, relay.Var)
                oshape = tuple(out_scale_node.type_annotation.shape)
                if oshape != ():
                    in_scale_shape = oshape
                    in_arg = fn.body.args[0]
            else:
                raise NotImplementedError()

        in_scale = relay.var('in_scale' + str(self._name_cnt), shape=in_scale_shape)
        out_scale = relay.var('out_scale' + str(self._name_cnt), 'float32')
        clip_min = relay.var('clip_min' + str(self._name_cnt), 'float32')
        clip_max = relay.var('clip_max' + str(self._name_cnt), 'float32')
        self._name_cnt += 1
        self.output_param_nodes.append((in_scale, out_scale, clip_min, clip_max))
        axis = None
        use_channel_quantize = current_qconfig().use_channel_quantize
        if use_channel_quantize and in_scale_shape != ():
            layout = self._node2layouts[in_arg]
            assert layout in ('NCHW', 'NHWC'), layout
            axis = self._node2channel_axis[in_arg]

        # axis = current_qconfig().per_channel_scale_axis
        new_body = _ffi_api.simulated_quantize(new_fn.body,
                                               in_scale,
                                               out_scale,
                                               clip_min,
                                               clip_max,
                                               in_dtype,
                                               DataType('float32'),
                                               True,
                                               "round",
                                               axis)

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
                scale = np.float32(1.0)
                return scale
            assert isinstance(node, relay.Call)
            finfer_scale = node.op.get_attr('FHagoInferScale')
            assert finfer_scale, "No FHagoInferScale for {}".format(node.op.name)
            input_scales = [internal_params[edge2idx[edge]].out_scale for edge in list_in_edges(node)]

            all_scalar = True
            for s in input_scales:
                if not isinstance(s, (int, float)):
                    all_scalar = False

            if all_scalar:
                scale = np.float32(finfer_scale(input_scales))
            else:
                # per channel scales
                assert len(input_scales) == 2
                lhs, rhs = input_scales
                # support conv2d now, so only do scale multiplication
                scale = lhs * rhs
            return scale

        print('\ncalculate parameters')
        def fvisit(node):
            if isinstance(node, relay.Call):
                in_scales = list()
                in_dtypes = list()
                out_dtypes = list()
                out_scales = list()
                for edge in list_in_edges(node):
                    src, _ = edge
                    eidx = edge2idx[edge]
                    in_scale = infer_scale_for_node(src)
                    in_scales.append(in_scale.tolist())
                    in_dtypes.append(prov_dtypes[node2idx[src]])
                    out_dtypes.append(req_dtypes[eidx])
                    if 'float' in str(req_dtypes[eidx]):
                        out_scale = np.float32(1.0)
                    else:
                        bit = edge2bit[(src, node)]
                        integer_range = 2 ** (bit - sign_bit)
                        thold = thresholds[node2idx[src]]
                        out_scale = thold / integer_range
                        if isinstance(out_scale, float):
                            out_scale = np.float32(out_scale)

                    out_scales.append(out_scale.tolist())

                rectified_output_scales = None
                frectify_scale = node.op.get_attr('FHagoRectifyScale')
                if frectify_scale:
                    new_output_scales = frectify_scale(node.args,
                                                       in_scales,
                                                       out_scales)
                    rectified_output_scales = list()
                    for idx, scale in enumerate(new_output_scales):
                        if isinstance(scale, expr.FloatImm):
                            rectified_output_scales.append(scale.value)
                        else:
                            raise NotImplementedError()


                arg_idx = 0
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
                        if rectified_output_scales is not None:
                            out_scale = rectified_output_scales[arg_idx]
                        else:
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
                    arg_idx += 1
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

        if current_qconfig().threshold_estimate_method == 'pot_range':
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
        if new_node.op.name == "nn.simulated_quantize":
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
        in_dtype = attrs.in_dtype
        out_dtype = attrs.out_dtype
        axis = attrs.axis
        if axis is None:
            axis = -1
        else:
            axis = axis.value
        clip_min = to_scalar(clip_min)
        clip_max = to_scalar(clip_max)
        print('  in_scale: {}'.format(in_scale))
        print('  out_scale: {}'.format(out_scale))
        print('  axis: {}'.format(axis))

        if in_dtype == 'float32' and out_dtype == 'float32':
            # do nothing
            return data
        elif out_dtype == 'float32':
            # dequantize
            assert to_scalar(out_scale) == 1.0, out_scale
            return relay.qnn.op.dequantize(data,
                                           input_scale=in_scale,
                                           input_zero_point=relay.const(0, 'int32'),
                                           axis=axis)
        elif in_dtype == 'float32':
            # quantize
            assert to_scalar(in_scale) == 1.0, in_scale
            return relay.qnn.op.quantize(data,
                                         output_scale=out_scale,
                                         output_zero_point=relay.const(0, 'int32'),
                                         axis=axis,
                                         out_dtype=out_dtype)
        else:
            # requantize
            dtype = in_dtype
            if DataType(out_dtype).bits > DataType(in_dtype).bits:
                # pre-casting
                data = relay.cast(data, out_dtype)
                dtype = out_dtype

            return relay.qnn.op.requantize(data,
                                           input_scale=in_scale,
                                           input_zero_point=relay.const(0, 'int32'),
                                           output_scale=out_scale,
                                           output_zero_point=relay.const(0, 'int32'),
                                           out_dtype=out_dtype,
                                           axis=axis)

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
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(self.simulated_graph))
        self.simulated_graph = mod['main']
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
    topology = analyze_topology(graph, hardware)
    return Quantizer(graph, hardware, topology, strategy.bits, strategy.thresholds)
