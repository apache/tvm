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
from ..relay.base import NodeBase, register_relay_node

import tvm
import sys
import numpy as np
import functools
from collections import namedtuple 

# TODO(ziheng): refactor the infra to modulized pass

def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return relay.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


@register_relay_node("hago.QConfig")
class QConfig(NodeBase):
    """Configure the quantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use qconfig instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "skip_conv_layers": [0],
        "search_strategy": "simulated_annealing",
        "threshold_estimate_strategy": "max_range",
        "global_scale": 8.0,
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(QConfig, self).__init__(handle)
        self.handle = handle

    def guard(self, ref_call):
        """Return true if op is enabled, otherwise return false"""
        op_name = ref_call.op.name
        if self.debug_enabled_ops is not None:
            name_list = [x.value for x in self.debug_enabled_ops]
            if op_name not in name_list:
                return False
        return True

    def get_nbit_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'nbit_' + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, 'dtype_' + name)

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope(self)

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError(
                "'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(QConfig, self).__setattr__(name, value)


def current_qconfig():
    """Get the current quantization configuration."""
    return _quantize._GetCurrentQConfig()


def qconfig(**kwargs):
    """Configure the quantization behavior by setting config variables.

    Parameters
    ---------
    nbit_dict: dict of QAnnotateKind -> int
        Number of bit for every kind of annotate field.

    global_scale: float
        The global scale for calibration.

    skip_conv_layers: list
        Specifying which layers to be skipped. Provide a list of indices
        that indicate which conv2d layers to leave untouched. Start from 0.

    do_simulation: boolean
        Whether to do simulation with float operation only.

    round_for_shift: boolean
        Whether to add bias for rounding during shift.

    debug_enabled_ops: None or list of str
        Partially quantize specified operators for debugging. The default value
        is None, which means will try to call all operartors' annotate rewrite
        function.

    Returns
    -------
    config: QConfig
        The quantization configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k]
                 for k, v in QConfig._node_defaults.items()}
    return tvm.make.node("hago.QConfig", **node_args)


class QuantizeContext(object):
    """An internal used global context object for annotation,
    for putting some state variables like `conv2d_counter`."""
    Current = None

    def __init__(self):
        self.qnode_map = dict()
        self._conv2d_counter = 0
        self._stop_quantize = False

    def check_to_skip(self, ref_call):
        """Check the index of conv2d layer to decide whether to
        skip the current operator."""
        if self._stop_quantize:
            return True

        if current_qconfig().skip_conv_layers is not None:
            # check skip conv layers
            skipped_indices = [int(x) for x in current_qconfig().skip_conv_layers]
            if self._conv2d_counter in skipped_indices:
                if ref_call.op.name == 'nn.conv2d':
                    self._conv2d_counter += 1
                return True
            if ref_call.op.name == 'nn.conv2d':
                self._conv2d_counter += 1

        return False

    def stop_quantize(self):
        self._stop_quantize = True

    def reset(self):
        self._conv2d_counter = 0
        self._stop_quantize = False

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, ptype, value, traceback):
        pass


def quantize_context():
    """Get the global singleton scope"""
    if QuantizeContext.Current is None:
        QuantizeContext.Current = QuantizeContext()
    return QuantizeContext.Current


SimulatedQuantizeParams = namedtuple("SimulatedQuantizeParams", ['scale',
                                                                 'clip_min',
                                                                 'clip_max',
                                                                 'overflow_min',
                                                                 'overflow_max'])


def simulate(graph, op_params):
    class Simulator(tvm.relay.ExprMutator):
        def __init__(self):
            super().__init__()

        def create_simulated_graph(self, graph, op_params):
            self._op_params = op_params
            self.relay2idx = {}  # expr(var/call) to idx
            def fvisit_build_index(e):
                if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
                    self.relay2idx[e] = fvisit_build_index.idx_cnt 
                    fvisit_build_index.idx_cnt += 1
            fvisit_build_index.idx_cnt = 0
            relay.analysis.post_order_visit(graph, fvisit_build_index)
            return self.visit(graph)

        def visit(self, e):
            new_e = super().visit(e)
            if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
                param = self._op_params[self.relay2idx[e]]
                ret = _quantize.simulated_quantize(new_e,
                                                   tvm.relay.const(param.scale),
                                                   tvm.relay.const(param.clip_min),
                                                   tvm.relay.const(param.clip_max),
                                                   tvm.relay.const(param.overflow_min),
                                                   tvm.relay.const(param.overflow_max),
                                                   True,
                                                   "round")
                return ret
            return new_e

        def visit_function(self, fn):
            # skip params
            new_body = self.visit(fn.body)
            return relay.Function(
                fn.params,
                new_body,
                fn.ret_type,
                fn.type_params,
                fn.attrs)

    # print('before simulating')
    # print(graph)
    # print('creare simulated graph')
    simulated_graph = Simulator().create_simulated_graph(graph, op_params)
    # print(simulated_graph)
    return simulated_graph


def calculate_quantize_op_params(graph, bits, thresholds, hw_desc):
    # map: tensor -> (num_bit, threshold)
    # set scale, clip_min, clip_max for every tensor
    # integer_range = 2 ^ (num_bit - sign) 
    # scale = threshold / integer_range
    # clip_min = - (integer_range - 1)
    # clip_max =   (integer_range - 1)
    #
    # prepare:
    # overflow_num_bit = 16bit (hardware_constrait)
    # input_scale
    #
    # overflow_lower_bound_integer = - (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_lower_bound_real    = overflow_lower_bound_quant * input_scale
    # overflow_upper_bound_integer = (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_upper_bound_real    = overflow_upper_bound_quant * input_scale
    assert len(bits) == len(thresholds)
    # print('num of tensors: {}'.format(len(bits)))
    sign = 1

    expr2idx = {}  # expr(var/call) to idx
    def fvisit_build_index(e):
        if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
            expr2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    relay.analysis.post_order_visit(graph, fvisit_build_index)

    op_params = []
    def fvisit(e):
        if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
            # if isinstance(e, relay.Var):
            #     print(e)
            # if isinstance(e, relay.Constant):
            #     print('constant')
            # if isinstance(e, relay.Call):
            #     print(e.op.name)

            bit = bits[fvisit.idx_cnt]
            threshold = thresholds[fvisit.idx_cnt]
            integer_range = pow(2, bit - sign)
            scale = threshold / integer_range
            clip_min = - float(integer_range - 1)
            clip_max =   float(integer_range - 1)

            overflow_min = np.array([- sys.float_info.max], dtype=np.float32)
            overflow_max = np.array([sys.float_info.max], dtype=np.float32)

            # consider hardware constraint to detect overflow
            if isinstance(e, relay.Call) and e.op.name in hw_desc.ops:
                # scale of inputs
                input_scales = [op_params[expr2idx[arg]].scale for arg in e.args]
                # print('input scales')
                # print(input_scales)
                # calculate op's output scale with input scales
                # TODO(ziheng): different rules for different op
                if e.op.name == 'nn.conv2d':
                    input_scale = functools.reduce(lambda x, y: x*y, input_scales, 1.0)
                elif e.op.name == 'add':
                    input_scale = max(input_scales)
                else:
                    raise ValueError('not support {0} yet.'.format(e.op.name))

                # print(hw_desc[e.op.name])
                overflow_num_bit = max(instr[1][0] for instr in hw_desc[e.op.name])
                overflow_min_integer = - (2 ^ (overflow_num_bit - sign) - 1)
                overflow_max_integer = (2 ^ (overflow_num_bit - sign) - 1)
                overflow_min = overflow_min_integer * input_scale
                overflow_max = overflow_max_integer * input_scale
                # TODO(ziheng) support scalar for extern function
                overflow_min = np.array([overflow_min], dtype=np.float32)
                overflow_max = np.array([overflow_max], dtype=np.float32)


            param = SimulatedQuantizeParams(scale, clip_min, clip_max,
                                            overflow_min,
                                            overflow_max)
            # print('op_param')
            # print(param)
            op_params.append(param)
            fvisit.idx_cnt += 1
    fvisit.idx_cnt = 0
    relay.analysis.post_order_visit(graph, fvisit)
    return op_params


def quantize(graph, bits, thresholds, hw_desc):
    cfg = current_qconfig()
    op_params = calculate_quantize_op_params(graph, bits, thresholds, hw_desc)
    simulated_graph = simulate(graph, op_params)
    sim_mod = tvm.relay.Module.fromrelay(final_simulated_graph)
    if cfg.do_simulation:
        return sim_mod
    else:
        # realize to low-precesion integer
        pass
# 
# 
# 
# # NOTE:
# # behavior of cast_hint
# # partition part, will insert cast_hint to denote, which will be
# # inserted simulated quantize during annotate
# 
# # in some condition we need to defer the cast operartion
# # will be transformed to real cast in realize
# # but work as identity before realize
# 
# 
# # behavior of quantized_add
# # add wiil be transformed to quantized_add during annotate,
# # oscale will be tuned by calibrate
# # if simulated, only do addition, which is used before realize
# # during realize, lhs and rhs's scale will be unified as oscale
# # then do add
# def partition():
#     """Partition graph into small low-precision sections by `cast_hint` and
#     `stop_fusion`.
# 
#     Returns
#     -------
#     ret: tvm.relay.Pass
#         The registered pass for VTA rewrite.
#     """
#     return _quantize.QuantizePartition()
# 
# 
# def realize():
#     """The realize pass will transform the simulated quantized graph, which
#     actually computes with float32, to a real low-bit integer graph. It will
#     replace the `simulated_quantize` with several fine-grained operators like
#     add, multiply, and shift as much as possible for better performance.
# 
#     Returns
#     -------
#     ret: tvm.relay.Pass
#         The registered pass for quantization realization.
#     """
#     return _quantize.QuantizeRealize()
# 
# 
def _bind_params(func, params):
    """Bind the params to the expression.
    """
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = relay.const(v)
    return relay.bind(func, bind_dict)


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
        mod['main'] = _bind_params(mod['main'], params)

    with relay.transform.PassContext(opt_level=3):
        mod = optimize(mod)
    return mod


# def quantize(mod, params=None, dataset=None):
#     """ The quantization procedure. Before running the three main
#     procedure of quantization, "annotate", "calibrate" and "realize"
#     , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
#     first for optimizing.
# 
#     Parameters
#     ---------
#     graph: Function
#         The original graph.
# 
#     params : dict of str to NDArray
#         Input parameters to the graph that do not change
#         during inference time. Used for constant folding.
# 
#     dataset: list of dict of Var -> NDArray
#         The calibration dataset.
# 
#     Returns
#     -------
#     ret: Function
#         The graph after quantization
#     """
#     mod = prerequisite_optimize(mod, params)
#     calibrate_pass = _transform.module_pass(_calibrate.calibrate(dataset),
#                                             opt_level=1,
#                                             name="QuantizeCalibrate")
#     quant_passes = [
#                     annotate(),
#                     calibrate_pass]
#     if not current_qconfig().do_simulation:
#         quant_passes.append(realize())
#     quant_passes.append(_transform.FoldConstant())
#     quantize_seq = _transform.Sequential(quant_passes)
#     with _transform.PassContext(opt_level=3):
#         with quantize_context():
#             mod = quantize_seq(mod)
# 
#     return mod
