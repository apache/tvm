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
from tvm._ffi.runtime_ctypes import TVMType
from collections import defaultdict

DType = TVMType

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
        "threshold_estimate_strategy": "power2_range",
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


def build_node_index(graph):
    node2idx = {}
    def fvisit_build_index(e):
        if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
            node2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    relay.analysis.post_order_visit(graph, fvisit_build_index)
    num_nodes = fvisit_build_index.idx_cnt
    return node2idx, num_nodes

def build_edge_index(graph):
    edge2idx = {} 
    def fvisit_build_index(e):
        if isinstance(e, relay.Call):
            for arg in e.args:
                edge2idx[(arg, e)] = fvisit_build_index.idx_cnt 
                fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    relay.analysis.post_order_visit(graph, fvisit_build_index)
    num_edges = fvisit_build_index.idx_cnt
    return edge2idx, num_edges

def build_node2edges(graph):
    node2edges = defaultdict(list)
    def fvisit_build_index(node):
        if isinstance(node, relay.Call):
            for src in node.args:
                node2edges[src].append((src, node)) 
    relay.analysis.post_order_visit(graph, fvisit_build_index)
    return node2edges


def bind_params(func, params):
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

def node_str(e):
    if isinstance(e, (relay.Var)):
        return e.name_hint
    if isinstance(e, relay.Constant):
        return 'constant'
    if isinstance(e, relay.Call):
        return e.op.name
    return None

def edge_str(edge):
    return "{} -> {}".format(node_str(edge[0]), node_str(edge[1]))

def print_bits_info(graph, bits):
    edge2idx, num_edge = build_edge_index(graph)

    def fvisit_print(e):
        if isinstance(e, relay.Call):
            for src in e.args:
                idx = edge2idx[(src, e)]
                print('{} <- {}: {}'.format(node_str(e), node_str(src), bits[idx]))
    relay.analysis.post_order_visit(graph, fvisit_print)

def print_scale_info(graph, bits, thresholds):
    node2idx, num_edge = build_node_index(graph)
    edge2idx, num_edge = build_edge_index(graph)
    def fvisit_print(node):
        if isinstance(node, relay.Call):
            for src in node.args:
                thold = thresholds[node2idx[src]]
                eidx = edge2idx[(src, node)]
                bit = bits[eidx]
                scale = thold / (2 ** (bit - 1) - 1)
                print('{} <- {}: {}'.format(node_str(node), node_str(src), scale))
    relay.analysis.post_order_visit(graph, fvisit_print)
