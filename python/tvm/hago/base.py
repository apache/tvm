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

import tvm._ffi
from tvm.runtime import Object
import math
import numpy as np
from collections import namedtuple, defaultdict, OrderedDict


@tvm._ffi.register_object("hago.QConfig")
class QConfig(Object):
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
        "threshold_estimate_method": "avg_range",
        "global_scale": 8.0,
        "log_file": ".quantize_strategy_search.log",
        # "do_simulation": False,
        # "round_for_shift": True,
        # "debug_enabled_ops": None,
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
        _quantize._ExitQConfigScope()

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
    return tvm.ir.make_node("hago.QConfig", **node_args)


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


def build_node_mapping(sgraph, graph):
    """build a snode -> node mapping"""
    def fvisit_collect_nodes(e):
        if isinstance(e, (relay.Var, relay.Constant, relay.Call)):
            fvisit_collect_nodes.nodes.append(e)
    fvisit_collect_nodes.nodes = []

    def fvisit_collect_snodes(e):
        if isinstance(e, relay.Call) and e.op.name == 'hago.simulated_quantize':
            node = e.args[0]
            if node not in fvisit_collect_snodes.set:
                # avoid multi-refer
                fvisit_collect_snodes.set.add(node)
                fvisit_collect_snodes.nodes.append(node)
    fvisit_collect_snodes.nodes = []
    fvisit_collect_snodes.set = set([])

    relay.analysis.post_order_visit(sgraph, fvisit_collect_snodes)
    snodes = fvisit_collect_snodes.nodes
    relay.analysis.post_order_visit(graph, fvisit_collect_nodes)
    nodes = fvisit_collect_nodes.nodes

    print('num of snodes:')
    print(len(snodes))
    print('num of nodes:')
    print(len(nodes))
    assert(len(snodes) == len(nodes))
    mapping = OrderedDict()
    for snode, node in zip(snodes, nodes):
        mapping[snode] = node
    return mapping


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


def min_with_none(a, b):
    # handle None
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)

def max_with_none(a, b):
    # handle None
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def node_str(node, node2idx=None):
    def _str(node):
        if isinstance(node, (relay.Var)):
            return node.name_hint
        elif isinstance(node, relay.Constant):
            return 'constant'
        elif isinstance(node, relay.Call):
            return node.op.name
        else:
            raise ValueError("{}, {}".format(type(node), node))
        return None
    if node2idx:
        return "{}[%{}]".format(_str(node), node2idx[node])
    return _str(node) 

def edge_str(edge, node2idx=None):
    return "{} -> {}".format(node_str(edge[0], node2idx), node_str(edge[1], node2idx))

def list_in_nodes(node):
    """Handle Tuple here."""
    assert isinstance(node, relay.Call)
    for arg in node.args:
        if isinstance(arg, (relay.Var, relay.Constant, relay.Call)):
            yield arg
        elif isinstance(arg, relay.expr.Tuple):
            for src in arg:
                yield src
        else:
            raise ValueError

def list_in_edges(node):
    """Handle Tuple here."""
    assert isinstance(node, relay.Call)
    for src in list_in_nodes(node):
        yield (src, node)


def evaluate(func, dataset, ctx=tvm.cpu(), target='llvm'):
    # [[numpy array]]: [num_outputs x num_batch x batched_output]
    with relay.transform.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(func, target=target)
    runtime = tvm.contrib.graph_runtime.create(graph, lib, ctx)
    runtime.set_input(**params)
    num_outputs = runtime.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]

    input_keys = [str(param.name_hint) for param in func.params]
    for batch_id, batch in enumerate(dataset):
        for key in input_keys:
            runtime.set_input(key, batch[key])
        runtime.run()
        for i in range(num_outputs):
            output = runtime.get_output(i).asnumpy()
            if len(output.shape) == 0:
                output = np.array([output])
            outputs[i].append(output)
    return outputs

def calculate_accuracy(dataset, outputs):
    num_correct = 0
    num_samples = 0
    for batch, output in zip(dataset.batches, outputs):
        predict = np.argmax(output, axis=1)
        label = batch['label'].asnumpy()
        num_correct += np.sum(predict == label)
        num_samples += output.shape[0]
    acc = num_correct / num_samples
    return acc


def exponent_based_two(val):
    exponent = math.log2(val)
    cond = (exponent == round(exponent))
    # cond = math.isclose(exponent, round(exponent), rel_tol=5e-5)
    if cond: 
        return cond, round(exponent)
    return cond, exponent


def to_scalar(constant):
    assert isinstance(constant, relay.Constant)
    scalar = constant.data.asnumpy()
    assert scalar.size == 1
    return scalar.item()
