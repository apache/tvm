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
import numpy as np

from . import _quantize
from .. import expr as _expr
from .. import module as _module
from .. import analysis as _analysis
from .. import transform as _transform
from .. import op as _op
from ... import make as _make
from ..base import NodeBase, register_relay_node


class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""
    IDENTITY = 0
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3


def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
        QAnnotateKind.IDENTITY: "identity"
    }
    assert kind in str_map
    return str_map[kind]


def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return _expr.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


@register_relay_node("relay.quantize.QConfig")
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
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "global_scale": 8.0,
        "skip_conv_layers": [0],
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
    return _make.node("relay.quantize.QConfig", **node_args)


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


def partition():
    """Partition graph into small low-precision sections by `cast_hint` and
    `stop_fusion`.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for VTA rewrite.
    """
    return _quantize.QuantizePartition()


def annotate():
    """Given a float32 graph, this pass will rewrite the graph and return
    a graph which simulates the error brought by the current quantization
    scheme.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for quantization annotation.
    """
    return _quantize.QuantizeAnnotate()


def collect_stats(graph):
    """Given an annotated graph, create a profile graph to collect profile data from the
    calibration dataset. This pass collects simulated_quantize op input into a tuple.
    Simulated_quantize ops are rewritten to identity mode. The tuple is the output of the profile
    graph.

    Parameters
    ----------
    graph: Function
        The simulation graph after annotation.

    Returns
    -------
    ret: Function
        The profile graph which outputs a tuple of profile data.
    """
    return _quantize.CollectStats(graph)


def calibrate(graph, mod=None, ctx=None, weight_scales='power2', scales=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max for every `simulated_quantize`
    operator.

    Parameters
    ---------
    graph: Function
        The simulation graph after annotation.

    mod: tvm.relay.Module
        The module where calibration happens on.

    ctx: tvm.relay.PassContext
        The pass context used for calibration.

    weight_scales: 'power2' or 'max'.
        The way to calculate scales for weights (annotated with QAnnotateKind.WEIGHT).
        power2: Find the maximum of the absolute value of the tensor, and then round up to power
        of two.
        max: Find the maximum of the absolute value of the tensor.

    scales: List[float]
        Pre-calculated scales for input and activations. Length and the order of elements of the
        scales list should match the output tuple of the profile graph created by collect_stats.

    Returns
    -------
    ret: Function
        The graph after calibration
    """
    def power2_scale(arr):
        """calculate weight scale with nearest mode-2 scale"""
        val = np.amax(np.abs(arr.asnumpy()))
        return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

    def max_scale(arr):
        """calculate weight scale with maximum absolute value"""
        val = np.amax(np.abs(arr.asnumpy()))
        return val

    scale_idx = 0

    cfg = current_qconfig()
    const_params = {}
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")

    def visit_func(expr):
        """Internal visit function"""
        nonlocal scale_idx
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            nbit = cfg.get_nbit_by_kind(kind)

            valid_bit = nbit - attrs.sign
            if kind in [QAnnotateKind.WEIGHT]:
                if all([isinstance(arg, _expr.Constant)
                        for arg in [ndom_scale, nclip_min, nclip_max]]):
                    return
                var = expr.args[0]
                assert isinstance(var, _expr.Constant)
                if weight_scales == 'max':
                    scale = max_scale(var.data)
                elif weight_scales == 'power2':
                    scale = power2_scale(var.data)
                else:
                    raise ValueError('{} not supported'.format(weight_scales))
            elif scales is not None:
                scale = scales[scale_idx]
                scale_idx += 1
            else:
                scale = cfg.global_scale

            def _make_const(val):
                return _expr.const(val, 'float32')

            valid_range = 2**valid_bit
            const_params[ndom_scale] = _make_const(scale / valid_range)
            const_params[nclip_min] = _make_const(- (valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    _analysis.post_order_visit(graph, visit_func)
    ret = _expr.bind(graph, const_params)
    return ret


def realize():
    """The realize pass will transform the simulated quantized graph, which
    actually computes with float32, to a real low-bit integer graph. It will
    replace the `simulated_quantize` with several fine-grained operators like
    add, multiply, and shift as much as possible for better performance.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for quantization realization.
    """
    return _quantize.QuantizeRealize()


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
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def prerequisite_optimize(graph, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = _transform.Sequential([_transform.SimplifyInference(),
                                      _transform.FoldConstant(),
                                      _transform.FoldScaleAxis(),
                                      _transform.CanonicalizeOps(),
                                      _transform.FoldConstant()])

    if params:
        graph = _bind_params(graph, params)

    mod = _module.Module.from_expr(graph)
    with _transform.PassContext(opt_level=3):
        mod = optimize(mod)
    return mod["main"]


def quantize(graph, params=None, dataset=None):
    """ The quantization procedure. Before running the three main
    procedure of quantization, "annotate", "calibrate" and "realize"
    , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
    first for optimizing.

    Parameters
    ---------
    graph: Function
        The original graph.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    graph = prerequisite_optimize(graph, params)

    mod = _module.Module.from_expr(graph)
    calibrate_pass = _transform.function_pass(calibrate, opt_level=1,
                                              name="QuantizeCalibrate")
    quant_passes = [partition(),
                    annotate(),
                    calibrate_pass]
    if not current_qconfig().do_simulation:
        quant_passes.append(realize())
    quant_passes.append(_transform.FoldConstant())
    quantize_seq = _transform.Sequential(quant_passes)
    with _transform.PassContext(opt_level=3,
                                required_pass=["QuantizeAnnotate",
                                               "QuantizeCalibrate",
                                               "QuantizeRealize"]):
        with quantize_context():
            mod = quantize_seq(mod)

    return mod["main"]
