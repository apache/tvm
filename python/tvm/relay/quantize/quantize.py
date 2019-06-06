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
import time
import numpy as np
from scipy import stats


from . import _quantize
from .. import expr as _expr
from .. import module as _module
from .. import analysis as _analysis
from .. import transform as _transform
from .. import op as _op
from ... import make as _make
from ..._ffi.function import register_func
from ..base import NodeBase, register_relay_node



class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3


def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
    }
    assert kind in str_map
    return str_map[kind]


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
        "round_for_shift": True,
        "store_lowbit_output": True,
        "debug_enabled_ops": None,
#TODO(eqy)
#=======
#        "skip_k_conv": 1,
#        "skip_conv_layers": None,
        "passthrough_bound": 1e9,
#        "round_for_shift": True,
#        "store_lowbit_output": True,
        "debug_enabled_ops": None,
#        "use_stop_fusion": True,
        "granularity": "layer",
#>>>>>>> check in
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

# TODO(tmoreau89, ZihengJiang) the skip parameters are
# hacky - we should explore a more future-proof way to
# skip operators based on pattern matching
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
        that indicate which conv2d layers to leave untouched.

    round_for_shift: boolean
        Whether to add bias for rounding during shift.

    store_lowbit_output: boolean
        Whether to store low-bit integer back as output before dequantizing.
        Some accelerators need this, e.g. VTA.

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


class AnnotateContext(object):
    """A global singleton annotate scope"""
    Current = None

    def __init__(self):
        self.qnode_map = dict()
        self._conv2d_counter = 0

    def __enter__(self):
        self._conv2d_counter = 0
        return self

    def conv2d_counter(self):
        """Get the counter for conv2d."""
        return self._conv2d_counter

    def count_conv2d(self):
        """Increase the value of the conv2d counter by one."""
        self._conv2d_counter += 1

    def __exit__(self, ptype, value, traceback):
        pass

def annotate_context():
    """Get the global singleton scope"""
    if AnnotateContext.Current is None:
        AnnotateContext.Current = AnnotateContext()
    return AnnotateContext.Current

#TODO(eqy)
#def calibrate(graph, mod=None, ctx=None):
#=======
SCALE_COUNTER = 0


def _get_scale_counter():
    """Get the global counter for scale setting."""
    return SCALE_COUNTER


def _set_scale_counter(n):
    """Set the value of the global scale setting counter."""
    global SCALE_COUNTER
    SCALE_COUNTER = n


LAYOUT_MAP = None


def _set_layout_map(layout_map):
    global LAYOUT_MAP
    LAYOUT_MAP = layout_map


def _layout_walk(expr):
    conv2d_op = _op.get("nn.conv2d")
    if isinstance(expr, _expr.Call):
        if expr.op == conv2d_op:
            return expr.attrs.data_layout if expr.attrs.out_layout == "" else expr.attrs.out_layout
        else:
            for arg in expr.args:
                if arg in LAYOUT_MAP:
                    return LAYOUT_MAP[arg]
                ret = _layout_walk(arg)
                if ret is not None:
                    return ret
            return None
    elif isinstance(expr, _expr.Tuple):
        for arg in expr.fields:
            ret = _layout_walk(arg)
            if ret is not None:
                return ret
        return None
    elif isinstance(expr, _expr.TupleGetItem):
        return _layout_walk(expr.tuple_value)
    raise Exception


@register_func("relay.quantize._get_layout")
def _get_layout(expr):
    try:
        return LAYOUT_MAP[expr]
    except KeyError:
        ret = _layout_walk(expr)
        if ret is not None:
            return ret
        raise KeyError


def annotate(graph, layout_map):
    """Given a float32 graph, annotate will rewrite the graph
    and return back a graph which simulates the error brought by
    current quantization scheme.

    Parameters
    ---------
    graph: Function
        The original graph

    Returns
    -------
    ret: Function
        The graph after annotation
    """
    _set_conv_counter(0)  # reset counter
    _set_layout_map(layout_map)
    return _quantize.annotate(graph)


def tag_layout(graph):
    conv2d_op = _op.get("nn.conv2d")
    dense_op = _op.get("nn.dense")
    _op_layout_map = dict()
    # layouts to tag later
    deferred = set()

    def extract_call_layout(args):
        cur_layout = None
        for arg in args:
            if isinstance(arg, _expr.Call):
                assert arg in _op_layout_map
                if cur_layout is None:
                    cur_layout = _op_layout_map[arg]
                else:
                    assert cur_layout == _op_layout_map[arg]
            elif isinstance(arg, _expr.Tuple):
                return extract_call_layout(arg.fields)
            elif isinstance(arg, _expr.TupleGetItem):
                return extract_call_layout(arg.tuple_value.args)
        return cur_layout

    def visit_func(expr):
        """Internal visit function"""
        if isinstance(expr, _expr.Call):
            cur_layout = None
            if expr.op == conv2d_op:
                if expr.attrs.out_layout == "":
                    _op_layout_map[expr] = expr.attrs.data_layout
                else:
                    _op_layout_map[expr] = expr.attrs.out_layout
                cur_layout = _op_layout_map[expr]
            else:
                cur_layout = extract_call_layout(expr.args)
                if cur_layout is None:
                    deferred.add(expr)
                else:
                    _op_layout_map[expr] = cur_layout
            if cur_layout is not None:
                for arg in expr.args:
                    if arg in deferred:
                        _op_layout_map[arg] = cur_layout
                        deferred.remove(arg)

    _ir_pass.post_order_visit(graph, visit_func)
    if len(deferred) > 0:
        raise ValueError

    return _op_layout_map


_WEIGHT_SCALE_OPTS = [2**i for i in range(-10, 8)]


def slice_idx(begin, end):
    import tvm.expr
    assert len(begin) == len(end)
    for i in range(0, len(begin)):
        if not isinstance(end[i], tvm.expr.IntImm) or end[i].value - begin[i].value == 0:
            continue
        return begin[i].value, end[i].value
    raise ValueError


# ALIGN weight * data scales for convolution
def match_scales(graph, const_params, mode='max'):
    conv2d_op = _op.get("nn.conv2d")
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")

    def visit_func(expr):
        if isinstance(expr, _expr.Call):
            if expr.op == conv2d_op:
                quant_weight = expr.args[1]
                weight_data, weight_scale_var, _, _  = quant_weight.args
                weight_scale = const_params[weight_scale_var].data

                if expr.args[0].op != quantize_op and\
                    expr.args[1].op == quantize_op:
                    unified_scale = np.empty(weight_scale.asnumpy().shape, dtype='float32')
                    # only weight shift possible
                    parent = expr.args[0].args[0]
                    assert parent.op == quantize_op
                    _, data_scale_var, _, _, = parent.args
                    data_scale = const_params[data_scale_var].data
                    if data_scale.shape != weight_scale.shape:
                        assert expr.args[0].op == _op.get("strided_slice")
                        begin, end = slice_idx(expr.args[0].attrs.begin, expr.args[0].attrs.end)
                        product = data_scale.asnumpy()[begin:end] * weight_scale.asnumpy()
                    else:
                        product = data_scale.asnumpy() * weight_scale.asnumpy()
                    if mode == 'max':
                        unified_scale = max(product)
                    else:
                        unified_scale = min(product)
                    # (d * s_d) * (w * s_w) = (o * s_d * s_w)
                    # (d * s_d) * (w * s_w') = (o * s_u)
                    # s_w' = s_u/s_d
                    gaps = unified_scale/product
                    weight_scale_transform = np.empty(gaps.shape, dtype='float32')
                    for i in range(0, gaps.shape[0]):
                        shift_width = np.log2(gaps[i])
                        if shift_width == 0:
                            weight_scale_transform[i] = 1.0
                        else:
                            weight_scale_transform[i] = 2**shift_width
                    new_weight_scale = weight_scale.asnumpy()*weight_scale_transform
                    const_params[weight_scale_var] = _expr.const(new_weight_scale)
                    return
                elif expr.args[0].op == quantize_op and\
                    expr.args[1].op != quantize_op:
                    raise ValueError
                elif expr.args[0].op != quantize_op and\
                     expr.args[1].op != quantize_op:
                    raise ValueError

                quant_data = expr.args[0]
                _, data_scale_var, _, _ = quant_data.args
                data_scale = const_params[data_scale_var].data
                assert len(data_scale.shape) == 1
                assert len(weight_scale.shape) == 1
                if data_scale.shape[0] == 1:
                    assert weight_scale.shape[0] == 1
                else:
                    assert weight_scale.shape[0] == data_scale.shape[0] or\
                    weight_scale.shape[0] == 1 # depthwise, no need to unify scales
                    if weight_scale.shape[0] != 1:
                        product = data_scale.asnumpy() * weight_scale.asnumpy()
                        if mode == 'max':
                            unified_scale = max(product)
                        else:
                            unified_scale = np.median(product)
                        # (d * s_d) * (w * s_w) = (o * s_d * s_w)
                        # (d * s_d) * (w * s_w') = (o * s_u)
                        # s_w' = s_u/s_d
                        gaps = unified_scale/product
                        data_scale_transform = np.empty(gaps.shape, dtype='float32')
                        weight_scale_transform = np.empty(gaps.shape, dtype='float32')
                        for i in range(0, gaps.shape[0]):
                            shift_width = np.log2(gaps[i])
                            if shift_width == 0:
                                weight_scale_transform[i] = 1.0
                            else:
                                # magic heuristic, change data scales more
                                # aggressively than weight scales for
                                # compensation
                                weight_scale_transform[i] = 2**(shift_width//2)
                        data_scale_transform = gaps/weight_scale_transform
                        new_data_scale = data_scale.asnumpy()*data_scale_transform
                        new_weight_scale = weight_scale.asnumpy()*weight_scale_transform
                        const_params[weight_scale_var] = _expr.const(new_weight_scale)
                        const_params[data_scale_var] = _expr.const(new_data_scale)

    _ir_pass.post_order_visit(graph, visit_func)
    return const_params


def _simulate_quantize(array, scale):
    # simulate rounding error

    valid_bit = 7
    valid_range = 2**valid_bit
    clip_min = - (valid_range - 1)
    clip_max = valid_range - 1

    scale = scale / valid_range
    assert scale > 0
    scaled_data = array/scale
    clipped_data = np.clip(scaled_data, clip_min, clip_max)

    round_data = np.round(clipped_data)
    return round_data*scale

def _mse_chooser(act, granularity, layout, op_hint=None):
    t1 = time.time()
    assert len(act.shape) <= 4, "Unsupported layout"
    # TODO block layouts
    assert layout.upper() == layout, "Blocked layouts not supported"

    if granularity == 'layer' or (op_hint is None and len(act.shape) < len(layout)):
        mses = list()
        for config_opt in _WEIGHT_SCALE_OPTS:
            q = _simulate_quantize(act, config_opt)
            mse = ((act - q)**2).mean()
            mses.append(mse)
        t2 = time.time()
        scale = _WEIGHT_SCALE_OPTS[np.argmin(mses)]
        return np.array([scale], dtype='float32')
    else:
        if len(act.shape) >= len(layout):
            if 'O' in layout and 'I' in layout:
                channel_dim = layout.index('I')
            else:
                channel_dim = layout.index('C')
            channels = act.shape[channel_dim]
        elif op_hint is not None and 'dense' in op_hint:
            channel_dim = 0
            channels = 1 
        else:
            assert 'broadcastable' in op_hint, "trying to broadcast non-broadcastable op"
            if len(act.shape) == len(layout) - 1:
                for i in range(0, len(act.shape)):
                    if act.shape[i] != 1:
                        channel_dim = i
                        channels = act.shape[i]
            else:
                channel_dim = 0
                channels = 1

        scales = np.array([0.0]*channels, dtype='float32')
        for i in range(0, channels):
            mses = list()
            for config_opt in _WEIGHT_SCALE_OPTS:
                sliced_act = np.take(act, i, channel_dim)
                q = _simulate_quantize(sliced_act, config_opt)
                mse = ((sliced_act - q)**2).mean()
                mses.append(mse)
                if mse == 0.0:
                    # use mode as fallback
                    scales[i] = -1
                    break
            if scales[i] == 0.0:
                scales[i] = _WEIGHT_SCALE_OPTS[np.argmin(mses)]
        mode = stats.mode(scales[scales > 0.0])[0]
        scales[scales < 0] = mode
        t2 = time.time()
        return scales


def calibrate(graph, dataset=None, profile_mode=False, scales=None):
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

    Returns
    -------
    ret: Function
        The graph after calibration
    """
    if profile_mode:
        assert scales is None, "scales should not be passed in with profile_mode"
    else:
        assert scales is not None, "did not receive scales"

    def power2_scale(arr, granularity, layout, op_hint):
        """calculate weight scale with nearest mode-2 scale"""
        val = np.amax(np.abs(arr.asnumpy()))

        # TODO blocked layout
        if granularity == 'channel' or granularity == 'layer':
            scale = _mse_chooser(arr.asnumpy(), granularity, layout, op_hint)
            return scale
            if len(arr.shape) >= 4:
                if 'I' in layout:
                    channel_dim = layout.index('I')
                else:
                    channel_dim = layout.index('C')
                channels = arr.shape[channel_dim]
                scales = list()
                for i in range(0, channels):
                    val = np.amax(np.abs(np.take(arr.asnumpy(), i, channel_dim)))
                    scale = 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0
                    scales.append(scale)
                return np.array(scales, dtype='float32')
            else:
                scale = 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0
                return np.array([scale], dtype='float32')
        else:
            return 2**np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0

    cfg = current_qconfig()
    const_params = {}
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    profile_data = []
    scale_idx = 0

    def visit_func(expr):
        """Internal visit function"""
        nonlocal scale_idx
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            granularity = attrs.granularity
            layout = attrs.layout

            nbit = cfg.get_nbit_by_kind(kind)

            valid_bit = nbit - attrs.sign
            valid_range = 2**valid_bit

            def _make_const(val):
                return _expr.const(val, 'float32')

            if kind == QAnnotateKind.WEIGHT:
                var = expr.args[0]
                assert isinstance(var, _expr.Constant)
                data = var.data
                if False and 'add' in attrs.op_hint:
                    data_np = data.asnumpy()
                    zero_ind = data_np < 2**-4
                    data_np[zero_ind] = np.mean(data_np)
                    data = _make_const(data).data
                scale = power2_scale(data, granularity, layout, attrs.op_hint)
                const = _make_const(scale / valid_range)
                assert len(const.data.shape) == 1
                const_params[ndom_scale] = const
            else:
                if profile_mode:
                    profile_data.append((ndom_scale.name_hint, expr.args[0],
                                         granularity, layout))
                else:
                    const = _make_const(scales[scale_idx]/valid_range)
                    const_params[ndom_scale] = const
                    assert len(const.data.shape) == 1
                    scale_idx += 1
            const_params[nclip_min] = _make_const(- (valid_range - 1))
            const_params[nclip_max] = _make_const((valid_range - 1))

    _analysis.post_order_visit(graph, visit_func)
    if profile_mode:
        for i, val  in enumerate(profile_data):
            profile_data[i] = (val[0], _expr.bind(val[1], const_params), val[2], val[3])
    else:
        const_params = match_scales(graph, const_params)
#TODO(eqy):
#    return _expr.bind(graph, const_params)
#=======
    #_ir_pass.post_order_visit(graph, visit_func)
    return _expr.bind(graph, const_params), profile_data
#>>>>>>> check in


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


def rewrite_for_vta():
    """Performs rewriting for VTA target.

    Returns
    -------
    ret: tvm.relay.Pass
        The registered pass for VTA rewrite.
    """
    return _quantize.QuantizeRewriteForVTA()


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
    if params:
        graph = _bind_params(graph, params)

    mod = _module.Module.from_expr(graph)
    # Perform "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    # "CanonicalizeOps" optimization before quantization.
    optimize = _transform.Sequential([_transform.SimplifyInference(),
                                      _transform.FoldConstant(),
                                      _transform.FoldScaleAxis(),
                                      _transform.CanonicalizeOps(),
                                      _transform.FoldConstant()])

    calibrate_pass = _transform.function_pass(calibrate, opt_level=1,
                                              name="QuantizeCalibrate")
    # Quantize pass list
    quant_passes = [annotate(),
                    calibrate_pass,
                    realize(),
                    _transform.FoldConstant()]
    if current_qconfig().store_lowbit_output:
        quant_passes = [rewrite_for_vta()] + quant_passes
    quantize_seq = _transform.Sequential(quant_passes)
    with _transform.PassContext(opt_level=3,
                                required_pass=["QuantizeAnnotate",
                                               "QuantizeCalibrate",
                                               "QuantizeRealize"]):
        mod = optimize(mod)
        mod = quantize_seq(mod)

    return mod["main"]

#TODO(eqy)
#    # TODO(zhiics) Move this to the pass manager.
#    graph = optimize(graph, params)
#
#    graph = annotate(graph)
#    graph = calibrate(graph, dataset)
#    graph = realize(graph)
#    graph = _ir_pass.fold_constant(graph)
#    return graph

def _evaluate(val_data, batch_fn, graph, lib, params, ctx, free_vars=[], config=[], num_classes=1000, early_stopping=32, log_iter=2):
    import mxnet as mx
    """Evaluate function for profiling."""
    import tvm
    import logging
    logging.basicConfig(level=logging.INFO)
    from tvm.contrib import graph_runtime

    # create runtime module
    m = graph_runtime.create(graph, lib, ctx)
    scales = {}

    for i in range(0, len(free_vars)):
        free_var = free_vars[i]
        if i >= len(config):
            shape = m.get_input(i+1).shape
            dummy = np.empty(shape=shape)
            if len(dummy.shape) > 0:
                dummy[:] = np.nan
            else:
                dummy = np.nan
            params[str(free_var.name_hint)] = np.array(dummy)
        else:
            params[str(free_var.name_hint)] = np.array(config[i]/128)

    m.set_input(**params)
    batch_size = 1
    oshape = (batch_size, num_classes)
    out_arr = tvm.nd.empty(oshape, "float32")
    # setup evaluaiton metric
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    # execute

    output_collection = [None]*(m.get_num_outputs() - 1)
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, [mx.cpu(0)])
        m.run(data=data[0].asnumpy())
        m.run(data=data[0].asnumpy(), **scales)
        m.get_output(0, out_arr)
        for o in range(0, len(output_collection)):
            if output_collection[o] is None:
                output_collection[o] = m.get_output(o+1).asnumpy()
            else:
                output_collection[o] = np.concatenate((output_collection[o], m.get_output(o+1).asnumpy()))
        acc_top1.update(label, [mx.nd.array(out_arr.asnumpy())])
        acc_top5.update(label, [mx.nd.array(out_arr.asnumpy())])
        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()

        if not (i + 1) % log_iter:
            nsamples = (i + 1) * batch_size
            print('[{0:d} samples] evaluation: acc-top1={1:f} acc-top5={2:f}'.format(nsamples, top1, top5))

        if (i+1)*batch_size >= early_stopping:
            return top1, output_collection

#def autoquantize(graph_callback, tr_data, tr_batch_fn, granularity='layer'):
def autoquantize(graph, params, tr_data, tr_batch_fn, granularity='layer'):

    import tvm
    import copy
    from tvm import relay
    from tvm.relay import ir_pass

    #graph, params = graph_callback()

    graph = optimize(graph, params)
    with qconfig(skip_k_conv=0,
                 passthrough_bound=int(-1),
                 nbit_input=8,
                 nbit_weight=8,
                 global_scale=8.0,
                 dtype_input='int8',
                 dtype_weight='int8',
                 dtype_activation='int32',
                 store_lowbit_output=True,
                 debug_enabled_ops=None,
                 granularity=granularity):
        layout_map = tag_layout(graph)
        graph = annotate(graph, layout_map)
        annotated = copy.deepcopy(graph)
        graph = ir_pass.infer_type(graph)
        graph, profile_data = calibrate(graph, profile_mode=True, scales=None)
        
        free_vars = list(ir_pass.free_vars(graph))
        graph = relay.Function(list(graph.params) + free_vars,
                                graph.body, graph.ret_type,
                                graph.type_params, graph.attrs)
        additional_outputs = list()
        metadata = list()
        for hint, data, granularity, layout in profile_data:
            additional_outputs.append(data)
            metadata.append((hint, granularity, layout))
        graph = relay.Function(graph.params,
                                relay.expr.Tuple([graph.body]+additional_outputs))
        target = 'llvm -mcpu=core-avx2'
        #target = 'cuda'
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(graph, target)
            ctx = tvm.nd.context(target)

        config = list()
        print("calibrating...")
        t1 = time.time()
        top1, outputs = _evaluate(tr_data, tr_batch_fn, graph, lib, params, ctx, free_vars, early_stopping=32)
        for i, output in enumerate(outputs):
            config.append(_mse_chooser(output, granularity, metadata[i][-1]))
    with qconfig(skip_k_conv=0,
                 passthrough_bound=int(1e9),
                 nbit_input=8,
                 nbit_weight=8,
                 global_scale=8.0,
                 dtype_input='int8',
                 dtype_weight='int8',
                 dtype_activation='int32',
                 store_lowbit_output=True,
                 debug_enabled_ops=None,
                 granularity=granularity):
        #graph, params = graph_callback()
        #graph = optimize(graph, params)
        #layout_map = tag_layout(graph)
        #graph = annotate(graph, layout_map)
        graph = annotated
        graph = ir_pass.infer_type(graph)
        graph, profile_data = calibrate(graph, profile_mode=False, scales=config)
        graph = realize(graph)
    t2 = time.time()
    print("calibrated in approx", t2-t1, "s")
    with relay.build_config(opt_level=3):
        graph = optimize(graph, params)
    return graph
