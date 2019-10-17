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
from . import calibrate as _calibrate
from .. import expr as _expr
from .. import transform as _transform
from .. import analysis as _analysis
from ... import make as _make
from ..base import NodeBase, register_relay_node

# TODO(contributor): remove kind in sq
# TODO(ziheng): refactor the infra to modulized pass

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
        "skip_conv_layers": [0],
        "calibrate_mode": "global_scale",
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


# NOTE:
# behavior of cast_hint
# partition part, will insert cast_hint to denote, which will be
# inserted simulated quantize during annotate

# in some condition we need to defer the cast operartion
# will be transformed to real cast in realize
# but work as identity before realize


# behavior of quantized_add
# add wiil be transformed to quantized_add during annotate,
# oscale will be tuned by calibrate
# if simulated, only do addition, which is used before realize
# during realize, lhs and rhs's scale will be unified as oscale
# then do add
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


def prerequisite_optimize(mod, params=None):
    """ Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization. """
    optimize = _transform.Sequential([_transform.SimplifyInference(),
                                      _transform.FoldConstant(),
                                      _transform.FoldScaleAxis(),
                                      _transform.CanonicalizeOps(),
                                      _transform.FoldConstant()])

    if params:
        mod['main'] = _bind_params(mod['main'], params)

    with _transform.PassContext(opt_level=3):
        mod = optimize(mod)
    return mod


def instantiate(func, num_bits, thresholds):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    const_params = {}
    def visit_func(expr):
        # TODO(ziheng) memorize, e.g. two sq share the same scales
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            sq = expr
            _, nscale, nclip_min, nclip_max = sq.args
            attrs = sq.attrs
            num_bit = num_bits[sq]
            threshold = thresholds[sq]

            valid_bit = num_bit - attrs.sign
            quant_range = 2**valid_bit
            const_params[nscale] = _make_const(threshold / quant_range)
            const_params[nclip_min] = _make_const(- (quant_range - 1))
            const_params[nclip_max] = _make_const((quant_range - 1))

    _analysis.post_order_visit(func, visit_func)
    func = _expr.bind(func, const_params)
    return _module.Module.from_expr(func)

def calibrate(func, num_bits):
    # search scales by auto-tvm
    # Metrics
    # - latency (smaller nbit)
    # - accuracy

    for thresholds in search_space:
        simulated_func = instantiate(func, num_bits, thresholds)
        acc = evaluate(simulated_func)
    return best_acc, best_thresholds

# Hardware Specific

# simulation mostly for accuracy
# track scale for every tensor

# During Search Phase:
# search num_bit for every connection
#   - estimate threshold for every connection by calibration
#   - calculate constant parameters of simulated quantize
#     1. calculate scale, clip_min, clip_max of simulated quantize with (num_bit, threshold)
#     2. inferred scale of every connection
#     2. calculate overflow_min, overflow_max with inferred scale of every connection
#   - build simulated graph with those constant parameters
#   - acc = evaluate it on validation set
#   - TODO: refine threshold after this procedure
# choose num_bit setting

# During Realize Phase:
# infer scale of every connection



# sq(data, scale, clip_min, clip_max, upper_bound, lower_bound, signed=True, rounding='round')
# scale infer: map[op] -> scale

# simulated_qunatize(conv_out, 8, threshold, upper_bound=16):
# - 1. we want requantize it into 8bit
# - 2. we make sure it can fit into 16 bit before quantize

# map: call(sq_op, data, scale, clip_min, clip_max) -> (num_bit, threshold)

# Two representation for tensors:
# - REAL number reprentation
# - INTEGER number reprentation
# can be transformed:
#   scale = real_threshold / integer_range
#   INTEGER = REAL / scale
#   REAL = INTEGER * scale
# 
# Behavior of SimulatedQuantize:
# def simulated_quantize(data, scale, clip_min, clip_max, overflow_lower_bound, overflow_upper_bound):
#     # simulated overflow error
#     # because scale here is output scale, you cannot it to recover the quant of last op's output
#     data = overflow_truncate(data, overflow_lower_bound, overflow_upper_bound)
# 
#     # transform from real to integer(simulated)
#     quant = data / scale
#
#     # simulated rounding error
#     quant = round(quant)
#     # simulated clipping error
#     quant = clip(quant, clip_min, clip_max)
#
#     # transform from integer to real
#     data = quant * scale
#     return data



from collections import namedtuple, defaultdict

TensorReprentation = namedtuple('TensorRepresentation', ['num_bit', 'threshold'])
SimulatedQuantizeParams = namedtuple("SimulatedQuantizeParams", ['scale',
                                                                 'clip_min',
                                                                 'clip_max',
                                                                 'overflow_min',
                                                                 'overflow_max'])


# for different hardwares, we need to consider instructions that it support. Reflect on graph level:
# - dtype constraint
# - shape constraint
# - layout constraint
# Consider:
# - Similarities with:
#   - TypeInfer of Op
#   - TensorIntrinsic
# - VTA, GPU:TensorCore, Quantization, LayoutTransform
class HardwareDescription(object):
    def __init__(self):
        self._op_constraints = defaultdict(list)

    def __getitem__(self, op_name):
        return self._op_constraints[op_name]

    @property
    def ops(self):
        return self._op_constraints.keys()

def create_accelerator_description():
    # TODO: change to DataType
    desc = HardwareDescription()
    desc['add'].append(([8, 8], [16]))
    desc['add'].append(([8, 8], [32]))
    desc['nn.conv2d'].append(([8, 8], [16]))
    desc['nn.conv2d'].append(([8, 8], [32]))
    return desc


def generate_bit_choices(graph, description):
    """tensor_id -> max_bit
       [set(8, 16)]
       Question:
        - do we need to consider output nbit constraint?
    """
    # build indexing map
    expr2idx = {}  # expr(var/call) to idx
    def fvisit_build_index(e):
        if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
            expr2idx[e] = fvisit_build_index.idx_cnt 
            fvisit_build_index.idx_cnt += 1
    fvisit_build_index.idx_cnt = 0
    _analysis.post_order_visit(graph, fvisit_build_index)

    # def fvisit_test(e):
    #     if isinstance(e, (_expr.Var, _expr.Constant, _expr.Call)):
    #         print(expr2idx[e])
    # _analysis.post_order_visit(graph, fvisit_test)

    # analysis maximum num of bit on every tensor/edge
    max_bits = [32 for _ in range(fvisit_build_index.idx_cnt)]
    def fvisit_max_bits(e):
        if isinstance(e, (_expr.Var, _expr.Constant)):
            # use 8 bit for variables/weights
            idx = expr2idx[e]
            max_bits[idx] = 8
        elif isinstance(e, _expr.Call):
            if e.op.name in description.ops:
                constraints = description[e.op.name]
                max_inputs_bit = constraints[0][0]
                for (inputs_bit, outputs_bit) in constraints:
                    max_inputs_bit = (max(v1, v2) for v1, v2
                                      in zip(inputs_bit, max_inputs_bit))
                for (input_expr, max_input_bit) in zip(e.args, max_inputs_bit):
                    idx = expr2idx[input_expr]
                    max_bits[idx] = max(max_bits[idx], max_input_bit)
        else:
            return
    _analysis.post_order_visit(graph, fvisit_max_bits)

    print(max_bits)
    bit_choices = [list(range(1, max_bit)) for max_bit in max_bits]
    return bit_choices


def grid_search_sample(bit_choices):
    # for resnet: 113^(8/32)
    pass

def calculate_quantize_op_params(bits, thresholds):
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
    # overflow_lower_bound_integer = - (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_lower_bound_real    = overflow_lower_bound_quant * input_scale
    # overflow_upper_bound_integer = (2 ^ (overflow_num_bit - 1) - 1)
    # overflow_upper_bound_real    = overflow_upper_bound_quant * input_scale
    pass


def search(graph, hardware_description, valid_dataset=None):
    cfg = current_qconfig()
    # map: call(sq_op) -> (scale, clip_min, clip_max, overflow_lower_bound, overflow_upper_bound)
    bit_choices = generate_bit_choices(graph, hardware_description)

    # search for bits settings with learning method
    for bits in grid_search_sample(bit_choices):
        thresholds = threshold_estimate(func, bits)
        op_params = calculate_quantize_op_params(bits, thresholds)
        simulated_graph = simulate(graph, op_params)
        acc = eval_acc(simulated_graph, valid_dataset)
        # [optional] calibrate threshold estimation
    return best_bits, best_thresholds, best_acc


def quantize(mod, params=None, dataset=None):
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
    mod = prerequisite_optimize(mod, params)

    desc = create_accelerator_description()
    search(mod['main'], desc)

    raise ValueError


    # calibrate_pass = _transform.module_pass(_calibrate.calibrate(dataset),
    #                                         opt_level=1,
    #                                         name="QuantizeCalibrate")
    # quant_passes = [
    #                 annotate(),
    #                 calibrate_pass]
    # if not current_qconfig().do_simulation:
    #     quant_passes.append(realize())
    # quant_passes.append(_transform.FoldConstant())
    # quantize_seq = _transform.Sequential(quant_passes)
    # with _transform.PassContext(opt_level=3):
    #     with quantize_context():
    #         mod = quantize_seq(mod)

    return mod
