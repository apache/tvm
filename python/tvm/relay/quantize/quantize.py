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
# pylint: disable=unused-argument, not-context-manager
"""Automatic quantization toolkit."""
import tvm.ir
import tvm
from tvm.runtime import Object

from . import _quantize
from ._calibrate import calibrate
from ._partition_conversions import partition_conversions
from .. import expr as _expr
from .. import transform as _transform


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
        QAnnotateKind.IDENTITY: "identity",
    }
    assert kind in str_map
    return str_map[kind]


def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return _expr.Call(ref_call.op, args, ref_call.attrs, ref_call.type_args, ref_call.span)


@tvm._ffi.register_object("relay.quantize.QConfig")
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
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "calibrate_mode": "global_scale",
        "global_scale": 8.0,
        "weight_scale": "power2",
        "skip_dense_layer": True,
        "skip_conv_layers": [0],
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
        "rounding": "UPWARD",
        "calibrate_chunk_by": -1,
        "partition_conversions": "disabled",
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
        return getattr(self, "nbit_" + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, "dtype_" + name)

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope()

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError(f"'{type(self)}' object cannot set attribute '{name}'")
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

    calibrate_mode: str
        The calibration mode. 'global_scale' or 'kl_divergence'.
        global_scale: use global scale
        kl_divergence: find scales by kl divergence on the dataset.

    global_scale: float
        The global scale for calibration.

    weight_scale: str
        The way to calculate scales for weights (annotated with QAnnotateKind.WEIGHT).
        power2: Find the maximum of the absolute value of the tensor, and then round up to power
        of two.
        max: Find the maximum of the absolute value of the tensor

    skip_dense_layer: boolean
        Whether to skip all nn.dense layer type. By default are skipped.

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

    rounding: "UPWARD" or "TONEAREST"
        Rounding direction for fixed point multiplications.

    partition_conversions: 'disabled', 'enabled', or 'fully_integral'
        If set to 'enabled' or 'fully_integral', partitions a quantized
        result into a module containing
        a prefix function (consisting of input conversion into the quantized data space),
        a middle function (consisting of the core quantized network),
        a suffix function (consisting of output dequantization),
        and a main function (that calls the prefix, middle, and suffix functions in succession).
        If set to 'fully_integral' and there are unquantized operators in the result,
        an exception is raised.
        The default value is 'disabled'.

    Returns
    -------
    config: QConfig
        The quantization configuration
    """
    node_args = {k: v if k not in kwargs else kwargs[k] for k, v in QConfig._node_defaults.items()}
    return tvm.ir.make_node("relay.quantize.QConfig", **node_args)


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
            if self._conv2d_counter in skipped_indices and ref_call.op.name == "nn.conv2d":
                self._conv2d_counter += 1
                return True
            if ref_call.op.name == "nn.conv2d":
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
    ret: tvm.transform.Pass
        The registered pass for VTA rewrite.
    """
    return _quantize.QuantizePartition()


def annotate():
    """Given a float32 graph, this pass will rewrite the graph and return
    a graph which simulates the error brought by the current quantization
    scheme.

    Returns
    -------
    ret: tvm.transform.Pass
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
    ret: tvm.transform.Pass
        The registered pass for quantization realization.
    """
    return _quantize.QuantizeRealize()


def _bind_params(func, params):
    """Bind the params to the expression."""
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
            raise ValueError(f"Multiple args in the function have name {k}")
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def prerequisite_optimize(mod, params=None):
    """Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization."""
    optimize = tvm.transform.Sequential(
        [
            _transform.SimplifyInference(),
            _transform.FoldConstant(),
            _transform.FoldScaleAxis(),
            _transform.CanonicalizeOps(),
            _transform.FoldConstant(),
        ]
    )

    if params:
        mod["main"] = _bind_params(mod["main"], params)

    mod = optimize(mod)
    return mod


def quantize(mod, params=None, dataset=None):
    """The quantization procedure. Before running the three main
    procedure of quantization, "annotate", "calibrate" and "realize"
    , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
    first for optimizing.

    Parameters
    ---------
    mod: Module
        The original module.

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

    calibrate_pass = tvm.transform.module_pass(
        calibrate(dataset), opt_level=1, name="QuantizeCalibrate"
    )
    quant_passes = [partition(), annotate(), calibrate_pass, tvm.relay.transform.InferType()]
    if not current_qconfig().do_simulation:
        quant_passes.append(realize())
    quant_passes.append(_transform.FoldConstant())
    quantize_seq = tvm.transform.Sequential(quant_passes)
    with tvm.transform.PassContext(
        opt_level=3, required_pass=["QuantizeAnnotate", "QuantizeCalibrate", "QuantizeRealize"]
    ):
        with quantize_context():
            mod = quantize_seq(mod)

    q_cfg = current_qconfig()
    assert q_cfg.partition_conversions in ["disabled", "enabled", "fully_integral"]
    if q_cfg.partition_conversions != "disabled":
        quantized_dtypes = {q_cfg.dtype_input, q_cfg.dtype_weight, q_cfg.dtype_activation}
        ensure_fully_integral = q_cfg.partition_conversions == "fully_integral"
        return partition_conversions(mod, quantized_dtypes, ensure_fully_integral)

    return mod
