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
from tvm.contrib import graph_executor

from . import _quantize
from ._calibrate import calibrate, read_calibrate
from ._partition_conversions import partition_conversions
from .. import expr as _expr
from .. import transform as _transform
import numpy as np
from tvm import relay
import os
import tqdm

def get_consine_similar(m1, m2):
    m1 = m1.flatten()
    m2 = m2.flatten()
    num = float(np.dot(m1, m2))
    denom = np.linalg.norm(m1) * np.linalg.norm(m2)
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def profile_data_per_layer(quantized_mod, target, params, dataset, ifdev=True):

    print("Profile data per layer...")
    func = quantized_mod["main"]
    func_original = _quantize.CreateStatsCollector(func)
    func_quantized = _quantize.CreateQActCollector(func)
    assert dataset
    assert len(dataset) == 1
    dev = tvm.device(str(target), 0)

    with tvm.transform.PassContext(opt_level=3):
        original_lib = relay.build(func_original, target=target, params=params)
    
    with tvm.transform.PassContext(opt_level=3):
        quantized_lib = relay.build(func_quantized, target=target, params=params)

    original_module = graph_executor.GraphModule(original_lib["default"](dev))
    quantized_module = graph_executor.GraphModule(quantized_lib["default"](dev))
    
    per_layer_list = []
    batch = dataset[0]
    original_module.set_input(**batch)
    original_module.run()
    quantized_module.set_input(**batch)
    quantized_module.run()
    num_original_outputs = original_module.get_num_outputs()
    num_quantized_outputs = quantized_module.get_num_outputs()

    assert num_original_outputs == num_quantized_outputs

    for j in range(num_original_outputs):
        original_module_output = original_module.get_output(j).numpy()
        quantized_module_output = quantized_module.get_output(j).numpy()
        assert original_module_output.shape == quantized_module_output.shape
        cosine_res_tmp = get_consine_similar(original_module_output, quantized_module_output)
        per_layer_list.append(cosine_res_tmp)
    
    assert os.path.exists(current_qconfig().get_rootdir_name())
    saved_file_name = "cosine_similarity_dev_pl" if ifdev else "cosine_similarity_calibration_pl"
    count = 0
    with open(current_qconfig().get_rootdir_name() + "/" + saved_file_name, "w") as f:
        for ele in per_layer_list:
            f.write("Layer{}".format(count) + ": {}".format(ele) + "\n")
            count = count + 1


def calculate_consine_similar(original_mod, quantized_mod, target, params, dataset=None, ifdev=True):
    """
    Calculate result's cosine_similarity and generate cosine_similarity.txt in dir.
    Returns: mean cosine similarity, similarity per batch
    """
    print("Calculate consine similarity...")
    assert dataset

    dev = tvm.device(str(target), 0)

    with tvm.transform.PassContext(opt_level=3):
        original_lib = relay.build(original_mod, target=target, params=params)
    
    with tvm.transform.PassContext(opt_level=3):
        quantized_lib = relay.build(quantized_mod, target=target, params=params)

    original_module = graph_executor.GraphModule(original_lib["default"](dev))
    quantized_module = graph_executor.GraphModule(quantized_lib["default"](dev))
    
    batch_count = 0
    cos_similar_result = 0
    batch_cos_list = []
    for batch in tqdm.tqdm(dataset):
        original_module.set_input(**batch)
        original_module.run()
        quantized_module.set_input(**batch)
        quantized_module.run()
        num_original_outputs = original_module.get_num_outputs()
        num_quantized_outputs = quantized_module.get_num_outputs()
        cos_tmp = 0
        assert num_original_outputs == num_quantized_outputs
        for j in range(num_original_outputs):
            original_module_output = original_module.get_output(j).numpy()
            quantized_module_output = quantized_module.get_output(j).numpy()
            assert original_module_output.shape == quantized_module_output.shape
            consine_res_tmp = get_consine_similar(original_module_output, quantized_module_output)
            cos_tmp += consine_res_tmp
        cos_similar_result += (cos_tmp / num_original_outputs)
        batch_cos_list.append(cos_tmp / num_original_outputs)
        batch_count = batch_count + 1
    
    assert os.path.exists(current_qconfig().get_rootdir_name())
    saved_file_name = "cosine_similarity_dev" if ifdev else "cosine_similarity_calibration"
    with open(current_qconfig().get_rootdir_name() + "/" + saved_file_name, "w") as f:
        f.write("Mean similarity: " + str(cos_similar_result/batch_count) + "\n")
        for i in range(batch_count):
            f.write("Batch{}".format(i) + ": {}".format(batch_cos_list[i]) + "\n")

    return cos_similar_result / batch_count, batch_cos_list


class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""

    IDENTITY = 0
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3
    BIAS = 4


def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
        QAnnotateKind.IDENTITY: "identity",
        QAnnotateKind.BIAS: "bias",
    }
    assert kind in str_map
    return str_map[kind]

def kind2aw(kind):
    """Convert a `QAnnotateKind` to `activation` or `weight`"""
    str_map = {
        QAnnotateKind.INPUT: "activation",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.BIAS: "bias",
        QAnnotateKind.ACTIVATION: "activation",
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
        "network_name": "Default",
        "have_prequantized": False,
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "nbit_bias": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "dtype_bias": "int32",
        "estimator_activation": "MSE",
        "estimator_weight": "MSE",
        "estimator_bias": "MSE",
        "skip_dense_layer": False,
        "skip_conv_layers": None,
        "skip_add_layers": None,
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
        "rounding": "UPWARD",
        "calibrate_chunk_by": -1,
        "partition_conversions": "disabled",
        "quantizer_weight": "Symmetric",
        "quantizer_activation": "Asymmetric",
        "quantizer_bias": "Symmetric",
        "per_channel": True,
        "opt_method": "grid",
        "debug_mode": False,
        "global_scale": 8.0,
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
    
    # format: weight + bits + Asym/Sym + estimator + C/F(per-channel or per-filter) + "_" + act + bits + Asym/Sym + estimator 
    def get_rootdir_name(self):
        wgt_bits = str(getattr(self, "nbit_weight"))
        wgt_as = "Asym" if getattr(self, "quantizer_weight") == "Asymmetric" else "Sym"
        wgt_es = getattr(self, "estimator_weight")
        wgt_cf = "C" if getattr(self, "per_channel") else "F"
        act_bits = str(getattr(self, "nbit_input"))
        act_as = "Asym" if getattr(self, "quantizer_activation") == "Asymmetric" else "Sym"
        act_es = getattr(self, "estimator_activation")
        dir_name = "Wgt" + wgt_bits + wgt_as + wgt_es.capitalize() + wgt_cf + "__" + "Act" + act_bits + act_as + act_es.capitalize()
        return dir_name

    def get_debug_mode(self):
        return getattr(self, "debug_mode")

    def get_prequantized(self):
        return getattr(self, "have_prequantized")

    def get_network_name(self):
        return getattr(self, "network_name")

    def get_nbit_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, "nbit_" + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, "dtype_" + name)
    
    def get_quantizer_by_kind(self, kind):
        name = kind2aw(kind)
        return getattr(self, "quantizer_" + name)
    
    def get_estimator_by_kind(self, kind):
        name = kind2aw(kind)
        return getattr(self, "estimator_" + name)
    
    def get_perChannel_by_kind(self, kind):
        if kind == QAnnotateKind.INPUT:
            return False
        elif kind == QAnnotateKind.WEIGHT:
            return getattr(self, "per_channel")
        else:
            return False
    
    def get_optMethod(self):
        return getattr(self, "opt_method")

    def get_thread_num(self):
        return getattr(self, "num_thread")

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope()

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError("'%s' object cannot set attribute '%s'" % (str(type(self)), name))
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
        self._add_counter = 0
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
        
        if current_qconfig().skip_add_layers is not None:
            # check skip add layers
            skipped_indices = [int(x) for x in current_qconfig().skip_add_layers]
            if self._add_counter in skipped_indices and ref_call.op.name == "add":
                self._add_counter += 1
                return True
            if ref_call.op.name == "add":
                self._add_counter += 1

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


def annotate_for_inference():
    """Given a float32 graph, this pass will rewrite the graph and return
    a graph which simulates the error brought by the current quantization
    scheme.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass for quantization annotation.
    """
    return _quantize.QuantizeAnnotateForInference()

def annotate_for_calibration():
    """Given a float32 graph, this pass will rewrite the graph and return
    a graph which simulates the error brought by the current quantization
    scheme.

    Returns
    -------
    ret: tvm.transform.Pass
        The registered pass for quantization calibrate annotation.
    """
    return _quantize.QuantizeAnnotateForCalibrate()


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
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def prerequisite_optimize(mod, params=None):
    """Prerequisite optimization passes for quantization. Perform
    "SimplifyInference", "FoldScaleAxis", "FoldConstant", and
    "CanonicalizeOps" optimization before quantization."""
    optimize = tvm.transform.Sequential(
        [
            _transform.CanonicalizeOps(),
            _transform.SimplifyInference(),
            _transform.RemoveUnusedFunctions(),
            _transform.FoldConstant(),
            _transform.FoldScaleAxis(),
            _transform.FoldSumsPass(), # for batchnorm condition
            _transform.FoldConstant(),
            _transform.InferType(),
        ]
    )

    if params:
        mod["main"] = _bind_params(mod["main"], params)

    mod = optimize(mod)
    return mod

def quantize_calibrate(mod, params=None, dataset=None):
    """The quantization calibrate procedure

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

    # this pass is to quantize every layers in networks for general purpose
    from ._annotate import annotate_for_calibrate_registry
    annotate_for_calibrate_registry(mod)

    calibrate_pass = tvm.transform.module_pass(
        calibrate(dataset), opt_level=1, name="QuantizeCalibrate"
    )
    quant_passes = [annotate_for_calibration(), calibrate_pass, tvm.relay.transform.InferType()]

    #quant_passes.append(_transform.FoldConstant())
    quantize_seq = tvm.transform.Sequential(quant_passes)
    with tvm.transform.PassContext(
        opt_level=3, required_pass=["QuantizeAnnotateForCalibrate", "QuantizeCalibrate"]
    ):
        with quantize_context():
            dbg_mod = quantize_seq(mod)
    
    # dbg_mod does not fold constant for print weight purpose
    mod = _transform.FoldConstant()(dbg_mod)

    return mod, dbg_mod

def quantize_inference(mod, params=None):
    """The quantization calibrate procedure

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

    from ._annotate import annotate_for_inference_registry
    annotate_for_inference_registry(mod)

    calibrate_read_pass = tvm.transform.module_pass(
        read_calibrate(), opt_level=1, name="QuantizeCalibrate"
    )
    quant_passes = [partition(), annotate_for_inference(), calibrate_read_pass, tvm.relay.transform.InferType()]
    if not current_qconfig().do_simulation:
        quant_passes.append(realize())

    quantize_seq = tvm.transform.Sequential(quant_passes)
    with tvm.transform.PassContext(
        opt_level=3, required_pass=["QuantizeAnnotate", "QuantizeCalibrate", "QuantizeRealize"]
    ):
        with quantize_context():
            dbg_mod = quantize_seq(mod)
    
    # dbg_mod does not fold constant for print weight purpose
    mod = _transform.FoldConstant()(dbg_mod)

    return mod, dbg_mod