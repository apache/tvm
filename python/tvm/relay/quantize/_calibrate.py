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
"""Find scales for quantization on the dataset."""
from __future__ import absolute_import
import logging
import multiprocessing as mp
from multiprocessing import managers
import numpy as np
import tvm
import tvm.driver
from tvm.ir import IRModule
import tqdm
import os
import re
import json
import time

from . import _quantize
from . import quantize
from .. import op as _op
from .. import expr as _expr
from .. import analysis as _analysis
from .. import build_module as _build_module
from ...contrib import graph_executor
from .kl_divergence import _find_scale_by_kl
from .quantizers import AsymmetricUniformQuantizer, SymmetricUniformQuantizer
from .range_estimators import OptMethod, MSE_Estimator, CrossEntropyEstimator, KLDivergence, AllMinMaxEstimator, CosineSimilarityEstimator
from ..frontend.common import (
    infer_shape,
    infer_type,
)

__quantize_version__ = "1.0"

class MyManager(managers.BaseManager):
    """
    multiprocessing manager
    """
    pass

# Register quantizer and estimator objects
MyManager.register("MSE_Estimator", MSE_Estimator)
MyManager.register("CrossEntropyEstimator", CrossEntropyEstimator)
MyManager.register("KLDivergence", KLDivergence)
MyManager.register("AllMinMaxEstimator", AllMinMaxEstimator)
MyManager.register("CosineSimilarityEstimator", CosineSimilarityEstimator)
# MyManager.register("AsymmetricUniformQuantizer", AsymmetricUniformQuantizer)
# MyManager.register("SymmetricUniformQuantizer", SymmetricUniformQuantizer)
manager = MyManager()
manager.start()

#zzk_debug: I think Hessein trace is better for finding out the vulnerable layers.
def get_cosine_similar(m1, m2):
    m1 = m1.flatten()
    m2 = m2.flatten()
    num = float(np.dot(m1, m2))
    denom = np.linalg.norm(m1) * np.linalg.norm(m2)
    return 0.5 + 0.5 * (num / denom) if denom !=0 else 0

def _get_profile_runtime(mod):
    func = mod["main"]
    func = _quantize.CreateStatsCollector(func)

    if tvm.target.Target.current():
        target = tvm.target.Target.current()
        dev = tvm.device(target.kind.name)
    else:
        target = "llvm"
        dev = tvm.device(target)

    with tvm.transform.PassContext(opt_level=3):
        lib = _build_module.build(func, target=target)
    runtime = graph_executor.GraphModule(lib["default"](dev))

    return runtime


def collect_stats(runtime, batch, estimator_node, chunk_by=-1):
    """Given an annotated graph, create a profile graph to collect profile data from the
    calibration dataset. This pass collects simulated_quantize op input into a tuple.
    Simulated_quantize ops are rewritten to identity mode. The tuple is the output of the profile
    graph.

    Parameters
    ----------
    mod: Module
        The simulation graph after annotation.

    dataset: Iterable[NDArray]
        The calibration dataset.

    chunk_by: optional, int
        The size of chunk to be returned in one iteration. It is meant to be
        used for reducing memory usage. If not specified, return samples for
        all layers in one chunk.

    Returns
    -------
    ret: Iterable[list of ndarray]
        List of output data of each layer, chunked by the chunk_by parameter
    """

    num_outputs = runtime.get_num_outputs()
    chunk_by = num_outputs if chunk_by == -1 else chunk_by

    runtime.set_input(**batch)
    runtime.run()

    # for i in range(0, num_outputs, chunk_by):
    #     outputs = []
    #     estimator_node_output = []
    #     for j in range(i, min(i + chunk_by, num_outputs)):
    #         outputs.append(runtime.get_output(j).numpy())
    #         estimator_node_output.append(estimator_node[j])
    #     yield [[outputs[k].reshape(-1), estimator_node_output[k]] \
    #             for k in range(min(chunk_by, num_outputs - i))]

    outputs = []
    estimator_node_output = []
    for i in range(0, num_outputs):
        outputs.append(runtime.get_output(i).numpy())
        estimator_node_output.append(estimator_node[i])
    yield [[outputs[k].reshape(-1), estimator_node_output[k]] \
            for k in range(num_outputs)]


def collect_min_max(mod, dataset):
    """Given an annotated graph, create a profile graph to collect profile data from the
    calibration dataset. This pass collects simulated_quantize op input into a tuple.
    Collect every simulated_quantize op's min max situation.

    Parameters
    ----------
    mod: Module
        The simulation graph after annotation.

    dataset: Iterable[NDArray]
        The calibration dataset.

    Returns
    -------
    ret: Iterable[list of ndarray]
        List of min-max of each layer.
    """
    print("Starting collect activation's min-max...")
    runtime = _get_profile_runtime(mod)
    num_outputs = runtime.get_num_outputs()

    # min max stores in [min, max] formats
    min_max_list = [[] for i in range(num_outputs)]

    for batch in tqdm.tqdm(dataset):
        runtime.set_input(**batch)
        runtime.run()
        for i in range(0, num_outputs):
            layer_output=runtime.get_output(i).numpy()
            max_value=np.max(layer_output)
            min_value=np.min(layer_output)
            if not min_max_list[i]:
                min_max_list[i].append(min_value)
                min_max_list[i].append(max_value)
            else:
                if min_value < min_max_list[i][0]:
                    min_max_list[i][0] = min_value
                if max_value > min_max_list[i][1]:
                    min_max_list[i][1] = max_value
    
    return min_max_list

def initialize_quantizer(mod, min_max):
    """
    Initialize quantizer and Estimator here
    """
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    cfg = quantize.current_qconfig()
    quantizer_weight_node = []
    estimator_weight_node = []
    quantizer_act_node = []
    estimator_act_node = []
    quantizer_bias_node = []
    print("Initializing quantizers and estimators...")

    def visit_func(expr):
        """visitor function for traverse"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            attrs = expr.attrs
            kind = attrs.kind
            assert kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT or \
                   kind == quantize.QAnnotateKind.BIAS
            
            if kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT:
                nbit = cfg.get_nbit_by_kind(kind)
                per_channel = attrs.per_channel
                quantizer_type = cfg.get_quantizer_by_kind(kind)
                estimator_type = cfg.get_estimator_by_kind(kind)
                opt_method_type = cfg.get_optMethod()
                quantizer_name = attrs.name

                if(opt_method_type == "grid"):
                    opt_method = OptMethod.grid
                elif(opt_method_type == "golden_section"):
                    opt_method = OptMethod.golden_section
                else:
                    raise ValueError("Unknown opt method type {}".format(opt_method_type))
                
                if(quantizer_type == "Asymmetric"):
                    quantizer_node = AsymmetricUniformQuantizer(quantizer_name, nbit, per_channel=per_channel)
                elif(quantizer_type == "Symmetric"):
                    quantizer_node = SymmetricUniformQuantizer(quantizer_name, nbit, per_channel=per_channel)
                else:
                    raise ValueError("Unknown quantizer type {}".format(quantizer_type))
                
                # Although we have defined many estimators, use MSE KL Crossentropy or min-max here
                if(estimator_type == "MSE"):
                    estimator_node = manager.MSE_Estimator(num_candidates=1000, opt_method=opt_method,
                                                        per_channel=per_channel, quantizer=quantizer_node)
                elif(estimator_type == "kl_divergence"):
                    estimator_node = manager.KLDivergence(num_bins=8001, per_channel=per_channel, 
                                                        quantizer=quantizer_node)
                elif(estimator_type == "cross_entropy"):
                    estimator_node = manager.CrossEntropyEstimator(per_channel=per_channel, 
                                            quantizer=quantizer_node)
                elif(estimator_type == "min_max"):
                    estimator_node = manager.AllMinMaxEstimator(per_channel=per_channel, quantizer=quantizer_node)
                elif(estimator_type == "cosine_similarity"):
                    estimator_node = manager.CosineSimilarityEstimator(per_channel=per_channel, quantizer=quantizer_node)
                else:
                    raise ValueError("Unknown estimator type {}".format(estimator_type))
                
                # per-channel only works for weight
                if kind == quantize.QAnnotateKind.INPUT and not cfg.get_prequantized():
                    estimator_node.set_min_max(min_max[visit_func.idx_act][0], min_max[visit_func.idx_act][1])
                    visit_func.idx_act += 1
                
                if kind == quantize.QAnnotateKind.INPUT:
                    quantizer_act_node.append(quantizer_node)
                    estimator_act_node.append(estimator_node)
                else:
                    quantizer_weight_node.append(quantizer_node)
                    estimator_weight_node.append(estimator_node)

            elif kind == quantize.QAnnotateKind.BIAS:
                nbit = cfg.get_nbit_by_kind(kind)
                quantizer_name = attrs.name
                quantizer_node = SymmetricUniformQuantizer(quantizer_name, nbit, per_channel=True)
                quantizer_bias_node.append(quantizer_node)

    visit_func.idx_act = 0
    visit_func.idx_weight = 0

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)
    return quantizer_weight_node, estimator_weight_node, quantizer_act_node, estimator_act_node, quantizer_bias_node

def quantize_weight(mod, quantizer_node, estimator_node, threshold=0.95):
    """
    Quantize Weight
    """
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    cfg = quantize.current_qconfig()

    cosine_similarity_results = []

    pbar = tqdm.tqdm(total = len(estimator_node), desc='Wgt Quantization')

    def visit_func_weight(expr):
        """visitor function for traverse"""
        nonlocal pbar
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            attrs = expr.attrs
            kind = attrs.kind
            assert kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT \
                    or kind == quantize.QAnnotateKind.BIAS

            if kind == quantize.QAnnotateKind.WEIGHT:
                assert isinstance(expr.args[0], _expr.Constant)
                weight_min, weight_max = estimator_node[visit_func_weight.idx].calibrate(expr.args[0].data.numpy())
                quantizer_node[visit_func_weight.idx].set_quant_range(weight_min, weight_max)
                quantizer_node[visit_func_weight.idx].adjust_per_channel(expr.args[0].data.numpy())
                if cfg.get_debug_mode():
                    quantized_wgt = quantizer_node[visit_func_weight.idx](expr.args[0].data.numpy())
                    original_wgt = expr.args[0].data.numpy()
                    cosine_similarity_results.append([quantizer_node[visit_func_weight.idx].name, get_cosine_similar(quantized_wgt, original_wgt)])
                visit_func_weight.idx += 1
                pbar.update(1)
    
    print("Start quantizing weight...")
    visit_func_weight.idx = 0
    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func_weight)

    if cfg.get_debug_mode():
        root_dir_name = cfg.get_rootdir_name()
        saved_dir_name = root_dir_name + "/debug_dir"
        if not os.path.exists(root_dir_name):
            os.mkdir(root_dir_name)
        if not os.path.exists(saved_dir_name):
            os.mkdir(saved_dir_name)
        
        with open(saved_dir_name + "/weight_debug.txt", "w") as f:
            f.write("Calibration results of consine similarity.\n")
            for ele in cosine_similarity_results:
                f.write("{}: {}\n".format(ele[0], ele[1]))
            f.write("\n")
            f.write("Layers which need higher bits.\n")
            for ele in cosine_similarity_results:
                if ele[1] < threshold:
                    f.write("{}: {}\n".format(ele[0], ele[1]))
            f.close()

def quantize_activation_intrin(data_tmp):
    layer_output = data_tmp[0]
    estimator_node = data_tmp[1]
    act_min, act_max = estimator_node.calibrate(layer_output)

    return [act_min, act_max]
        

def quantize_activation(mod, quantizer_node, estimator_node, dataset, min_max):
    """
    Quantize Activation

    returns: [min-value, max-value]
    """
    print("Start calibrating activations...")
    cfg = quantize.current_qconfig()
    chunk_by = cfg.calibrate_chunk_by
    
    qact_runtime = _get_profile_runtime(mod)
    num_workers = chunk_by if chunk_by != -1 else 32
    quantization_infos = []

    # skip allminmax since the information has been collected
    if cfg.get_estimator_by_kind(quantize.QAnnotateKind.INPUT) == "min_max":
        assert(len(min_max) == len(quantizer_node))
        for i, element in enumerate(min_max):
           quantizer_node[i].set_quant_range(element[0], element[1])
    else:
        for batch in tqdm.tqdm(dataset, desc="Calibrating Activation"):
            for samples in collect_stats(qact_runtime, batch, estimator_node, chunk_by):
                with mp.Pool(num_workers) as pool:
                    quantization_infos += list(pool.map(quantize_activation_intrin, samples))
        assert(len(quantization_infos) == len(quantizer_node))
        assert(len(quantization_infos) == len(estimator_node))
        for i in range(len(quantization_infos)):
            quantizer_node[i].set_quant_range(quantization_infos[i][0], quantization_infos[i][1])

    return quantization_infos

def calculate_precision(mod, quantizer_node, dataset, threshold=0.9):
    """
    Calculate Every activation quantization Precision

    returns: void
    output: debug_file
    """
    
    print("Output precision...")
    cfg = quantize.current_qconfig()
    runtime = _get_profile_runtime(mod)
    num_outputs = runtime.get_num_outputs()
    #[layer_name, cosine similarity format]
    cosine_similarity_out = [0 for i in range(num_outputs)]
    output_list = []

    for batch in tqdm.tqdm(dataset, desc="Calculate Precision"):
        runtime.set_input(**batch)
        runtime.run()
        for i in range(0, num_outputs):
            output_tmp = runtime.get_output(i).numpy()
            this_quantizer_node = quantizer_node[i]
            quantized_layer_out = this_quantizer_node(output_tmp)
            cosine_similarity_out[i] += get_cosine_similar(output_tmp, quantized_layer_out)

    for i in range(0, num_outputs):
        output_list.append([quantizer_node[i].name, cosine_similarity_out[i]/len(dataset)])
    
    root_dir_name = cfg.get_rootdir_name()
    saved_dir_name = root_dir_name + "/debug_dir"
    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    if not os.path.exists(saved_dir_name):
        os.mkdir(saved_dir_name)

    with open(saved_dir_name + "/activation_calibration_debug.txt", "w") as f:
        f.write("Calibration results of cosine similarity\n")
        for ele in output_list:
            f.write("{}: {}\n".format(ele[0], ele[1]))
        f.write("\n")
        f.write("Layers which need higher bits\n")
        for ele in output_list:
            if ele[1] < threshold:
                f.write("{}: {}\n".format(ele[0], ele[1]))
        f.close()

def _set_params(mod, quantizer_weight_node, quantizer_act_node):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    cfg = quantize.current_qconfig()
    const_params = {}

    def visit_func(expr):
        """visitor function for traverse"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max, zero_point = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            if (kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT):
                assert kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT

                if kind == quantize.QAnnotateKind.WEIGHT:
                    scale = quantizer_weight_node[visit_func.weight_index].scale
                    clip_min = quantizer_weight_node[visit_func.weight_index].int_min
                    clip_max = quantizer_weight_node[visit_func.weight_index].int_max
                    zero_point_q = quantizer_weight_node[visit_func.weight_index].zero_point
                    visit_func.weight_index += 1
                elif kind == quantize.QAnnotateKind.INPUT:
                    scale = quantizer_act_node[visit_func.act_index].scale
                    clip_min = quantizer_act_node[visit_func.act_index].int_min
                    clip_max = quantizer_act_node[visit_func.act_index].int_max
                    zero_point_q = quantizer_act_node[visit_func.act_index].zero_point
                    visit_func.act_index += 1
                else:
                    raise ValueError

                def _make_const(val):
                    return _expr.const(val)
                    
                const_params[ndom_scale] = _make_const(scale)
                const_params[nclip_min] = _make_const(np.array(clip_min, dtype="float32"))
                const_params[nclip_max] = _make_const(np.array(clip_max, dtype="float32"))
                const_params[zero_point] = _make_const(zero_point_q)

    visit_func.weight_index = 0
    visit_func.act_index = 0

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)
    main_func = _expr.bind(main_func, const_params)
    func_dict = {}
    for global_var, func in mod.functions.items():
        if global_var.name_hint != "main":
            func_dict[global_var] = func
    return IRModule.from_expr(main_func, func_dict)

def _set_params_rc(mod, act_dict, wgt_dict):
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    const_params = {}
    cfg = quantize.current_qconfig()

    def visit_func(expr):
        """visitor function for traverse"""
        if isinstance(expr, _expr.Call) and expr.op == quantize_op:
            _, ndom_scale, nclip_min, nclip_max, zero_point = expr.args
            attrs = expr.attrs
            kind = attrs.kind
            if (kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT \
                or kind == quantize.QAnnotateKind.BIAS):
                assert kind == quantize.QAnnotateKind.WEIGHT or kind == quantize.QAnnotateKind.INPUT \
                    or kind == quantize.QAnnotateKind.BIAS
                      
                if kind == quantize.QAnnotateKind.WEIGHT:
                    quantizer_type = cfg.get_quantizer_by_kind(kind)
                    wgt_dict_tmp = wgt_dict[attrs.name]
                    scale = np.array(wgt_dict_tmp["scale"], dtype="float32")
                    clip_min = wgt_dict_tmp["int_min"]
                    clip_max = wgt_dict_tmp["int_max"]
                    zero_point_q = np.array(wgt_dict_tmp["zero_point"], dtype="float32")
                    if attrs.per_channel:
                        scale = scale.reshape((-1, 1, 1, 1))
                        zero_point_q = zero_point_q.reshape((-1 ,1 ,1 ,1)) if quantizer_type == "Asymmetric" else zero_point_q
                elif kind == quantize.QAnnotateKind.INPUT:
                    act_dict_tmp = act_dict[attrs.name]
                    scale = np.array(act_dict_tmp["scale"], dtype="float32")
                    clip_min = act_dict_tmp["int_min"]
                    clip_max = act_dict_tmp["int_max"]
                    zero_point_q = np.array(act_dict_tmp["zero_point"], dtype="float32")
                elif kind == quantize.QAnnotateKind.BIAS:
                    scale = np.array([1], dtype="float32")
                    clip_min = 1
                    clip_max = 1
                    zero_point_q = np.array(0, dtype="float32")
                else:
                    raise ValueError

                def _make_const(val):
                    return _expr.const(val)
                    
                const_params[ndom_scale] = _make_const(scale)
                const_params[nclip_min] = _make_const(np.array(clip_min, dtype="float32"))
                const_params[nclip_max] = _make_const(np.array(clip_max, dtype="float32"))
                const_params[zero_point] = _make_const(zero_point_q)

    main_func = mod["main"]
    _analysis.post_order_visit(main_func, visit_func)
    main_func = _expr.bind(main_func, const_params)
    func_dict = {}
    for global_var, func in mod.functions.items():
        if global_var.name_hint != "main":
            func_dict[global_var] = func
    return IRModule.from_expr(main_func, func_dict)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.reshape(-1).tolist()
        else:
            return super(NpEncoder, self).default(obj)

def dump_json(act_quantizer_list, weight_quantizer_list, bias_quantizer_list):
    
    cfg = quantize.current_qconfig()
    act_dict = {}
    wgt_dict = {}

    for i in range(len(act_quantizer_list)):
        this_act_quantizer = act_quantizer_list[i]
        act_dict_tmp = {}
        act_dict_tmp["name"] = this_act_quantizer.name
        act_dict_tmp["quantizer"] = "Symmetric" if this_act_quantizer.symmetric else "Asymmetric"
        act_dict_tmp["bit_width"] = this_act_quantizer.bitwidth
        act_dict_tmp["int_max"] = this_act_quantizer.int_max
        act_dict_tmp["int_min"] = this_act_quantizer.int_min
        act_dict_tmp["round_type"] = "Rounding"
        act_dict_tmp["max_value"] = this_act_quantizer.float_max
        act_dict_tmp["min_value"] = this_act_quantizer.float_min
        act_dict_tmp["scale"] = this_act_quantizer.scale
        act_dict_tmp["zero_point"] = this_act_quantizer.zero_point
        act_dict[this_act_quantizer.name] = act_dict_tmp
    
    for i in range(len(weight_quantizer_list)):
        this_weight_quantizer = weight_quantizer_list[i]
        wgt_dict_tmp = {}
        wgt_dict_tmp["name"] = this_weight_quantizer.name
        wgt_dict_tmp["quantizer"] = "Symmetric" if this_act_quantizer.symmetric else "Asymmetric"
        wgt_dict_tmp["bit_width"] = this_weight_quantizer.bitwidth
        wgt_dict_tmp["int_max"] = this_weight_quantizer.int_max
        wgt_dict_tmp["int_min"] = this_weight_quantizer.int_min
        wgt_dict_tmp["round_type"] = "Rounding"
        wgt_dict_tmp["per_channel"] = "True" if this_weight_quantizer.perchannel else "False"
        wgt_dict_tmp["max_value"] = this_weight_quantizer.float_max
        wgt_dict_tmp["min_value"] = this_weight_quantizer.float_min
        wgt_dict_tmp["scale"] = this_weight_quantizer.scale
        wgt_dict_tmp["zero_point"] = this_weight_quantizer.zero_point
        wgt_dict[this_weight_quantizer.name] = wgt_dict_tmp
    
    for i in range(len(bias_quantizer_list)):
        this_bias_quantizer = bias_quantizer_list[i]
        bias_dict_tmp = {}
        bias_dict_tmp["name"] = this_bias_quantizer.name
        bias_dict_tmp["bit_width"] = this_bias_quantizer.bitwidth
        bias_dict_tmp["quantize_method"] = "round"
        wgt_dict[this_bias_quantizer.name] = bias_dict_tmp

    root_dir_name = cfg.get_rootdir_name()
    saved_dir_name = root_dir_name
    if not os.path.exists(root_dir_name):
        os.mkdir(root_dir_name)
    if not os.path.exists(saved_dir_name):
        os.mkdir(saved_dir_name)

    with open(saved_dir_name + "/act_calibrate.json", "w") as f:
        f.write(json.dumps(act_dict, indent=4, cls=NpEncoder))
        f.close()    

    with open(saved_dir_name + "/wgt_calibrate.json", "w") as f:
        f.write(json.dumps(wgt_dict, indent=4, cls=NpEncoder))
        f.close()    


def calibrate(dataset=None):
    """The calibrate procedure will try to calculate the content of
    dom_scale, nbit, clip_min, clip_max and zero_point for every `simulated_quantize`
    operator.

    Parameters
    ---------
    dataset: Optional[Iterable[NDArray]]
        The calibration dataset.

    Returns
    -------
    ret: Function
        The module pass function.
    """
    def wrapped_func(mod, _):
        """make transform.module pass happy"""
        cfg = quantize.current_qconfig()
        debug_mode = cfg.get_debug_mode()
        min_max = 0

        """Step 1. Get the all input data's min max"""
        min_max = collect_min_max(mod, dataset)

        """Step 2. Initialize quantizer for every quantize Node"""
        quantizer_weight_node, estimator_weight_node, quantizer_act_node, estimator_act_node, quantizer_bias_node= \
            initialize_quantizer(mod, min_max)

        """Step 3. Start Quantizing Weight"""
        quantize_weight(mod, quantizer_weight_node, estimator_weight_node)

        """Step 4. Start Quantizing Activation"""
        # q_act_list format is [min_act, max_act]
        activation_info = quantize_activation(mod, quantizer_act_node, estimator_act_node, dataset, min_max)

        """Step 5. Calculating every layers' cosine similarity if needed """
        if debug_mode:
            calculate_precision(mod, quantizer_act_node, dataset)
        
        """Step 6. Dump quantization results into json file"""
        dump_json(quantizer_act_node, quantizer_weight_node, quantizer_bias_node)

        return _set_params(mod, quantizer_weight_node, quantizer_act_node)

    return wrapped_func

class CalibrateFileNotExistedError(Exception):
    def __init__(self):
        super(CalibrateFileNotExistedError, self).__init__("Can't find Calibrate file.")

def read_calibrate():
    
    def wrapped_func(mod, _):
        cfg = quantize.current_qconfig()

        print("Into calibration...")

        """Step 1. Read Json files"""
        saved_dir_name = cfg.get_rootdir_name()
        act_file = saved_dir_name + "/act_calibrate.json"
        wgt_file = saved_dir_name + "/wgt_calibrate.json"
        
        if not os.path.exists(act_file):
            raise CalibrateFileNotExistedError()
        
        if not os.path.exists(wgt_file):
            raise CalibrateFileNotExistedError()
        
        f_act = open(act_file, "r")
        f_wgt = open(wgt_file, "r")

        act_dict = json.load(f_act)
        wgt_dict = json.load(f_wgt)

        f_act.close()
        f_wgt.close()

        return _set_params_rc(mod, act_dict, wgt_dict)

    return wrapped_func