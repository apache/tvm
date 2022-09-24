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
# pylint: disable=unused-argument,inconsistent-return-statements
"""Internal module for registering attribute for annotation."""
import warnings
from tvm import topi
import tvm._ffi
from tvm.relay.op import op as _reg
from .. import expr as _expr
from .. import analysis as _analysis
from .. import op as _op
from . import _quantize
from .quantize import QAnnotateKind, current_qconfig, quantize_context
from .quantize import _forward_op


@_op.register_compute("relay.op.annotation.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 5
    assert attrs.rounding == "round"

    data, scale, clip_min, clip_max, zero_point = inputs

    if attrs.kind == QAnnotateKind.IDENTITY:
        return [topi.identity(data)]
    
    if attrs.kind == QAnnotateKind.BIAS:
        return [topi.identity(data)]
    
    if attrs.kind == QAnnotateKind.ACTIVATION:
        return [topi.identity(data)]

    # simulate rounding error
    scaled_data = topi.divide(data, scale)
    round_data = topi.add(topi.round(scaled_data), zero_point) 
    clipped_data = topi.maximum(topi.minimum(round_data, clip_max), clip_min)

    # recover data
    rdata = topi.multiply(topi.subtract(clipped_data, zero_point), scale)
    return [rdata]


_reg.register_injective_schedule("relay.op.annotation.simulated_quantize")
_reg.register_pattern("relay.op.annotation.simulated_quantize", _reg.OpPattern.ELEMWISE)
_reg.register_injective_schedule("annotation.cast_hint")


@tvm._ffi.register_object("relay.QAnnotateExpr")
class QAnnotateExpr(_expr.TempExpr):
    """A special kind of Expr for Annotating.

    Parameters
    ---------
    expr: Expr
        the original relay ir expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """

    def __init__(self, expr, kind):
        self.__init_handle_by_constructor__(_quantize.make_annotate_expr, expr, kind)


def _get_expr_kind(anno):
    """Get the expression and QAnnotateKind from QAnnotateExpr or Expr"""
    if isinstance(anno, QAnnotateExpr):
        return anno.expr, anno.kind
    return anno, None


def register_annotate_inference_function(op_name, frewrite=None, level=10):
    """register a rewrite function for operator, used by annotation.

    Parameters
    ---------
    op_name: str
        The name of operation

    frewrite : function, optional
        The function to be registered.

    level : int, optional
        The priority level
    """

    def default_rewrite(ref_call, new_args, ctx):
        # recover from QAnnotateExpr
        args = [_get_expr_kind(x)[0] for x in new_args]
        return _forward_op(ref_call, args)

    def _register(func):
        """internal register function"""

        def frewrite_with_guard(ref_call, new_args, ctx):
            if not current_qconfig().guard(ref_call):
                return default_rewrite(ref_call, new_args, ctx)
            return func(ref_call, new_args, ctx)

        return tvm.ir.register_op_attr(op_name, "FQAnnotateRewrite", frewrite_with_guard, level)

    return _register(frewrite) if frewrite is not None else _register

def register_annotate_calibrate_function(op_name, frewrite=None, level=10):
    """register a rewrite function for operator, used by annotation.

    Parameters
    ---------
    op_name: str
        The name of operation

    frewrite : function, optional
        The function to be registered.

    level : int, optional
        The priority level
    """

    def default_rewrite(ref_call, new_args, ctx):
        # recover from QAnnotateExpr
        args = [_get_expr_kind(x)[0] for x in new_args]
        return _forward_op(ref_call, args)

    def _register(func):
        """internal register function"""

        def frewrite_with_guard(ref_call, new_args, ctx):
            if not current_qconfig().guard(ref_call):
                return default_rewrite(ref_call, new_args, ctx)
            return func(ref_call, new_args, ctx)

        return tvm.ir.register_op_attr(op_name, "FQAnnotateForCalibrateRewrite", frewrite_with_guard, level)

    return _register(frewrite) if frewrite is not None else _register

def attach_simulated_quantize(data, kind, name="qnode", can_per_channel = False, rounding="round"):
    """Attach a simulated quantize operation after input data expr.

    Parameters
    ---------
    data: Expr
        the original data expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """
    per_channel = current_qconfig().get_perChannel_by_kind(kind) if can_per_channel else False
    asymmetric = False
    quantizer_type = current_qconfig().get_quantizer_by_kind(kind)
    if(quantizer_type == "Asymmetric"):
        asymmetric = True

    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    if isinstance(data, _expr.Call) and data.op == quantize_op:
        if data.attrs.kind == kind and data.attrs.rounding == rounding and \
            data.attrs.per_channel == per_channel and data.attrs.asymmetric == asymmetric:
            return data

    qctx = quantize_context()
    key = tuple([data, kind, rounding, per_channel, asymmetric, name])
    if key in qctx.qnode_map:
        return qctx.qnode_map[key]

    dom_scale = _expr.var("dom_scale")
    clip_min = _expr.var("clip_min")
    clip_max = _expr.var("clip_max")
    zero_point = _expr.var("zero_point")
    qnode = _quantize.simulated_quantize(data, dom_scale, clip_min, clip_max, zero_point, kind, rounding, per_channel, asymmetric, name)
    qctx.qnode_map[key] = qnode
    return qnode


tvm._ffi.register_func("relay.quantize.attach_simulated_quantize", attach_simulated_quantize)

################################ Register inference annotate function #####################################

inference_annotate_dict = {}
inference_layer_count = 0
expr_count_map = {}
input_map = {}

def get_layer_name(expr):
    global input_map
    global expr_count_map
    stop_fusion_op = _op.get("annotation.stop_fusion")
    cast_op = _op.get("annotation.cast_hint")
    if(isinstance(expr, _expr.Call)):
        if(expr.op == stop_fusion_op or expr.op == cast_op):
            raise ValueError
        else:
            qnode_name = expr.op.name + "_" + str(expr_count_map[expr]) + ":out"
            return qnode_name
    elif(isinstance(expr, _expr.TupleGetItem)):
        expr_tmp = expr.tuple_value
        item_idx = expr.index
        qnode_name = "split" + "_" + str(expr_count_map[expr_tmp]) + ":out_" + str(item_idx)
        return qnode_name
    elif(isinstance(expr, _expr.Var)):
        if expr not in input_map:
            qnode_name = "network_input_" + str(inference_layer_count)
            input_map[expr] = qnode_name
        else:
            qnode_name = input_map[expr]
        return qnode_name
    else:
        raise ValueError

def register_annotate_inference_function_dict(op_name, frewrite):
    inference_annotate_dict[op_name] = frewrite

def conv2d_nchwc_rewrite(ref_call, new_args, ctx):
    warnings.warn(
        "NCHWc layout Conv2D detected, please use a lower "
        "optimization level before applying the quantization "
        "pass as quantization will have no effect here..."
    )
register_annotate_inference_function_dict("nn.contrib_conv2d_NCHWc", conv2d_nchwc_rewrite)

def conv2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(lhs_expr)
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)

    assert rhs_kind is None
    qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, qnode_name, True)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
register_annotate_inference_function_dict("nn.conv2d", conv2d_rewrite)

def conv1d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv1d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(lhs_expr)
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)

    assert rhs_kind is None
    qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, qnode_name, True)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
register_annotate_inference_function_dict("nn.conv1d", conv1d_rewrite)

def dense_rewrite(ref_call, new_args, ctx):
    """Rewrite function for dense. Lhs of dense will be quantized to input field, and rhs of
    dense will be quantized to weight field. Output would be in activation field."""

    if current_qconfig().skip_dense_layer:
        return None

    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(lhs_expr)
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)

    assert rhs_kind is None
    qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, qnode_name)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
register_annotate_inference_function_dict("nn.dense", dense_rewrite)


def multiply_rewrite(ref_call, new_args, ctx):
    """Rewrite function for multiply."""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None

    if lhs_kind in [QAnnotateKind.ACTIVATION, QAnnotateKind.INPUT] and rhs_kind is None:
        # quantize lhs to INPUT field
        if lhs_kind == QAnnotateKind.ACTIVATION:
            qnode_name = get_layer_name(lhs_expr)
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)
        if _analysis.check_constant(rhs_expr):
            qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, qnode_name)
        else:
            qnode_name = get_layer_name(lhs_expr)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT, qnode_name)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    
    if rhs_kind in [QAnnotateKind.ACTIVATION, QAnnotateKind.INPUT] and lhs_kind is None:
        # quantize lhs to INPUT field
        if rhs_kind == QAnnotateKind.ACTIVATION:
            qnode_name = get_layer_name(rhs_expr)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT, qnode_name)
        if _analysis.check_constant(lhs_expr):
            qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.WEIGHT, qnode_name)
        else:
            qnode_name = get_layer_name(lhs_expr)
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    raise ValueError
register_annotate_inference_function_dict("multiply", multiply_rewrite)


def add_rewrite(ref_call, new_args, ctx):
    """Rewrite function for add."""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        # trivial case
        return None

    if lhs_kind is None and rhs_kind is not None:
        # quantize lhs to INPUT field if it is normal expression
        assert rhs_kind in [QAnnotateKind.INPUT, QAnnotateKind.ACTIVATION]
        qnode_name = get_layer_name(lhs_expr)
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.INPUT)

    if lhs_kind is not None and rhs_kind is None:
        if _analysis.check_constant(rhs_expr):
            # - introduced by batch_norm: add(out, const)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.BIAS, "bias")
            # zzk_debug: for add bias, we don't need to quantize to int8
            pass
        else:
            qnode_name = get_layer_name(rhs_expr)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT, qnode_name)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is not None:
        if lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.INPUT:
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.INPUT)
        if lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.ACTIVATION:
            qnode_name = get_layer_name(rhs_expr)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT, qnode_name)
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
        if (lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.INPUT) or (
            lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.ACTIVATION
        ):
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    raise ValueError()
register_annotate_inference_function_dict("add", add_rewrite)


def identity_rewrite(ref_call, new_args, ctx):
    """Simply forward the original operation"""
    if quantize_context().check_to_skip(ref_call):
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = _forward_op(ref_call, [x_expr])
    return QAnnotateExpr(ret_expr, x_kind)


register_annotate_inference_function_dict("reshape", identity_rewrite)
register_annotate_inference_function_dict("clip", identity_rewrite)
register_annotate_inference_function_dict("nn.relu", identity_rewrite)
register_annotate_inference_function_dict("strided_slice", identity_rewrite)
register_annotate_inference_function_dict("nn.avg_pool2d", identity_rewrite)
register_annotate_inference_function_dict("nn.batch_flatten", identity_rewrite)
register_annotate_inference_function_dict("transpose", identity_rewrite)
register_annotate_inference_function_dict("annotation.stop_fusion", identity_rewrite)


def pool2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool2d"""
    if quantize_context().check_to_skip(ref_call):
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(expr)
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT, qnode_name)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_inference_function_dict("nn.max_pool2d", pool2d_rewrite)


def pool1d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool1d"""
    if quantize_context().check_to_skip(ref_call):
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(expr)
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT, qnode_name)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_inference_function_dict("nn.max_pool1d", pool1d_rewrite)


def cast_hint_rewrite(ref_call, new_args, ctx):
    """Rewrite function to force cast"""
    expr, x_kind = _get_expr_kind(new_args[0])

    if quantize_context().check_to_skip(ref_call):
        return expr

    if x_kind is None:
        return new_args[0]
    if x_kind == QAnnotateKind.ACTIVATION:
        qnode_name = get_layer_name(expr)
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT, qnode_name)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)
register_annotate_inference_function_dict("annotation.cast_hint", cast_hint_rewrite)


def concatenate_rewrite(ref_call, new_args, ctx):
    """Rewrite function for concatenate"""
    if quantize_context().check_to_skip(ref_call):
        return None

    input_tuple = new_args[0]
    expr_list = [_get_expr_kind(x)[0] for x in input_tuple]
    kind_list = [_get_expr_kind(x)[1] for x in input_tuple]

    # make sure the inputs of concatenate are all normal
    # expression or annotate expression
    if all([k is None for k in kind_list]):
        return None
    #zzk_debug: Will this cause some problems?
    for i, k in enumerate(kind_list):
        if k is None:
            expr_list[i] = attach_simulated_quantize(expr_list[i], QAnnotateKind.ACTIVATION, "qactivation")
    expr = _forward_op(ref_call, [_expr.Tuple(expr_list)])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
register_annotate_inference_function_dict("concatenate", concatenate_rewrite)


def global_avg_pool2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for global_avg_pool2d for stopping quantize"""
    if quantize_context().check_to_skip(ref_call):
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    expr = _forward_op(ref_call, [new_args[0].realize()])

    # stop quantize after global_avg_pool2d
    quantize_context().stop_quantize()
    return expr
register_annotate_inference_function_dict("nn.global_avg_pool2d", global_avg_pool2d_rewrite)


def batch_matmul_rewrite(ref_call, new_args, ctx):
    """Rewrite function for batch_matmul"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(lhs_expr):
            qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.WEIGHT, qnode_name)
        else:
            qnode_name = get_layer_name(lhs_expr)
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT, qnode_name)

    if rhs_kind is None or rhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(rhs_expr):
            qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(inference_layer_count) + "_0" + ":in"
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT, qnode_name)
        else:
            qnode_name = get_layer_name(rhs_expr)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT, qnode_name)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
register_annotate_inference_function_dict("nn.batch_matmul", batch_matmul_rewrite)

################################## Inference   annotate ##################################


def inference_rewrite(ref_call, new_args, ctx):
    """Rewrite function for every call node"""
    global inference_layer_count
    global inference_annotate_dict
    global expr_count_map

    stop_fusion_op = _op.get("annotation.stop_fusion")
    cast_op = _op.get("annotation.cast_hint")

    forward_args = []
    for arg in new_args:
        forward_args.append(_get_expr_kind(arg)[0])

    if ref_call.op.name not in inference_annotate_dict:
        expr_out = _forward_op(ref_call, forward_args)
    else:
        expr_out = inference_annotate_dict[ref_call.op.name](ref_call, new_args, ctx)

    if ref_call.op == stop_fusion_op or ref_call.op == cast_op:
        pass
    else:
        if expr_out is None:
            expr_out = _forward_op(ref_call, forward_args)
            expr_count_map[expr_out] = inference_layer_count
        else:
            expr_count_map[_get_expr_kind(expr_out)[0]] = inference_layer_count

        inference_layer_count = inference_layer_count + 1

    return expr_out
        
have_registered_op_list_inference = []

def annotate_for_inference_registry(mod):
    main_func = mod["main"]
    global inference_layer_count
    global expr_count_map
    global input_map
    global have_registered_op_list_inference

    inference_layer_count = 0
    expr_count_map = {}
    input_map = {}

    if "annotation.stop_fusion" not in have_registered_op_list_inference:
        register_annotate_inference_function("annotation.stop_fusion", inference_rewrite)
        have_registered_op_list_inference.append("annotation.stop_fusion")
    
    if "annotation.cast_hint" not in have_registered_op_list_inference:
        register_annotate_inference_function("annotation.cast_hint", inference_rewrite)
        have_registered_op_list_inference.append("annotation.cast_hint")

    def register_op_for_annotate(expr):
        if isinstance(expr, _expr.Call):
            if expr.op.name not in have_registered_op_list_inference:
                register_annotate_inference_function(expr.op.name, inference_rewrite)
                have_registered_op_list_inference.append(expr.op.name)

    _analysis.post_order_visit(main_func, register_op_for_annotate)

################################## Calibration annotate ##################################

layer_count = 0
have_annotated_node = [] # prevent situation like input tensor be used by multiple node
                         # which will cause this multiple quantization to the same node
                         # only document special tensor like input and split tensor
split_op_dict = {} # {expr: name}
input_dict = {} # {input: name}

def calibrate_rewrite(ref_call, new_args, ctx):
    """Rewrite function for every call node"""
    global layer_count
    global have_annotated_node
    global split_op_dict
    global input_dict

    weight_count = 0
    
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    add_op = _op.get("add")
    split_op = _op.get("split")
    conv2d_op = _op.get("nn.conv2d")
    conv1d_op = _op.get("nn.conv1d")
    pad_op = _op.get("nn.pad")
    
    # get the lenth of argument
    args_out = []
    for arg in new_args:
        if(isinstance(arg, _expr.Call)):
            if(arg.op != quantize_op):
                print("Something went wrong, this op is {}, last op is {}.".format(ref_call.op.name, arg.op.name))
                raise ValueError
            else:
                args_out.append(arg)
        elif(isinstance(arg, _expr.Constant)):
            if(ref_call.op == add_op):
                qnode_name = ref_call.op.name + "_" + "bias" + "_" + str(layer_count) + "_" + str(weight_count) + ":in"
                new_arg = attach_simulated_quantize(arg, QAnnotateKind.BIAS, qnode_name)
            elif(ref_call.op == pad_op): #pad
                new_arg = arg
            else:
                qnode_name = ref_call.op.name + "_" + "weight" + "_" + str(layer_count) + "_" + str(weight_count) + ":in"
                if(ref_call.op == conv1d_op or ref_call.op == conv2d_op):
                    new_arg = attach_simulated_quantize(arg, QAnnotateKind.WEIGHT, qnode_name, True)
                else:
                    new_arg = attach_simulated_quantize(arg, QAnnotateKind.WEIGHT, qnode_name)
            args_out.append(new_arg)
            weight_count = weight_count + 1
        elif(isinstance(arg, _expr.TupleGetItem)):
            expr_tmp = arg.tuple_value
            item_idx = arg.index
            qnode_name = split_op_dict[expr_tmp] + "_" + str(item_idx)
            if qnode_name not in have_annotated_node:
                new_arg = attach_simulated_quantize(arg, QAnnotateKind.INPUT, qnode_name)
                args_out.append(new_arg)
                have_annotated_node.append(qnode_name)
            else:
                args_out.append(arg)
        elif(isinstance(arg, _expr.Var)):
            # add input expr
            qnode_name = "network_input_" + str(layer_count)
            if arg not in input_dict:
                input_dict[arg] = qnode_name
            else:
                qnode_name = input_dict[arg]
            if qnode_name not in have_annotated_node:
                new_arg = attach_simulated_quantize(arg, QAnnotateKind.INPUT, qnode_name)
                args_out.append(new_arg)
                have_annotated_node.append(qnode_name)
            else:
                args_out.append(arg)
        elif(isinstance(arg, _expr.Tuple)):
            args_out.append(arg)
        else:
            print("Something is wrong.")
            raise ValueError

    expr = _forward_op(ref_call, args_out)

    if ref_call.op != split_op:
        expr_out = attach_simulated_quantize(expr, QAnnotateKind.INPUT, ref_call.op.name + "_" + str(layer_count) + ":out")
    else:
        split_op_dict[expr] = ref_call.op.name + "_" + str(layer_count) + ":out"
        expr_out = expr

    layer_count = layer_count + 1

    return expr_out

have_registered_op_list_calibrate = []
def annotate_for_calibrate_registry(mod):
    main_func = mod["main"]
    global layer_count
    global have_annotated_node
    global split_op_dict
    global input_dict
    global have_registered_op_list_calibrate

    layer_count = 0
    have_annotated_node = []
    split_op_dict = {}
    input_dict = {}

    def register_op_for_annotate(expr):
        if isinstance(expr, _expr.Call):
            if expr.op.name not in have_registered_op_list_calibrate:
                register_annotate_calibrate_function(expr.op.name, calibrate_rewrite)
                have_registered_op_list_calibrate.append(expr.op.name)

    _analysis.post_order_visit(main_func, register_op_for_annotate)