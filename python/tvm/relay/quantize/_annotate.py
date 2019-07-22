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
#pylint: disable=unused-argument,inconsistent-return-statements
"""Internal module for registering attribute for annotation."""
from __future__ import absolute_import
import warnings

import topi
from . import _quantize
from .quantize import QAnnotateKind, current_qconfig
from .quantize import annotate_context
from .. import expr as _expr
from .. import op as _op
from ..op import op as _reg
from ..base import register_relay_node
from ..._ffi.function import register_func


@_reg.register_compute("relay.op.annotation.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 4
    assert attrs.sign
    assert attrs.rounding == "round"

    data, scale, clip_min, clip_max = inputs

    # simulate rounding error
    scaled_data = topi.divide(data, scale)
    clipped_data = topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    round_data = topi.round(clipped_data)

    # recover data
    rdata = topi.multiply(round_data, scale)
    return [rdata]


_reg.register_schedule("relay.op.annotation.simulated_quantize",
                       _reg.schedule_injective)
_reg.register_pattern("relay.op.annotation.simulated_quantize",
                      _reg.OpPattern.OPAQUE)


@register_relay_node
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
        self.__init_handle_by_constructor__(
            _quantize.make_annotate_expr, expr, kind)


def _forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return _expr.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


def _get_expr_kind(anno):
    """Get the expression and QAnnotateKind from QAnnotateExpr or Expr"""
    if isinstance(anno, QAnnotateExpr):
        return anno.expr, anno.kind
    return anno, None


def register_annotate_function(op_name, frewrite=None, level=10):
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
        _op.op._Register(op_name, "FQAnnotateRewrite", frewrite_with_guard, level)
        return frewrite_with_guard

    return _register(frewrite) if frewrite is not None else _register


def attach_simulated_quantize(data, kind, sign=True, rounding="round"):
    """Attach a simulated quantize operation after input data expr.

    Parameters
    ---------
    data: Expr
        the original data expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """
    quantize_op = _op.get("relay.op.annotation.simulated_quantize")
    if isinstance(data, _expr.Call) and data.op == quantize_op:
        if data.attrs.kind == kind and data.attrs.sign == sign and data.attrs.rounding == rounding:
            return data

    actx = annotate_context()
    key = tuple([data, kind, sign, rounding])
    if key in actx.qnode_map:
        return actx.qnode_map[key]

    dom_scale = _expr.var("dom_scale")
    clip_min = _expr.var("clip_min")
    clip_max = _expr.var("clip_max")
    qnode = _quantize.simulated_quantize(
        data, dom_scale, clip_min, clip_max, kind, sign, rounding)
    actx.qnode_map[key] = qnode
    return qnode

register_func("relay.quantize.attach_simulated_quantize", attach_simulated_quantize)


@register_annotate_function("nn.contrib_conv2d_NCHWc")
def conv2d_nchwc_rewrite(ref_call, new_args, ctx):
    warnings.warn("NCHWc layout Conv2D detected, please use a lower "
                  "optimization level before applying the quantization "
                  "pass as quantization will have no effect here...")


@register_annotate_function("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    actx = annotate_context()
    if current_qconfig().skip_conv_layers is not None:
        skipped_indices = [int(x) for x in current_qconfig().skip_conv_layers]
        if actx.conv2d_counter() in skipped_indices:
            actx.count_conv2d()
            return None
    actx.count_conv2d()

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


def check_to_skip():
    """Check the index of conv2d layer to decide whether to skip the current operator."""
    if current_qconfig().skip_conv_layers is not None:
        skipped_indices = [int(x) for x in current_qconfig().skip_conv_layers]
        if annotate_context().conv2d_counter() - 1 in skipped_indices:
            return True
    return False


# TODO(tmoreau89,ziheng) need to include an option to turn off dense quant
# @register_annotate_function("nn.dense")
def dense_rewrite(ref_call, new_args, ctx):
    """Rewrite function for dense. Lhs of dense will be quantized to input field, and rhs of
    dense will be quantized to weight field. Output would be in activation field."""
    if check_to_skip():
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("multiply")
def multiply_rewrite(ref_call, new_args, ctx):
    """Rewrite function for multiply."""
    if check_to_skip():
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None

    if lhs_kind in [QAnnotateKind.ACTIVATION, QAnnotateKind.INPUT] and rhs_kind is None:
        # quantize lhs to INPUT field
        if lhs_kind == QAnnotateKind.ACTIVATION:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        # quantize rhs to WEIGHT field
        rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    raise ValueError


@register_annotate_function("add")
def add_rewrite(ref_call, new_args, ctx):
    """Rewrite function for add."""
    if check_to_skip():
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None

    if lhs_kind is None and rhs_kind is not None:
        # quantize lhs to INPUT field if it is normal expression
        assert rhs_kind == QAnnotateKind.INPUT
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.INPUT)

    if lhs_kind is not None and rhs_kind is None:
        if isinstance(rhs_expr, _expr.Constant):
            # quantize rhs to WEIGHT field if it is Constant
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            # quantize rhs to INPUT field if it is not Constant
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is not None:
        if lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.INPUT:
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.INPUT)
        if lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.ACTIVATION:
            # quantize rhs to INPUT field if both lhs and rhs are ACTIVATION
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
        if lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.INPUT:
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    raise ValueError()


@register_annotate_function("stop_fusion")
def stop_fusion_rewrite(ref_call, new_args, ctx):
    """Rewrite function for add."""
    if check_to_skip():
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = attach_simulated_quantize(x_expr, QAnnotateKind.INPUT)
    ret_expr = _forward_op(ref_call, [ret_expr])
    return QAnnotateExpr(ret_expr, QAnnotateKind.INPUT)


def identity_rewrite(ref_call, new_args, ctx):
    """Simply forward the original operation"""
    if check_to_skip():
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = _forward_op(ref_call, [x_expr])
    return QAnnotateExpr(ret_expr, x_kind)


register_annotate_function("clip", identity_rewrite)
register_annotate_function("nn.relu", identity_rewrite)
register_annotate_function("strided_slice", identity_rewrite)
register_annotate_function("nn.avg_pool2d", identity_rewrite)
register_annotate_function("annotation.stop_fusion", identity_rewrite)


def pool2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool2d"""
    if check_to_skip():
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_function("nn.max_pool2d", pool2d_rewrite)


@register_annotate_function("annotation.force_cast")
def force_cast_rewrite(ref_call, new_args, ctx):
    """Rewrite function to force cast"""
    if check_to_skip():
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return new_args[0]
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


@register_annotate_function("concatenate")
def concatenate_rewrite(ref_call, new_args, ctx):
    """Rewrite function for concatenate"""
    if check_to_skip():
        return None

    input_tuple = new_args[0]
    expr_list = [_get_expr_kind(x)[0] for x in input_tuple]
    kind_list = [_get_expr_kind(x)[1] for x in input_tuple]

    # make sure the inputs of concatenate are all normal
    # expression or annotate expression
    if all([k is None for k in kind_list]):
        return None
    for i, k in enumerate(kind_list):
        if k is None:
            expr_list[i] = attach_simulated_quantize(expr_list[i], QAnnotateKind.ACTIVATION)
    expr = _forward_op(ref_call, [_expr.Tuple(expr_list)])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


# Graph rewrite function registration for VTA target
def register_vta_rewrite(op_name, frewrite=None, level=10):
    def _register(func):
        return _op.op._Register(op_name, "FQVTARewrite", func, level)
    return _register(frewrite) if frewrite is not None else _register


@register_relay_node
class QVTAExpr(_expr.TempExpr):
    def __init__(self, expr):
        self.__init_handle_by_constructor__(
            _quantize.make_vta_expr, expr)

    def realize(self):
        return _quantize.temp_expr_realize(self)


def vta_expr_check(expr):
    if isinstance(expr, QVTAExpr):
        return True, expr.expr
    return False, expr


@register_vta_rewrite("nn.conv2d")
def conv2d_vta_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d for VTA target"""
    actx = annotate_context()
    if current_qconfig().skip_conv_layers is not None:
        skipped_indices = [int(x) for x in current_qconfig().skip_conv_layers]
        if actx.conv2d_counter() in skipped_indices:
            actx.count_conv2d()
            return None
    actx.count_conv2d()

    data_cond, data = vta_expr_check(new_args[0])
    kernel_cond, kernel = vta_expr_check(new_args[1])

    assert not kernel_cond
    if data_cond:
        data = new_args[0].realize()
    ret = _forward_op(ref_call, [data, kernel])
    return QVTAExpr(ret)


def identity_vta_rewrite(ref_call, new_args, ctx):
    cond, expr = vta_expr_check(new_args[0])
    if cond:
        return QVTAExpr(_forward_op(ref_call, [expr]))
    return None

register_vta_rewrite("nn.relu", identity_vta_rewrite)
register_vta_rewrite("nn.max_pool2d", identity_vta_rewrite)


@register_vta_rewrite("add")
def add_vta_rewrite(ref_call, new_args, ctx):
    """Rewrite function for ewise add for VTA target"""
    lhs_cond, lhs = vta_expr_check(new_args[0])
    rhs_cond, rhs = vta_expr_check(new_args[1])
    if lhs_cond and rhs_cond:
        lhs = new_args[0].realize()
        rhs = new_args[1].realize()
        return _forward_op(ref_call, [lhs, rhs])
    elif lhs_cond and not rhs_cond:
        return QVTAExpr(_forward_op(ref_call, [lhs, rhs]))
    return None
