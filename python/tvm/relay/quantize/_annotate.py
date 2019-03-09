#pylint: disable=unused-argument
"""Internal module for registering attribute for annotation."""
from __future__ import absolute_import
import warnings

import topi
from . import _quantize
from .quantize import QAnnotateKind, current_qconfig
from .quantize import _conv_counter, _set_conv_counter
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


@register_func("relay.quantize.attach_simulated_quantize")
def attach_simulated_quantize(data, kind, sign=True, rounding="round"):
    """Attach a simulated quantize operation after input data expr.

    Parameters
    ---------
    data: Expr
        the original data expr.

    kind: QAnnotateKind
        the kind of annotation field.
    """
    dom_scale = _expr.var("dom_scale")
    clip_min = _expr.var("clip_min")
    clip_max = _expr.var("clip_max")
    return _quantize.simulated_quantize(
        data, dom_scale, clip_min, clip_max, kind, sign, rounding)


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
    cnt = _conv_counter()
    if cnt < current_qconfig().skip_k_conv:
        _set_conv_counter(cnt + 1)
        return None
    _set_conv_counter(cnt + 1)

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind != QAnnotateKind.INPUT:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("multiply")
def multiply_rewrite(ref_call, new_args, ctx):
    """Rewrite function for multiply."""
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None
    if lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind is None:
        # quantize lhs to INPUT field
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        # quantize rhs to WEIGHT field
        rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    raise ValueError


@register_annotate_function("add")
def add_rewrite(ref_call, new_args, ctx):
    """Rewrite function for add."""
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None and rhs_kind is None:
        return None
    if lhs_kind is None and rhs_kind is not None:
        # quantize lhs to INPUT field if it is normal expression
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
    if lhs_kind is not None and rhs_kind is None:
        if isinstance(rhs_expr, _expr.Constant):
            # quantize rhs to WEIGHT field if it is Constant
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            # quantize rhs to INPUT field if it is not Constant
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


def identity_rewrite(ref_call, new_args, ctx):
    """Simply forward the original operation"""
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = _forward_op(ref_call, [x_expr])
    return QAnnotateExpr(ret_expr, x_kind)


register_annotate_function("nn.relu", identity_rewrite)
register_annotate_function("strided_slice", identity_rewrite)
register_annotate_function("nn.avg_pool2d", identity_rewrite)


def pool2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool2d"""
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None
    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)
    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_function("nn.max_pool2d", pool2d_rewrite)


@register_annotate_function("concatenate")
def concatenate_rewrite(ref_call, new_args, ctx):
    """Rewrite function for concatenate"""
    if _conv_counter() <= current_qconfig().skip_k_conv:
        return None

    input_tuple = new_args[0]
    expr_list = [_get_expr_kind(x)[0] for x in input_tuple]
    kind_list = [_get_expr_kind(x)[1] for x in input_tuple]

    # make sure the inputs of concatenate are all normal
    # expression or annotate expression
    if kind_list[0] is None:
        for k in kind_list:
            assert k is None
        return None
    for k in kind_list:
        assert k is not None
    expr = _forward_op(ref_call, [_expr.Tuple(expr_list)])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
