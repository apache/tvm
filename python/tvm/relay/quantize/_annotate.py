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
    assert len(inputs) == 4
    assert attrs.sign
    assert attrs.rounding == "round"

    data, scale, clip_min, clip_max = inputs

    if attrs.kind == QAnnotateKind.IDENTITY:
        return [topi.identity(data)]

    # simulate rounding error
    scaled_data = topi.divide(data, scale)
    clipped_data = topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    round_data = topi.round(clipped_data)

    # recover data
    rdata = topi.multiply(round_data, scale)
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

        return tvm.ir.register_op_attr(op_name, "FQAnnotateRewrite", frewrite_with_guard, level)

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

    qctx = quantize_context()
    key = tuple([data, kind, sign, rounding])
    if key in qctx.qnode_map:
        return qctx.qnode_map[key]

    dom_scale = _expr.var("dom_scale")
    clip_min = _expr.var("clip_min")
    clip_max = _expr.var("clip_max")
    qnode = _quantize.simulated_quantize(data, dom_scale, clip_min, clip_max, kind, sign, rounding)
    qctx.qnode_map[key] = qnode
    return qnode


tvm._ffi.register_func("relay.quantize.attach_simulated_quantize", attach_simulated_quantize)


@register_annotate_function("nn.contrib_conv2d_NCHWc")
def conv2d_nchwc_rewrite(ref_call, new_args, ctx):
    warnings.warn(
        "NCHWc layout Conv2D detected, please use a lower "
        "optimization level before applying the quantization "
        "pass as quantization will have no effect here..."
    )


@register_annotate_function("nn.conv2d")
def conv2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv2d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("nn.conv1d")
def conv1d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for conv1d. Lhs of conv will be quantized to
    input field, and rhs of conv will be quantized to weight field.
    Output would be in activation field"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("nn.dense")
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
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    assert rhs_kind is None
    rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])

    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("multiply")
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
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        if _analysis.check_constant(rhs_expr):
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    if rhs_kind in [QAnnotateKind.ACTIVATION, QAnnotateKind.INPUT] and lhs_kind is None:
        # quantize rhs to INPUT field
        if rhs_kind == QAnnotateKind.ACTIVATION:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
        if _analysis.check_constant(lhs_expr):
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.WEIGHT)
        else:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    raise ValueError


@register_annotate_function("add")
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
        lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is None:
        if _analysis.check_constant(rhs_expr):
            # - introduced by batch_norm: add(out, const)
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
        expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
        return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)

    if lhs_kind is not None and rhs_kind is not None:
        if lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.INPUT:
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
        if lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.ACTIVATION:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
        if (lhs_kind == QAnnotateKind.ACTIVATION and rhs_kind == QAnnotateKind.INPUT) or (
            lhs_kind == QAnnotateKind.INPUT and rhs_kind == QAnnotateKind.ACTIVATION
        ):
            expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
            return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
    raise ValueError()


def identity_rewrite(ref_call, new_args, ctx):
    """Simply forward the original operation"""
    if quantize_context().check_to_skip(ref_call):
        return None

    x_expr, x_kind = _get_expr_kind(new_args[0])
    if x_kind is None:
        return None

    ret_expr = _forward_op(ref_call, [x_expr])
    return QAnnotateExpr(ret_expr, x_kind)


register_annotate_function("reshape", identity_rewrite)
register_annotate_function("clip", identity_rewrite)
register_annotate_function("nn.relu", identity_rewrite)
register_annotate_function("strided_slice", identity_rewrite)
register_annotate_function("nn.avg_pool2d", identity_rewrite)
register_annotate_function("nn.batch_flatten", identity_rewrite)
register_annotate_function("transpose", identity_rewrite)
register_annotate_function("annotation.stop_fusion", identity_rewrite)


def pool2d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool2d"""
    if quantize_context().check_to_skip(ref_call):
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_function("nn.max_pool2d", pool2d_rewrite)


def pool1d_rewrite(ref_call, new_args, ctx):
    """Rewrite function for max pool1d"""
    if quantize_context().check_to_skip(ref_call):
        return None

    expr, x_kind = _get_expr_kind(new_args[0])

    if x_kind is None:
        return None
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


register_annotate_function("nn.max_pool1d", pool1d_rewrite)


@register_annotate_function("annotation.cast_hint")
def cast_hint_rewrite(ref_call, new_args, ctx):
    """Rewrite function to force cast"""
    expr, x_kind = _get_expr_kind(new_args[0])

    if quantize_context().check_to_skip(ref_call):
        return expr

    if x_kind is None:
        return new_args[0]
    if x_kind == QAnnotateKind.ACTIVATION:
        expr = attach_simulated_quantize(expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [expr])
    return QAnnotateExpr(expr, QAnnotateKind.INPUT)


@register_annotate_function("concatenate")
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
    for i, k in enumerate(kind_list):
        if k is None:
            expr_list[i] = attach_simulated_quantize(expr_list[i], QAnnotateKind.ACTIVATION)
    expr = _forward_op(ref_call, [_expr.Tuple(expr_list)])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)


@register_annotate_function("nn.global_avg_pool2d")
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


@register_annotate_function("nn.batch_matmul")
def batch_matmul_rewrite(ref_call, new_args, ctx):
    """Rewrite function for batch_matmul"""
    if quantize_context().check_to_skip(ref_call):
        return None

    lhs_expr, lhs_kind = _get_expr_kind(new_args[0])
    rhs_expr, rhs_kind = _get_expr_kind(new_args[1])

    if lhs_kind is None or lhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(lhs_expr):
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.WEIGHT)
        else:
            lhs_expr = attach_simulated_quantize(lhs_expr, QAnnotateKind.INPUT)

    if rhs_kind is None or rhs_kind == QAnnotateKind.ACTIVATION:
        if _analysis.check_constant(rhs_expr):
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.WEIGHT)
        else:
            rhs_expr = attach_simulated_quantize(rhs_expr, QAnnotateKind.INPUT)

    expr = _forward_op(ref_call, [lhs_expr, rhs_expr])
    return QAnnotateExpr(expr, QAnnotateKind.ACTIVATION)
