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
import tvm
from .. import expr as _expr
from .. import analysis as _analysis
from . import _quantize
from .quantize import _forward_op


def register_partition_function(op_name, frewrite=None, level=10):
    return tvm.ir.register_op_attr(op_name, "FQPartitionRewrite", frewrite, level)


@tvm._ffi.register_object("relay.QPartitionExpr")
class QPartitionExpr(_expr.TempExpr):
    def __init__(self, expr):
        self.__init_handle_by_constructor__(_quantize.make_partition_expr, expr)


def partition_expr_check(expr):
    if isinstance(expr, QPartitionExpr):
        return True, expr.expr
    return False, expr


@register_partition_function("nn.conv2d")
def conv2d_partition_function(ref_call, new_args, ctx):
    """Rewrite function for conv2d for partition"""
    data_cond, data = partition_expr_check(new_args[0])
    kernel_cond, kernel = partition_expr_check(new_args[1])

    assert not kernel_cond
    if data_cond:
        data = new_args[0].realize()
    ret = _forward_op(ref_call, [data, kernel])
    return QPartitionExpr(ret)


def identity_partition_function(ref_call, new_args, ctx):
    cond, expr = partition_expr_check(new_args[0])
    if cond:
        return QPartitionExpr(_forward_op(ref_call, [expr]))
    return None


register_partition_function("clip", identity_partition_function)
register_partition_function("nn.relu", identity_partition_function)
register_partition_function("nn.max_pool2d", identity_partition_function)


def add_partition_generic(ref_call, new_args, ctx):
    """Rewrite function for ewise add for partition for generic devices"""
    lhs_cond, lhs = partition_expr_check(new_args[0])
    rhs_cond, rhs = partition_expr_check(new_args[1])
    if lhs_cond and rhs_cond:
        # - introduced by ResNet, when for the first residual connection
        #     ...
        #     %0 = nn.conv2d(%data, %meta[relay.Constant])
        #     %1 = add(%0, %meta[relay.Constant])
        #     %2 = nn.relu(%1)
        #     %3 = nn.max_pool2d(%2)
        #     ...
        #     %9 = nn.conv2d(%8, %meta[relay.Constant])
        #     %10 = add(%9, %meta[relay.Constant])
        #     %11 = add(%3, %10)  <- need to insert annotations for %3, %10
        #     ...
        lhs = new_args[0].realize()
        rhs = new_args[1].realize()
        return QPartitionExpr(_forward_op(ref_call, [lhs, rhs]))
    if not lhs_cond and rhs_cond:
        # - introduced by residual connection in ResNet
        #     ...
        #     %13 = nn.conv2d(%12, %meta[relay.Constant])
        #     %14 = add(%13, %meta[relay.Constant])
        #     %15 = annotation.cast_hint(%15, 'int8')
        #     %16 = annotation.stop_fusion(%16)
        #     %17 = add(%5, %16)
        #     %18 = nn.relu(%17)
        #     ...
        #     %24 = nn.conv2d(%23, %meta[relay.Constant])
        #     %25 = add(%24, %meta[relay.Constant])
        #     %26 = add(%18, %25)  <- need to insert annotations for %25
        #     ...
        rhs = new_args[1].realize()
        return _forward_op(ref_call, [lhs, rhs])
    if lhs_cond and not rhs_cond:
        if _analysis.check_constant(rhs):
            # - introduced by batch_norm: add(out, bias)
            return QPartitionExpr(_forward_op(ref_call, [lhs, rhs]))
        # - introduced by residual connection in MobileNetV2
        #     ...
        #     %81 = add(%80, meta[relay.Constant])
        #     %82 = annotation.cast_hint(%81, 'int8')
        #     %83 = annotation.stop_fusion(%82)
        #     %84 = add(%79, %83)
        #     ...
        #     %96 = nn.conv2d(%94, %meta[relay.Constant])
        #     %96 = add(%95, %meta[relay.Constant])
        #     %97 = add(%96, %84)  <- need to insert annotations for %96
        #     ...
        lhs = new_args[0].realize()
        return _forward_op(ref_call, [lhs, rhs])
    if not lhs_cond and not rhs_cond:
        # trivial case
        return None

    raise ValueError


def mul_partition_generic(ref_call, new_args, ctx):
    """Rewrite function for ewise mul for partition for generic devices"""
    lhs_cond, lhs = partition_expr_check(new_args[0])
    rhs_cond, rhs = partition_expr_check(new_args[1])

    if lhs_cond:
        # introduced by bn: multiply(out, scale)
        lhs = new_args[0].realize()
        return QPartitionExpr(_forward_op(ref_call, [lhs, rhs]))

    if rhs_cond:
        # introduced by efficientnet
        rhs = new_args[1].realize()
        return QPartitionExpr(_forward_op(ref_call, [lhs, rhs]))

    if not lhs_cond and not rhs_cond:
        # trivial case
        return None

    raise ValueError


# TODO(ziheng) enhance `register_partition_function` to dispatch
# for target automatically
@register_partition_function("add")
def add_partition_function(ref_call, new_args, ctx):
    """Rewrite function for ewise add for partition"""
    target = tvm.target.Target.current()
    if target and "cuda" in target.keys:
        # TODO(wuwei/ziheng) cuda specific rules
        return add_partition_generic(ref_call, new_args, ctx)
    return add_partition_generic(ref_call, new_args, ctx)


@register_partition_function("multiply")
def multiply_partition_function(ref_call, new_args, ctx):
    """Rewrite function for ewise multiply for partition"""
    return mul_partition_generic(ref_call, new_args, ctx)


# add cast after the relu op to make it run on vta
@register_partition_function("nn.global_avg_pool2d")
def global_avg_pool2d_partition_function(ref_call, new_args, ctx):
    cond, expr = partition_expr_check(new_args[0])
    if cond:
        expr = new_args[0].realize()
    else:
        expr = QPartitionExpr(new_args[0]).realize()

    return _forward_op(ref_call, [expr])
