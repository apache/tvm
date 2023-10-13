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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Dense alter op functions for x86"""

import tvm
from tvm import autotvm, relay, te
from tvm.target.codegen import target_has_features
from tvm.target.x86 import get_x86_simd_32bit_lanes

from .. import nn
from ..nn import dense_alter_layout
from ..utils import get_const_tuple
from .dense import _default_dense_pack_config


def check_int8_applicable(x, y, allow_padding=False):
    """Check (u)int8 SIMD elegibility."""
    # x86 SIMD
    simd_avai = target_has_features(["avx512bw", "avx512f"])
    simd_avai |= target_has_features("amx-int8")
    simd_avai |= target_has_features("avx512vnni")
    simd_avai |= target_has_features("avxvnni")
    simd_avai |= target_has_features("avx2")
    simd_avai |= target_has_features("ssse3")
    # arm SIMD
    simd_avai |= target_has_features("dotprod")

    vec_width = get_x86_simd_32bit_lanes() if get_x86_simd_32bit_lanes() else 16

    return (
        simd_avai
        and x.dtype in ("int8", "uint8")
        and y.dtype in ("int8", "uint8")
        and (allow_padding or y.shape[-2] % vec_width == 0)
    )


@dense_alter_layout.register(["cpu", "arm_cpu"])
def _alter_dense_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    data_tensor, weight_tensor = tinfos
    out_dtype = out_type.dtype
    M, K = get_const_tuple(data_tensor.shape)
    N, _ = get_const_tuple(weight_tensor.shape)

    if (
        check_int8_applicable(data_tensor, weight_tensor, allow_padding=True)
        and data_tensor.dtype in ("uint8", "int8")
        and weight_tensor.dtype in ("uint8", "int8")
    ):
        vec_width = get_x86_simd_32bit_lanes() if get_x86_simd_32bit_lanes() else 16
        weight_layout = f"NC{vec_width}n4c"
        return relay.nn.contrib_dense_pack(inputs[0], inputs[1], weight_layout, None, out_dtype)

    _, outs = relay.backend.te_compiler.select_implementation(
        relay.op.get("nn.dense"), attrs, tinfos, out_type, target
    )
    workload = autotvm.task.get_workload(outs)

    if workload:
        cfg = dispatch_ctx.query(target, workload)
        topi_impl = workload[0]
        if topi_impl == "dense_pack.x86":
            if cfg.is_fallback:
                _default_dense_pack_config(cfg, M, N, K)
            packw_bn = cfg["tile_x"].size[-1]
            weight_layout = f"NC{packw_bn}n"
            new_weight = te.placeholder((N // packw_bn, K, packw_bn), dtype=weight_tensor.dtype)
            # Relay dense doesn't have bias.
            new_workload = autotvm.task.args_to_workload(
                [data_tensor, new_weight, None, out_dtype], topi_impl
            )
            dispatch_ctx.update(target, new_workload, cfg)
            return relay.nn.contrib_dense_pack(inputs[0], inputs[1], weight_layout, None, out_dtype)

    return None


def int8_int8_legalize(inputs, arg_types, op, attrs, need_expand=False):
    """Legalizes s8, s8 -> s32 GEMM op for SIMD."""
    if check_int8_applicable(arg_types[0], arg_types[1], allow_padding=True):

        x, y = inputs

        # x{data} int8 -> uint8
        if arg_types[0].dtype == "int8":
            x = relay.cast(x, "int32")
            x = relay.add(x, relay.const(128, "int32"))
            x = relay.cast(x, "uint8")

            x_adjust_shift = relay.const(128, "int32") * relay.sum(
                relay.cast(y, "int32"), axis=[-1]
            )

            if need_expand:
                x_adjust_shift = relay.expand_dims(x_adjust_shift, axis=1)

        # y{weight} uint8 -> int8
        if arg_types[1].dtype == "uint8":
            y = relay.cast(y, "int32")
            y = relay.subtract(y, relay.const(128, "int32"))
            y = relay.cast(y, "int8")

            y_adjust_shift = relay.const(128, "int32") * relay.sum(
                relay.cast(x, "int32"), axis=[-1]
            )

            if need_expand:
                y_adjust_shift = relay.expand_dims(y_adjust_shift, axis=1)

        analyzer = tvm.arith.Analyzer()
        x_shape = arg_types[0].shape
        y_shape = arg_types[1].shape

        inst_n = get_x86_simd_32bit_lanes() if get_x86_simd_32bit_lanes() else 16
        inst_k = get_x86_simd_32bit_lanes() if get_x86_simd_32bit_lanes() else 4
        pad_n = analyzer.simplify((inst_n - y_shape[-2] % inst_n) % inst_n)
        pad_k = analyzer.simplify((inst_k - y_shape[-1] % inst_k) % inst_k)
        if pad_k != 0 or pad_n != 0:
            ndim = len(x_shape)
            unpadded_dims = [(0, 0)] * (ndim - 2)
            padding_y = [(0, 0)] * (len(y_shape) - 2) + [(0, pad_n), (0, pad_k)]
            padded_y = relay.nn.pad(y, pad_width=padding_y, pad_value=0)
            if pad_k != 0:
                padding_x = [(0, 0)] * (len(x_shape) - 1) + [(0, pad_k)]
                padded_x = relay.nn.pad(x, pad_width=padding_x, pad_value=0)
            else:
                padded_x = x
            out = op(padded_x, padded_y, **attrs)
            if pad_n != 0:
                begin = [0] * len(x_shape)
                end = x_shape[:-2] + [x_shape[-2], y_shape[-2]]
                out = relay.strided_slice(out, begin, end, slice_mode="size")
        else:
            out = op(x, y, **attrs)

        if arg_types[0].dtype == "int8":
            # int8->uint8 +adjust +padding
            out = relay.subtract(out, x_adjust_shift)
        if arg_types[1].dtype == "uint8":
            # uint8->int8 +adjust +padding
            out = relay.add(out, y_adjust_shift)

        return out

    return None


@nn.dense_legalize.register("cpu")
def _dense_legalize(attrs, inputs, arg_types):
    """Legalizes s8, s8 -> s32 dense for SIMD."""
    return int8_int8_legalize(inputs, arg_types, relay.nn.dense, attrs)


@nn.batch_matmul_legalize.register("cpu")
def _batch_matmul_legalize(attrs, inputs, arg_types):
    """Legalizes s8, s8 -> s32 batch_matmul for SIMD."""
    if attrs["transpose_a"] or not attrs["transpose_b"]:
        return None
    return int8_int8_legalize(inputs, arg_types, relay.nn.batch_matmul, attrs, need_expand=True)
