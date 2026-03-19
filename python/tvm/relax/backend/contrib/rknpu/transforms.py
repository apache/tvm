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

"""RKNPU-specific Relax IR transformations.

These passes decompose high-level ops (LayerNorm, Softmax) into
NPU-compatible primitives before pattern matching / fusion.

Must run BEFORE FuseOpsByPattern in the compilation pipeline.
"""

import numpy as np
import tvm
from tvm import relax
from tvm.relax import Call, Expr, PyExprMutator, expr_functor
from tvm.ir.module import IRModule


@expr_functor.mutator
class _LayerNormDecomposer(PyExprMutator):
    """Visitor that decomposes nn.layer_norm into NPU primitives.

    Rewrites ``layer_norm(x, gamma, beta)`` over the last axis of a 2-D
    ``[M, K]`` tensor into a sequence of matmul / add / multiply / rsqrt
    operations that each independently match RKNPU patterns.

    Decomposition (9 ops):
        1. neg_mean      = x @ W_neg_mean            [M,K]x[K,1] -> [M,1]
        2. centered      = x + neg_mean              [M,K]  (column broadcast)
        3. sq            = centered * centered        [M,K]
        4. var           = sq @ W_pos_mean            [M,K]x[K,1] -> [M,1]
        5. var_eps       = var + eps_const            [M,1]
        6. inv_std       = rsqrt(var_eps)             [M,1]
        7. normed        = centered * inv_std         [M,K]  (column broadcast)
        8. scaled        = normed * gamma             [M,K]  (row broadcast)
        9. output        = scaled + beta              [M,K]  (row broadcast)

    Requirements for decomposition:
        - 2-D input ``[M, K]`` with concrete (non-symbolic) shapes
        - Normalization over last axis only (``axes=[-1]``)
        - ``K % 32 == 0`` (NPU matmul alignment)
    """

    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: Call) -> Expr:
        # Visit children first.
        call = self.visit_expr_post_order(call)

        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "relax.nn.layer_norm":
            return call

        x, gamma, beta = call.args

        # --- Validate that we can decompose ---
        axes = [int(a) for a in call.attrs.axes]
        eps = float(call.attrs.epsilon)

        x_shape = x.struct_info.shape
        ndim = len(x_shape)

        # Only normalise over the last axis of a 2-D tensor.
        if ndim != 2:
            return call
        if axes != [-1] and axes != [ndim - 1]:
            return call

        # Need concrete shapes.
        try:
            M = int(x_shape[0])
            K = int(x_shape[1])
        except (TypeError, ValueError):
            return call

        # K must be 32-aligned for NPU matmul.
        if K % 32 != 0:
            return call

        bb = self.builder_

        # --- Constant weight matrices (fp16) ---
        w_neg_mean = relax.const(
            np.full((K, 1), -1.0 / K, dtype="float16"), "float16"
        )
        w_pos_mean = relax.const(
            np.full((K, 1), 1.0 / K, dtype="float16"), "float16"
        )
        eps_const = relax.const(
            np.full((M, 1), eps, dtype="float16"), "float16"
        )

        # 1. neg_mean = x @ W_neg_mean  ->  [M, 1]
        neg_mean = bb.emit(relax.op.matmul(x, w_neg_mean))

        # 2. centered = x + neg_mean  (column-broadcast add)
        centered = bb.emit(relax.op.add(x, neg_mean))

        # 4. sq = centered * centered
        sq = bb.emit(relax.op.multiply(centered, centered))

        # 5. var = sq @ W_pos_mean  ->  [M, 1]
        var = bb.emit(relax.op.matmul(sq, w_pos_mean))

        # 6. var_eps = var + eps  (same-shape add, goes to NPU)
        var_eps = bb.emit(relax.op.add(var, eps_const))

        # 7. inv_std = rsqrt(var_eps)  ->  [M, 1]
        inv_std = bb.emit(relax.op.rsqrt(var_eps))

        # 8. normed = centered * inv_std  (column-broadcast mul)
        normed = bb.emit(relax.op.multiply(centered, inv_std))

        # 10-11. Keep gamma/beta as 1-D vectors so legalization can lower them
        # through the existing bias-broadcast elementwise path without extra
        # reshape/broadcast_to buffers that FuseTIR later rejects.
        scaled = bb.emit(relax.op.multiply(normed, gamma))
        output = bb.emit(relax.op.add(scaled, beta))

        return output


@expr_functor.mutator
class _SoftmaxDecomposer(PyExprMutator):
    """Visitor that decomposes nn.softmax into NPU-compatible primitives.

    Rewrites ``softmax(x, axis=-1)`` on a 2-D ``[M, K]`` tensor:

        1. max_val     = max(x, axis=-1, keepdims=True)     [M,1]  CPU  (no NPU reduce)
        2. neg_max     = negative(max_val)                  [M,1]  CPU
        3. x_shifted   = add(x, neg_max)                    [M,K]  NPU  (column broadcast)
        4. exp_x       = exp(x_shifted)                     [M,K]  NPU
        5. sum_exp     = matmul(exp_x, ones_K1)             [M,1]  NPU
        6. inv_sum     = divide(one, sum_exp)               [M,1]  CPU
        7. result      = multiply(exp_x, inv_sum)           [M,K]  NPU  (column broadcast)

    CPU-only: max-reduce, negate, and reciprocal.
    The broadcasted add/multiply remain on the NPU using a dedicated Mx1
    column-broadcast stage mode instead of k=1 matmul expansion.

    Requirements for decomposition:
        - 2-D input ``[M, K]`` with concrete (non-symbolic) shapes
        - Softmax over the last axis (``axis=-1``)
    """

    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: Call) -> Expr:
        call = self.visit_expr_post_order(call)

        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "relax.nn.softmax":
            return call

        (x,) = call.args
        axis = int(call.attrs.axis)

        x_shape = x.struct_info.shape
        ndim = len(x_shape)

        # Only decompose 2-D softmax on last axis.
        if ndim != 2:
            return call
        if axis != -1 and axis != ndim - 1:
            return call

        try:
            M = int(x_shape[0])
            K = int(x_shape[1])
        except (TypeError, ValueError):
            return call

        bb = self.builder_

        # --- 1. max_val = max(x, axis=-1, keepdims=True) → [M, 1]  (CPU) ---
        # No NPU reduce-max op exists.
        max_val = bb.emit(relax.op.max(x, axis=[-1], keepdims=True))

        # --- 2-3. Broadcast -max via dedicated column-broadcast add stage ---
        neg_max = bb.emit(relax.op.negative(max_val))
        x_shifted = bb.emit(relax.op.add(x, neg_max))

        # --- 4. exp_x = exp(x_shifted) → [M, K]  (NPU) ---
        exp_x = bb.emit(relax.op.exp(x_shifted))

        # --- 5. sum_exp = matmul(exp_x, ones_K1) → [M, 1]  (NPU) ---
        ones_K1 = relax.const(np.ones((K, 1), dtype="float16"), "float16")
        sum_exp = bb.emit(relax.op.matmul(exp_x, ones_K1))

        # --- 6. inv_sum = 1 / sum_exp → [M, 1]  (CPU) ---
        # Stays on CPU: [M,1] is tiny (M divides, microseconds).
        # Reciprocal LUT domain [0,64] is too small for sum_exp when K>64.
        one = relax.const(np.ones((1, 1), dtype="float16"), "float16")
        inv_sum = bb.emit(relax.op.divide(one, sum_exp))

        # --- 7. result = exp_x * inv_sum → [M, K]  (NPU column broadcast) ---
        result = bb.emit(relax.op.multiply(exp_x, inv_sum))

        return result


def _apply_decomposer(mod, decomposer_cls):
    """Helper to apply a PyExprMutator-based decomposer to all functions."""
    decomposer = decomposer_cls(mod)
    for gv, func in mod.functions_items():
        if isinstance(func, relax.Function):
            if func.attrs and "Primitive" in func.attrs.keys():
                if func.attrs["Primitive"] != 0:
                    continue
            new_func = decomposer.visit_expr(func)
            decomposer.builder_.update_func(gv, new_func)
    return decomposer.builder_.get()


@tvm.transform.module_pass(opt_level=0, name="DecomposeLayerNormForRKNPU")
class DecomposeLayerNormForRKNPU:
    """Module pass that decomposes nn.layer_norm into NPU-compatible primitives.

    Must run BEFORE FuseOpsByPattern in the compilation pipeline.
    """

    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        return _apply_decomposer(mod, _LayerNormDecomposer)


@tvm.transform.module_pass(opt_level=0, name="DecomposeSoftmaxForRKNPU")
class DecomposeSoftmaxForRKNPU:
    """Module pass that decomposes nn.softmax into CPU+NPU primitives.

    Must run BEFORE FuseOpsByPattern in the compilation pipeline.
    """

    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        return _apply_decomposer(mod, _SoftmaxDecomposer)
