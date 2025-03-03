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
import tvm
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T, ir as I
import tvm.testing


def test_nll_loss_backward():
    # fmt: off
    @tvm.script.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64"), weights: R.Tensor((4,), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def nll_loss_backward(rxplaceholder: T.Buffer((), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_2: T.Buffer((T.int64(2), T.int64(4), T.int64(5)), "int64"), rxplaceholder_3: T.Buffer((T.int64(4),), "float32"), pred_grad: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            all_weights = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            T_broadcast_to = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            all_weights_red = T.alloc_buffer(())
            T_divide = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("all_weights"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder_3[rxplaceholder_2[v_i0, v_i1, v_i2]], rxplaceholder_2[v_i0, v_i1, v_i2])
                    T.writes(all_weights[v_i0, v_i1, v_i2])
                    all_weights[v_i0, v_i1, v_i2] = rxplaceholder_3[rxplaceholder_2[v_i0, v_i1, v_i2]]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_broadcast_to"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[()])
                    T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                    T_broadcast_to[v_ax0, v_ax1, v_ax2] = rxplaceholder[()]
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("all_weights_red"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(all_weights[v_k0, v_k1, v_k2])
                    T.writes(all_weights_red[()])
                    with T.init():
                        all_weights_red[()] = T.float32(0)
                    all_weights_red[()] = all_weights_red[()] + all_weights[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_divide"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_broadcast_to[v_ax0, v_ax1, v_ax2], all_weights_red[()])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T_broadcast_to[v_ax0, v_ax1, v_ax2] / all_weights_red[()]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("pred_grad"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3], T_divide[v_i0, v_i2, v_i3])
                    T.writes(pred_grad[v_i0, v_i1, v_i2, v_i3])
                    pred_grad[v_i0, v_i1, v_i2, v_i3] = T.Select(v_i1 == rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3] * T.float32(-1) * T_divide[v_i0, v_i2, v_i3], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.nll_loss_backward, (output_grad, predictions, targets, weights), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_backward_no_weight():
    # fmt: off
    @I.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def te_nll_loss_backward_no_weight(rxplaceholder: T.Buffer((), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_2: T.Buffer((T.int64(2), T.int64(4), T.int64(5)), "int64"), pred_grad: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            T_full = T.alloc_buffer((T.int64(3),))
            all_weights = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            T_broadcast_to = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            all_weights_red = T.alloc_buffer(())
            T_divide = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(5)))
            for ax0 in range(T.int64(3)):
                with T.block("T_full"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads()
                    T.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(1)
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("all_weights"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_full[rxplaceholder_2[v_i0, v_i1, v_i2]], rxplaceholder_2[v_i0, v_i1, v_i2])
                    T.writes(all_weights[v_i0, v_i1, v_i2])
                    all_weights[v_i0, v_i1, v_i2] = T_full[rxplaceholder_2[v_i0, v_i1, v_i2]]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_broadcast_to"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[()])
                    T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                    T_broadcast_to[v_ax0, v_ax1, v_ax2] = rxplaceholder[()]
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("all_weights_red"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(all_weights[v_k0, v_k1, v_k2])
                    T.writes(all_weights_red[()])
                    with T.init():
                        all_weights_red[()] = T.float32(0)
                    all_weights_red[()] = all_weights_red[()] + all_weights[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_divide"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_broadcast_to[v_ax0, v_ax1, v_ax2], all_weights_red[()])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = T_broadcast_to[v_ax0, v_ax1, v_ax2] / all_weights_red[()]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("pred_grad"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3], T_divide[v_i0, v_i2, v_i3])
                    T.writes(pred_grad[v_i0, v_i1, v_i2, v_i3])
                    pred_grad[v_i0, v_i1, v_i2, v_i3] = T.Select(v_i1 == rxplaceholder_2[v_i0, v_i2, v_i3], all_weights[v_i0, v_i2, v_i3] * T.float32(-1) * T_divide[v_i0, v_i2, v_i3], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_nll_loss_backward_no_weight, (output_grad, predictions, targets), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_backward_no_batch():
    # fmt: off
    @tvm.script.ir_module
    class NLLLossBackward:
        @R.function
        def main(output_grad: R.Tensor((), "float32"), predictions: R.Tensor((4,), "float32"), targets: R.Tensor((), "int64"), weights: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
            gv: R.Tensor((4,), "float32") = R.grad.nll_loss_backward(output_grad, predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(output_grad: R.Tensor((), dtype="float32"), predictions: R.Tensor((4,), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor((4,), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.nll_loss_backward, (output_grad, predictions, targets, weights), out_sinfo=R.Tensor((4,), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def nll_loss_backward(rxplaceholder: T.Buffer((), "float32"), rxplaceholder_1: T.Buffer((T.int64(4),), "float32"), rxplaceholder_2: T.Buffer((), "int64"), rxplaceholder_3: T.Buffer((T.int64(4),), "float32"), pred_grad: T.Buffer((T.int64(4),), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            all_weights = T.alloc_buffer(())
            T_broadcast_to = T.alloc_buffer(())
            T_divide = T.alloc_buffer(())
            with T.block("all_weights"):
                vi = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(rxplaceholder_3[rxplaceholder_2[()]], rxplaceholder_2[()])
                T.writes(all_weights[()])
                all_weights[()] = rxplaceholder_3[rxplaceholder_2[()]]
            with T.block("T_broadcast_to"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(rxplaceholder[()])
                T.writes(T_broadcast_to[()])
                T_broadcast_to[()] = rxplaceholder[()]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_broadcast_to[()], all_weights[()])
                T.writes(T_divide[()])
                T_divide[()] = T_broadcast_to[()] / all_weights[()]
            for i in range(T.int64(4)):
                with T.block("pred_grad"):
                    v_i = T.axis.spatial(T.int64(4), i)
                    T.reads(rxplaceholder_2[()], all_weights[()], T_divide[()])
                    T.writes(pred_grad[v_i])
                    pred_grad[v_i] = T.Select(v_i == rxplaceholder_2[()], all_weights[()] * T.float32(-1) * T_divide[()], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(NLLLossBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_backward():
    # fmt: off
    @tvm.script.ir_module
    class MaxPool2DBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")):
            gv = R.grad.max_pool2d_backward(output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def max_pool2d_backward(A: T.Buffer((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "float32"), B: T.Buffer((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"), T_pool_grad: T.Buffer((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            pad_temp = T.alloc_buffer((T.int64(3), T.int64(2), T.int64(15), T.int64(13)))
            maxpool_grad_argmax_v0 = T.alloc_buffer((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "int64")
            maxpool_grad_argmax_v1 = T.alloc_buffer((T.int64(3), T.int64(2), T.int64(6), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(3), T.int64(2), T.int64(15), T.int64(13)):
                with T.block("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(B[v_ax0, v_ax1, v_ax2 - T.int64(2), v_ax3 - T.int64(1)])
                    T.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(2) <= v_ax2 and v_ax2 < T.int64(12) and T.int64(1) <= v_ax3 and v_ax3 < T.int64(11), B[v_ax0, v_ax1, v_ax2 - T.int64(2), v_ax3 - T.int64(1)], T.float32(-3.4028234663852886e+38))
            for ax0, ax1, ax2, ax3, dh, dw in T.grid(T.int64(3), T.int64(2), T.int64(6), T.int64(5), T.int64(5), T.int64(5)):
                with T.block("maxpool_grad_argmax"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_dh, v_dw = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, dh, dw])
                    T.reads(pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw])
                    T.writes(maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3], maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3])
                    with T.init():
                        maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] = T.int64(-1)
                        maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-3.4028234663852886e+38)
                    v_maxpool_grad_argmax_v0: T.int64 = T.Select(maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] > pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw] or maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] == pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw] and maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] < v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(26) + v_dh * T.int64(13) + v_ax3 * T.int64(2) + v_dw, maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3], v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(26) + T.Cast("int64", v_dh) * T.int64(13) + v_ax3 * T.int64(2) + T.Cast("int64", v_dw))
                    v_maxpool_grad_argmax_v1: T.float32 = T.Select(maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] > pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw], maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3], pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_dh, v_ax3 * T.int64(2) + v_dw])
                    maxpool_grad_argmax_v0[v_ax0, v_ax1, v_ax2, v_ax3] = v_maxpool_grad_argmax_v0
                    maxpool_grad_argmax_v1[v_ax0, v_ax1, v_ax2, v_ax3] = v_maxpool_grad_argmax_v1
            for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(3), T.int64(2), T.int64(10), T.int64(10), T.int64(3), T.int64(3)):
                with T.block("T_pool_grad"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                    T.reads(maxpool_grad_argmax_v0[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww], A[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww])
                    T.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                    with T.init():
                        T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(3), T.int64(0), T.Div(v_ax2 - T.int64(3), T.int64(2)) + T.int64(1)) <= T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh and T.Select(v_ax3 < T.int64(4), T.int64(0), T.Div(v_ax3 - T.int64(4), T.int64(2)) + T.int64(1)) <= T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww and maxpool_grad_argmax_v0[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww] == v_ax0 * T.int64(390) + v_ax1 * T.int64(195) + v_ax2 * T.int64(13) + v_ax3 + T.int64(27), A[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww], T.float32(0))

        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), dtype="float32"), data: R.Tensor((3, 2, 10, 10), dtype="float32")) -> R.Tensor((3, 2, 10, 10), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.max_pool2d_backward, (output_grad, data), out_sinfo=R.Tensor((3, 2, 10, 10), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(MaxPool2DBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_backward():
    # fmt: off
    @tvm.script.ir_module
    class AvgPool2DBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), "float32"), data: R.Tensor((3, 2, 10, 10), "float32")):
            gv = R.grad.avg_pool2d_backward(output_grad, data, (5, 5), (2, 2), (2, 1, 2, 1), (1, 1), True)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def avg_pool2d_backward(rxplaceholder: T.Buffer((T.int64(3), T.int64(2), T.int64(6), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32"), T_pool_grad: T.Buffer((T.int64(3), T.int64(2), T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(3), T.int64(2), T.int64(10), T.int64(10), T.int64(3), T.int64(3)):
                with T.block("T_pool_grad"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                    T.reads(rxplaceholder[v_ax0, v_ax1, T.Div((v_ax2 + T.int64(2)), T.int64(2)) - v_wh, T.Div((v_ax3 + T.int64(1)), T.int64(2)) - v_ww])
                    T.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                    with T.init():
                        T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(3), T.int64(0), T.Div(v_ax2 - T.int64(3), T.int64(2)) + T.int64(1)) <= T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh and T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh < T.int64(6) and T.Select(v_ax3 < T.int64(4), T.int64(0), T.Div(v_ax3 - T.int64(4), T.int64(2)) + T.int64(1)) <= T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww and T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww < T.int64(5), rxplaceholder[v_ax0, v_ax1, T.Div(v_ax2 + T.int64(2), T.int64(2)) - v_wh, T.Div(v_ax3 + T.int64(1), T.int64(2)) - v_ww] / T.Cast("float32", T.max((T.min(T.Div(v_ax2 + T.int64(2), T.int64(2)) * T.int64(2) + T.int64(3) - v_wh * T.int64(2), T.int64(10)) - T.max(T.Div(v_ax2 + T.int64(2), T.int64(2)) * T.int64(2) - v_wh * T.int64(2) - T.int64(2), T.int64(0))) * (T.min(T.Div(v_ax3 + T.int64(1), T.int64(2)) * T.int64(2) + T.int64(4) - v_ww * T.int64(2), T.int64(10)) - T.max(T.Div(v_ax3 + T.int64(1), T.int64(2)) * T.int64(2) - v_ww * T.int64(2) - T.int64(1), T.int64(0))), T.int64(1))), T.float32(0))
        @R.function
        def main(output_grad: R.Tensor((3, 2, 6, 5), dtype="float32"), data: R.Tensor((3, 2, 10, 10), dtype="float32")) -> R.Tensor((3, 2, 10, 10), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.avg_pool2d_backward, (output_grad, data), out_sinfo=R.Tensor((3, 2, 10, 10), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2DBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_take_backward():
    # fmt: off
    @tvm.script.ir_module
    class TakeBackward:
        @R.function
        def main(output_grad: R.Tensor((3, 2, 5), "float32"), x: R.Tensor((3, 4, 5), "float32"), indices: R.Tensor((2,), "int32")):
            gv = R.grad.take_backward(output_grad, x, indices, axis=1)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def take_backward(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, out_buf: T.Buffer((T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(3), T.int64(2), T.int64(5)), offset_factor=1)
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(3), T.int64(4), T.int64(5)), offset_factor=1)
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (T.int64(2),), "int32", offset_factor=1)
            with T.block("take_backward"):
                for i in range(T.int64(60)):
                    out_buf[i // T.int64(5) // T.int64(4), i // T.int64(5) % T.int64(4), i % T.int64(5)] = T.float32(0)
                for parallel, serial in T.grid(T.int64(15), T.int64(2)):
                    out_buf[(parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) % T.int64(5)] = out_buf[(parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(4), (parallel // T.int64(5) * T.int64(5) * T.int64(4) + T.Cast("int64", rxplaceholder_2[serial]) * T.int64(5) + parallel % T.int64(5)) % T.int64(5)] + rxplaceholder[(parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) // T.int64(5) // T.int64(2), (parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) // T.int64(5) % T.int64(2), (parallel // T.int64(5) * T.int64(5) * T.int64(2) + serial * T.int64(5) + parallel % T.int64(5)) % T.int64(5)]

        @R.function
        def main(output_grad: R.Tensor((3, 2, 5), dtype="float32"), x: R.Tensor((3, 4, 5), dtype="float32"), indices: R.Tensor((2,), dtype="int32")) -> R.Tensor((3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.take_backward, (output_grad, x, indices), out_sinfo=R.Tensor((3, 4, 5), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(TakeBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_take_backward_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class TakeBackward:
        @R.function
        def main(output_grad: R.Tensor(("m", "i"), "float32"), x: R.Tensor(("m", "n"), "float32"), indices: R.Tensor(("i",), "int32")):
            m = T.int64()
            i = T.int64()
            gv = R.grad.take_backward(output_grad, x, indices, axis=1)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def take_backward(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_take_backward: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            m, i = T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (m, i), offset_factor=1)
            n = T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (m, n), offset_factor=1)
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (i,), "int32", offset_factor=1)
            out_buf = T.match_buffer(var_take_backward, (m, n))
            with T.block("take_backward"):
                for i_1 in range(m * n):
                    out_buf[i_1 // n, i_1 % n] = T.float32(0)
                for parallel, serial in T.grid(m, i):
                    out_buf[(parallel * n + T.Cast("int64", rxplaceholder_2[serial])) // n, (parallel * n + T.Cast("int64", rxplaceholder_2[serial])) % n] = out_buf[(parallel * n + T.Cast("int64", rxplaceholder_2[serial])) // n, (parallel * n + T.Cast("int64", rxplaceholder_2[serial])) % n] + rxplaceholder[(parallel * i + serial) // i, (parallel * i + serial) % i]

        @R.function
        def main(output_grad: R.Tensor(("m", "i"), dtype="float32"), x: R.Tensor(("m", "n"), dtype="float32"), indices: R.Tensor(("i",), dtype="int32")) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.int64()
            n = T.int64()
            i = T.int64()
            cls = Expected
            gv = R.call_tir(cls.take_backward, (output_grad, x, indices), out_sinfo=R.Tensor((m, n), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(TakeBackward)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
