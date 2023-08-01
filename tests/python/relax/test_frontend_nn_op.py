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
import pytest

import tvm
import tvm.testing
from tvm import relax, te, tir
from tvm.relax.frontend.nn import Module, Tensor, op, spec
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_binary():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = op.add(x, y)
            z1 = op.multiply(x, y)
            z2 = op.divide(x, y)
            z3 = op.matmul(x, y)
            z4 = op.maximum(x, y)
            z5 = op.minimum(x, y)
            return (z0, z1, z2, z3, z4, z5)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), y: R.Tensor((10, 1), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((1, 1), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")), R.Tuple(R.Object)):
        with R.dataflow():
            add: R.Tensor((10, 10), dtype="float32") = R.add(x, y)
            mul: R.Tensor((10, 10), dtype="float32") = R.multiply(x, y)
            divide: R.Tensor((10, 10), dtype="float32") = R.divide(x, y)
            matmul: R.Tensor((1, 1), dtype="float32") = R.matmul(x, y, out_dtype="void")
            maximum: R.Tensor((10, 10), dtype="float32") = R.maximum(x, y)
            minimum: R.Tensor((10, 10), dtype="float32") = R.minimum(x, y)
            gv1: R.Tuple(R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((1, 1), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")), R.Tuple(R.Object)) = (add, mul, divide, matmul, maximum, minimum), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([1, 10], "float32"), "y": spec.Tensor([10, 1], "float32")}}
    )

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_manipulate():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.broadcast_to(x, [2, 5, 2])
            z1 = op.permute_dims(x, [2, 1, 0])
            z2 = op.reshape(x, [1, 10])
            z3 = op.repeat(x, repeats=2, axis=1)
            z4 = op.squeeze(x, 0)
            return (z0, z1, z2, z3, z4)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 5, 2), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32")), R.Tuple(R.Object)):
        with R.dataflow():
            broadcast_to: R.Tensor((2, 5, 2), dtype="float32") = R.broadcast_to(x, R.shape([2, 5, 2]))
            permute_dims: R.Tensor((2, 5, 1), dtype="float32") = R.permute_dims(x, axes=[2, 1, 0])
            reshape: R.Tensor((1, 10), dtype="float32") = R.reshape(x, R.shape([1, 10]))
            repeat: R.Tensor((1, 10, 2), dtype="float32") = R.repeat(x, repeats=2, axis=1)
            squeeze: R.Tensor((5, 2), dtype="float32") = R.squeeze(x, axis=[0])
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32")), R.Tuple(R.Object)) = (broadcast_to, permute_dims, reshape, repeat, squeeze), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(spec={"test": {"x": spec.Tensor([1, 5, 2], "float32")}})

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_index():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = op.take(x, y, axis=2)
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 1, 10), dtype="float32"), y: R.Tensor((5,), dtype="int32"), _io: R.Object) -> R.Tuple(R.Tensor((2, 1, 5), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            take: R.Tensor((2, 1, 5), dtype="float32") = R.take(x, y, axis=2)
            gv1: R.Tuple(R.Tensor((2, 1, 5), dtype="float32"), R.Tuple(R.Object)) = take, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(
        spec={"test": {"x": spec.Tensor([2, 1, 10], "float32"), "y": spec.Tensor([5], "int32")}}
    )

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_datatype():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.astype(x, "float16")
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 1, 10), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tensor((2, 1, 10), dtype="float16"), R.Tuple(R.Object)):
        with R.dataflow():
            astype: R.Tensor((2, 1, 10), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tuple(R.Tensor((2, 1, 10), dtype="float16"), R.Tuple(R.Object)) = astype, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(spec={"test": {"x": spec.Tensor([2, 1, 10], "float32")}})

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_nn():
    class Model(Module):
        def test(self, x: Tensor, weight: Tensor):
            silu_out = op.silu(x)
            softmax_out = op.softmax(x, axis=2)
            rms_norm_out = op.rms_norm(x, weight, axes=[-2, -1])
            rms_norm_with_bias_out = op.rms_norm(x, weight, axes=[-2, -1])
            return x

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 3, 4, 5), dtype="float32"), weight: R.Tensor((4, 5), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            silu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.silu(x)
            softmax: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.softmax(x, axis=2)
            rms_norm: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05)
            rms_norm1: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05)
            gv1: R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Object)) = x, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(
        spec={
            "test": {
                "x": spec.Tensor([2, 3, 4, 5], "float32"),
                "weight": spec.Tensor([4, 5], "float32"),
                "bias": spec.Tensor([4, 5], "float32"),
            }
        }
    )

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_create():
    class Model(Module):
        def test(self, x: Tensor):
            triu_out = op.triu(x)
            full_with_scalar_out = op.full([10, 10], fill_value=10)
            full_with_FloatImm_out = op.full(
                [10, 10], fill_value=tir.FloatImm(dtype="float32", value=10)
            )
            full_with_Tensor_out = op.full(
                [10, 10], fill_value=Tensor.from_scalar(10, dtype="float32")
            )
            zeros_out = op.zeros([10, 10])
            zeros_fp16_out = op.zeros([10, 10], dtype="float16")
            return x

    # fmt: off
    @R.function
    def test(x: R.Tensor((10, 10), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            triu: R.Tensor((10, 10), dtype="float32") = R.triu(x, k=0)
            full: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            full1: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            full2: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            zeros: R.Tensor((10, 10), dtype="float32") = R.zeros(R.shape([10, 10]), dtype="float32")
            zeros1: R.Tensor((10, 10), dtype="float16") = R.zeros(R.shape([10, 10]), dtype="float16")
            gv1: R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Object)) = x, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(spec={"test": {"x": spec.Tensor([10, 10], "float32")}})

    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_tensor_expr_op():
    class Model(Module):
        def test(self, x: Tensor):
            tensor_expr_op_out = op.tensor_expr_op(
                tensor_expr_func=lambda x: x + 1, name_hint="add_one", args=[x]
            )
            return x

    # fmt: off
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def add_one(A: T.Buffer((T.int64(10), T.int64(10)), "float32"), T_add: T.Buffer((T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(10), T.int64(10)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + T.float32(1)

        @R.function
        def _initialize_effect() -> R.Tuple(R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tuple(R.Object) = (_io,)
                gv: R.Tuple(R.Object) = lv
                R.output(gv)
            return gv

        @R.function
        def test(x: R.Tensor((10, 10), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Object)):
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.add_one, (x,), out_sinfo=R.Tensor((10, 10), dtype="float32"))
                add_one1: R.Tensor((10, 10), dtype="float32") = lv1
                gv1: R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Object)) = x, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(spec={"test": {"x": spec.Tensor([10, 10], "float32")}})

    tvm.ir.assert_structural_equal(irmodule, Expected)


if __name__ == "__main__":
    tvm.testing.main()
