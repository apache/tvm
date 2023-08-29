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
import torch
import sys
import io

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
            z6 = op.subtract(x, y)
            return (z0, z1, z2, z3, z4, z5, z6)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), y: R.Tensor((10, 1), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((1, 1), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")), R.Tuple(R.Object)):
        with R.dataflow():
            add: R.Tensor((10, 10), dtype="float32") = R.add(x, y)
            mul: R.Tensor((10, 10), dtype="float32") = R.multiply(x, y)
            divide: R.Tensor((10, 10), dtype="float32") = R.divide(x, y)
            matmul: R.Tensor((1, 1), dtype="float32") = R.matmul(x, y, out_dtype="void")
            maximum: R.Tensor((10, 10), dtype="float32") = R.maximum(x, y)
            minimum: R.Tensor((10, 10), dtype="float32") = R.minimum(x, y)
            subtract: R.Tensor((10, 10), dtype="float32") = R.subtract(x, y)
            gv1: R.Tuple(R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((1, 1), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")), R.Tuple(R.Object)) = (add, mul, divide, matmul, maximum, minimum, subtract), (_io,)
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
            z5 = op.unsqueeze(x, 0)
            z6 = op.concat([x, x], dim=0)
            return (z0, z1, z2, z3, z4, z5, z6)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 5, 2), dtype="float32"), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32"), R.Tensor((1, 1, 5, 2), dtype="float32"), R.Tensor((2, 5, 2), dtype="float32")), R.Tuple(R.Object)):
        with R.dataflow():
            broadcast_to: R.Tensor((2, 5, 2), dtype="float32") = R.broadcast_to(x, R.shape([2, 5, 2]))
            permute_dims: R.Tensor((2, 5, 1), dtype="float32") = R.permute_dims(x, axes=[2, 1, 0])
            reshape: R.Tensor((1, 10), dtype="float32") = R.reshape(x, R.shape([1, 10]))
            repeat: R.Tensor((1, 10, 2), dtype="float32") = R.repeat(x, repeats=2, axis=1)
            squeeze: R.Tensor((5, 2), dtype="float32") = R.squeeze(x, axis=[0])
            unsqueeze: R.Tensor((1, 1, 5, 2), dtype="float32") = R.expand_dims(x, axis=0)
            concat: R.Tensor((2, 5, 2), dtype="float32") = R.concat([x, x], axis=0)
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32"), R.Tensor((1, 1, 5, 2), dtype="float32"), R.Tensor((2, 5, 2), dtype="float32")), R.Tuple(R.Object)) = (broadcast_to, permute_dims, reshape, repeat, squeeze, unsqueeze, concat), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([1, 5, 2], "float32")}})

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


def test_image():
    class Model(Module):
        def test(self, x: Tensor, weight: Tensor, bias: Tensor):
            padded = op.pad(x, [0, 0, 0, 0, 1, 1, 1, 1])
            conv2d = op.conv2d(padded, weight, bias)
            interpolate = op.interpolate(x, size=[40, 40])
            return (conv2d, interpolate)

    @R.function
    def test(
        x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        weight: R.Tensor((32, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(
        R.Tuple(
            R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tensor((1, 3, 40, 40), dtype="float32")
        ),
        R.Tuple(R.Object),
    ):
        with R.dataflow():
            lv0: R.Tensor((1, 3, 34, 34), dtype="float32") = R.nn.pad(x, (0, 0, 0, 0, 1, 1, 1, 1))
            lv1: R.Tensor((1, 32, 32, 32), dtype="float32") = R.nn.conv2d(
                lv0,
                weight,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="void",
            )
            lv2: R.Tensor((1, 32, 1, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1, 1]))
            conv2d: R.Tensor((1, 32, 32, 32), dtype="float32") = R.add(lv1, lv2)
            interpolate: R.Tensor((1, 3, 40, 40), dtype="float32") = R.image.resize2d(
                x,
                R.shape([40, 40]),
                roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)],
                layout="NCHW",
                method="nearest_neighbor",
                coordinate_transformation_mode="asymmetric",
                rounding_method="round",
                cubic_alpha=-0.5,
                cubic_exclude=0,
                extrapolation_value=0,
                out_dtype="void",
            )
            gv1: R.Tuple(
                R.Tuple(
                    R.Tensor((1, 32, 32, 32), dtype="float32"),
                    R.Tensor((1, 3, 40, 40), dtype="float32"),
                ),
                R.Tuple(R.Object),
            ) = (conv2d, interpolate), (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "x": spec.Tensor([1, 3, 32, 32], "float32"),
                "weight": spec.Tensor([32, 3, 3, 3], "float32"),
                "bias": spec.Tensor([32], "float32"),
            }
        }
    )
    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_chunk():
    class Model(Module):
        def test(self, x: Tensor):
            chunk = op.chunk(x, chunks=4)
            return chunk

    @R.function
    def test(
        x: R.Tensor((8,), dtype="float32"), _io: R.Object
    ) -> R.Tuple(
        R.Tuple(
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
        ),
        R.Tuple(R.Object),
    ):
        with R.dataflow():
            chunk: R.Tuple(
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
            ) = R.split(x, indices_or_sections=4, axis=0)
            chunk_0: R.Tensor((2,), dtype="float32") = chunk[0]
            chunk_1: R.Tensor((2,), dtype="float32") = chunk[1]
            chunk_2: R.Tensor((2,), dtype="float32") = chunk[2]
            chunk_3: R.Tensor((2,), dtype="float32") = chunk[3]
            gv1: R.Tuple(
                R.Tuple(
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                ),
                R.Tuple(R.Object),
            ) = (chunk_0, chunk_1, chunk_2, chunk_3), (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([8], "float32")}})
    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_nn():
    class Model(Module):
        def test(self, x: Tensor, weight: Tensor, bias: Tensor):
            silu_out = op.silu(x)
            gelu_out = op.gelu(x)
            softmax_out = op.softmax(x, axis=2)
            rms_norm_out = op.rms_norm(x, weight, axes=[-2, -1])
            rms_norm_with_bias_out = op.rms_norm(x, weight, axes=[-2, -1])
            group_norm_out = op.group_norm(x, num_groups=1, weight=bias, bias=bias)
            return x

    @R.function
    def test(
        x: R.Tensor((2, 3, 4, 5), dtype="float32"),
        weight: R.Tensor((4, 5), dtype="float32"),
        bias: R.Tensor((3,), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            silu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.silu(x)
            gelu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.gelu(x)
            softmax: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.softmax(x, axis=2)
            rms_norm: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05
            )
            rms_norm1: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05
            )
            group_norm: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.group_norm(
                x, bias, bias, num_groups=1, channel_axis=1, axes=[2, 3]
            )
            gv1: R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Object)) = x, (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, params = m.export_tvm(
        spec={
            "test": {
                "x": spec.Tensor([2, 3, 4, 5], "float32"),
                "weight": spec.Tensor([4, 5], "float32"),
                "bias": spec.Tensor([3], "float32"),
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


def test_timestep_embedding():
    class Model(Module):
        def test(self, x: Tensor):
            get_timestep_out = op.get_timestep_embedding(x, 10)
            return get_timestep_out

    @R.function
    def test(
        x: R.Tensor((3,), dtype="float32"), _io: R.Object
    ) -> R.Tuple(R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            lv1: R.Tensor((3,), dtype="float32") = R.astype(x, dtype="float32")
            lv2: R.Tensor((3, 1), dtype="float32") = R.expand_dims(lv1, axis=[1])
            lv3: R.Tensor((5,), dtype="float32") = R.arange(
                R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="float32"
            )
            lv4: R.Tensor((5,), dtype="float32") = R.multiply(
                R.const(-9.2103404998779297, "float32"), lv3
            )
            lv5: R.Tensor((5,), dtype="float32") = R.divide(lv4, R.const(4, "float32"))
            lv6: R.Tensor((5,), dtype="float32") = R.exp(lv5)
            lv7: R.Tensor((1, 5), dtype="float32") = R.expand_dims(lv6, axis=[0])
            lv8: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, lv7)
            lv9: R.Tensor((3, 5), dtype="float32") = R.sin(lv8)
            lv10: R.Tensor((3, 5), dtype="float32") = R.cos(lv8)
            get_timestep_embedding: R.Tensor((3, 10), dtype="float32") = R.concat(
                (lv9, lv10), axis=-1
            )
            gv1: R.Tuple(
                R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Object)
            ) = get_timestep_embedding, (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([3], "float32")}})
    tvm.ir.assert_structural_equal(irmodule["test"], test)


def test_scaled_dot_product_attention():
    class Model(Module):
        def test(self, query: Tensor, key: Tensor, value: Tensor):
            scaled_dot_product_attention = op.scaled_dot_product_attention(query, key, value)
            return scaled_dot_product_attention

    @R.function
    def test(
        query: R.Tensor((1, 32, 32, 32), dtype="float32"),
        key: R.Tensor((1, 32, 32, 32), dtype="float32"),
        value: R.Tensor((1, 32, 32, 32), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            scaled_dot_product_attention: R.Tensor(
                (1, 32, 32, 32), dtype="float32"
            ) = R.nn.attention(query, key, value, scale=None, causal_mask=None)
            gv1: R.Tuple(
                R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tuple(R.Object)
            ) = scaled_dot_product_attention, (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "query": spec.Tensor([1, 32, 32, 32], "float32"),
                "key": spec.Tensor([1, 32, 32, 32], "float32"),
                "value": spec.Tensor([1, 32, 32, 32], "float32"),
            }
        }
    )
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


def test_print():
    class Model(Module):
        def test(self, x: Tensor):
            z = op.add(x, x)
            op.print_(z)
            return x

    # fmt: off
    @I.ir_module
    class Expected:
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
            with R.dataflow():
                add: R.Tensor((10, 10), dtype="float32") = R.add(x, x)
                _io1: R.Object = R.call_pure_packed("effect.print", _io, add, sinfo_args=(R.Object(),))
                gv1: R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Object)) = x, (_io1,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(spec={"test": {"x": spec.Tensor([10, 10], "float32")}})

    tvm.ir.assert_structural_equal(irmodule["test"], Expected["test"])


if __name__ == "__main__":
    tvm.testing.main()
