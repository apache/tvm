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

import sys
import pytest

import tvm
from tvm.script import tir as T
import numpy as np
import tvm.testing


from typing import List, Tuple
from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir, topi
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.topi.utils import get_const_tuple


@tvm.testing.requires_cuda_compute_version(8, 9)
def test_e4m3_conversions():
    dtype = "e4m3_float8"

    @T.prim_func
    def add(
        A: T.Buffer((64,), dtype),
        B: T.Buffer((64,), dtype),
        C: T.Buffer((64,), dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(64):
            with T.block("C"):
                v_i = T.axis.spatial(64, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(dtype, T.Cast("float16", A[v_i]) + T.Cast("float16", B[v_i]))

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)

    cuda_src = fadd.imported_modules[0].get_source()
    assert "fp8_e4_t" in cuda_src, "FP8E4M3 (fp8_e4_t) datatype not found in generated CUDA"

    dev = tvm.device(target, 0)

    numpytype = "float8_e4m3fn"
    a = tvm.nd.array(np.random.uniform(low=0, high=5, size=64).astype(numpytype), dev)
    b = tvm.nd.array(np.random.uniform(low=0, high=5, size=64).astype(numpytype), dev)
    c = tvm.nd.array(np.zeros(64, dtype=numpytype), dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype("float16"), (a.numpy() + b.numpy()).astype("float16")
    )


@tvm.testing.requires_cuda_compute_version(8, 9)
def test_e4m3_packing():
    length = 64
    vector_length = 4
    native_dtype, packed_dtype = ("e4m3_float8x4", "uint32")

    @T.prim_func
    def add(
        A: T.Buffer((length,), native_dtype),
        R: T.Buffer((length,), packed_dtype),
        B: T.Buffer((length,), native_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(length):
            with T.block("R"):
                v_i = T.axis.spatial(length, i)
                T.reads(A[v_i])
                T.writes(R[v_i])
                R[v_i] = T.reinterpret(packed_dtype, A[v_i])
        for i in range(length):
            with T.block("B"):
                v_i = T.axis.spatial(length, i)
                T.reads(R[v_i])
                T.writes(B[v_i])
                B[v_i] = T.reinterpret(native_dtype, R[v_i])

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("R")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    block = sch.get_block("B")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    f = tvm.build(sch.mod, target=target)
    dev = tvm.device(target, 0)

    numpytype = "float8_e4m3fn"
    np_shape = (length, vector_length)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    a = tvm.nd.empty(shape=(length,), dtype=native_dtype, device=dev)
    r = tvm.nd.empty(shape=(length,), dtype=packed_dtype, device=dev)
    b = tvm.nd.empty(shape=(length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    f(a, r, b)
    tvm.testing.assert_allclose(a.numpy().astype("float16"), b.numpy().astype("float16"))


native_dtype, promoted_dtype = tvm.testing.parameters(
    ("e4m3_float8", "float32"),
    ("e4m3_float8", "float16"),
    ("e4m3_float8x2", "float32x2"),
    ("e4m3_float8x2", "float16x2"),
    ("e4m3_float8x4", "float32x4"),
    # Supported via half4 vector type extension in codegen
    ("e4m3_float8x4", "float16x4"),
)


@tvm.testing.requires_cuda_compute_version(8, 9)
def test_e4m3_vector_conversions(native_dtype, promoted_dtype):
    vector_length = 64

    @T.prim_func
    def add(
        A: T.Buffer((vector_length,), native_dtype),
        B: T.Buffer((vector_length,), native_dtype),
        C: T.Buffer((vector_length,), native_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(vector_length):
            with T.block("C"):
                v_i = T.axis.spatial(vector_length, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = T.Cast(
                    native_dtype, T.Cast(promoted_dtype, A[v_i]) + T.Cast(promoted_dtype, B[v_i])
                )

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)
    cuda_src = fadd.imported_modules[0].get_source()
    dev = tvm.device(target, 0)

    numpytype = "float8_e4m3fn"
    if "x" in native_dtype:
        lanes = int(native_dtype.split("x")[-1])
    else:
        lanes = 1

    if "x" in promoted_dtype:
        promoted_base_dtype = promoted_dtype.split("x")[0]
    else:
        promoted_base_dtype = promoted_dtype

    np_shape = (vector_length, lanes) if lanes > 1 else (vector_length,)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    a = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    b = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.nd.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
    )


bcast_length = tvm.testing.parameter(2, 4, 6, 8)


@tvm.testing.requires_cuda_compute_version(8)
def test_half_broadcast(bcast_length):
    dtype = "float16"

    @T.prim_func
    def vector_broadcast(a: T.Buffer[(), dtype], vec: T.Buffer[(bcast_length,), dtype]):
        for t in range(1):
            with T.block("broadcast"):
                vec[0:bcast_length] = T.broadcast(a[()], bcast_length)

    sch = tvm.tir.Schedule(vector_broadcast)
    block = sch.get_block("broadcast")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 1])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    func = tvm.build(sch.mod, target=target)
    dev = tvm.device(target, 0)

    a_np = np.random.uniform(low=0, high=4, size=()).astype(dtype)
    a = tvm.nd.array(a_np, device=dev)
    b = tvm.nd.empty((bcast_length,), dtype=dtype, device=dev)

    func(a, b)

    b_np = np.full((bcast_length,), a_np)

    tvm.testing.assert_allclose(b.numpy(), b_np)


vector_length = tvm.testing.parameter(2, 4)


@tvm.testing.requires_cuda_compute_version(8)
def test_half_misaligned_vector_load(vector_length):
    dtype = "float16"
    vec_dtype = dtype + "x" + str(vector_length)
    length = 256

    @T.prim_func
    def vector_load(
        A: T.Buffer[(length,), dtype], B: T.Buffer[(length // vector_length,), vec_dtype]
    ):
        for b in T.thread_binding(1, thread="blockIdx.x"):
            for i in T.thread_binding(length // vector_length, thread="threadIdx.x"):
                vec_index = T.ramp((i + 1) * vector_length - 1, -1, vector_length)
                B[i] = A[vec_index]

    target = "cuda"
    f = tvm.build(vector_load, target=target)

    dev = tvm.device(target, 0)
    a_np = np.random.uniform(low=0, high=1, size=(length,)).astype(dtype)
    a = tvm.nd.array(a_np, device=dev)

    b = tvm.nd.empty((length // vector_length,), dtype=vec_dtype, device=dev)

    f(a, b)

    b_np = np.empty((length // vector_length, vector_length), dtype=dtype)

    for i in range(length // vector_length):
        start_index = (i + 1) * vector_length - 1
        b_np[i, :] = a_np[start_index - vector_length + 1 : start_index + 1][::-1]

    tvm.testing.assert_allclose(b.numpy(), b_np)


@tvm.testing.requires_cuda_compute_version(8)
def test_half4_vector_add():
    dtype = "float16"
    length = 64
    vector_length = 4
    vec_dtype = dtype + "x" + str(vector_length)

    @T.prim_func
    def add(
        A: T.Buffer((length,), vec_dtype),
        B: T.Buffer((length,), vec_dtype),
        C: T.Buffer((length,), vec_dtype),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(length):
            with T.block("C"):
                v_i = T.axis.spatial(length, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = A[v_i] + B[v_i]

    sch = tvm.tir.Schedule(add)
    block = sch.get_block("C")
    b = sch.get_loops(block)
    bx, tx = sch.split(b[0], factors=[None, 32])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")

    target = "cuda"
    fadd = tvm.build(sch.mod, target=target)
    dev = tvm.device(target, 0)

    a_np = np.random.uniform(-1, 1, (length, vector_length)).astype(dtype)
    a = tvm.nd.empty(shape=(length,), dtype=vec_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(-1, 1, (length, vector_length)).astype(dtype)
    b = tvm.nd.empty(shape=(length,), dtype=vec_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.nd.empty(shape=(length,), dtype=vec_dtype, device=dev)

    fadd(a, b, c)
    c_expected = a_np + b_np
    tvm.testing.assert_allclose(c.numpy(), c_expected, atol=1e-5, rtol=1e-5)


class BaseFP8E4M3QuantScaleOnly:
    @classmethod
    def create_quantize_func(
        cls,
        weight_shape,
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_elem_per_storage,
        max_int_value,
        axis,
        output_transpose,
    ) -> IRModule:
        if DataType(quantize_dtype).type_code == DataTypeCode.E4M3Float:
            quantize_func = cls.quantize_fp8x4_e4m3
        else:
            assert NotImplementedError()

        bb = relax.BlockBuilder()  # pylint: disable=invalid-name
        weight_var = relax.Var("weight", relax.TensorStructInfo(weight_shape, model_dtype))
        compute_scale, compute_quantize, compute_transpose = quantize_func(
            weight_shape,
            model_dtype,
            quantize_dtype,
            storage_dtype,
            group_size,
            num_elem_per_storage,
            max_int_value,
            axis,
            output_transpose,
        )
        with bb.function(name="main", params=[weight_var]):
            with bb.dataflow():
                lv_scale = bb.emit_te(compute_scale, weight_var)
                lv_quantized_weight = compute_quantize(bb, (weight_var, lv_scale))
                if compute_transpose:
                    lv_output = bb.emit_te(compute_transpose, lv_quantized_weight, lv_scale)
                    lv_quantized_weight = lv_output[0]
                    lv_scale = lv_output[1]
                tuple_output = bb.emit((lv_quantized_weight, lv_scale))
                gv = bb.emit_output(tuple_output)
            bb.emit_func_output(gv)
        return bb.finalize()

    @classmethod
    def create_dequantize_func(
        cls,
        packed_weight_shape,
        scale_shape,
        dequantized_shape,
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_elem_per_storage,
        axis,
    ) -> IRModule:
        if DataType(quantize_dtype).type_code == DataTypeCode.E4M3Float:
            dequantize_func = cls.dequantize_fp8x4_e4m3
        else:
            assert NotImplementedError()

        bb = relax.BlockBuilder()  # pylint: disable=invalid-name
        packed_weight_var = relax.Var(
            "weight", relax.TensorStructInfo(packed_weight_shape, storage_dtype)
        )
        scale_var = relax.Var("scale", relax.TensorStructInfo(scale_shape, model_dtype))
        compute_dequantize = dequantize_func(
            packed_weight_shape,
            scale_shape,
            dequantized_shape,
            model_dtype,
            quantize_dtype,
            storage_dtype,
            group_size,
            num_elem_per_storage,
            axis,
        )
        with bb.function(name="main", params=[packed_weight_var, scale_var]):
            with bb.dataflow():
                lv = compute_dequantize(bb, (packed_weight_var, scale_var))
                gv = bb.emit_output(lv)
            bb.emit_func_output(gv)
        return bb.finalize()

    @classmethod
    def quantize_fp8x4_e4m3(  # pylint: disable=too-many-locals
        cls,
        weight_shape: List[tir.PrimExpr],
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_elem_per_storage,
        max_int_value,
        axis: int = -1,
        output_transpose: bool = False,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        max_int = tir.const(max_int_value, model_dtype)
        shape = weight_shape  # pylint: disable=invalid-name
        axis = axis if axis >= 0 else len(shape) + axis
        k = shape[axis]
        quantize_dtype = DataType(quantize_dtype)
        # compute scale per group
        r = te.reduce_axis((0, group_size), name="r")  # pylint: disable=invalid-name
        num_group = tir.ceildiv(k, group_size)
        # (4096, 4096) -> quantize axis = 0, group size = 32 -> (128, 4096)
        # for channel quant group_size = 4096 -> (1, 4096)
        scale_shape = (*shape[:axis], num_group, *shape[axis + 1 :])

        def compute_scale(weight: te.Tensor):
            min_scaling_factor = tir.const(1.0 / (max_int_value * 512.0), model_dtype)
            max_abs = te.compute(
                shape=scale_shape,
                fcompute=lambda *idx: te.max(
                    tir.if_then_else(
                        idx[axis] * group_size + r < k,
                        te.abs(weight(*idx[:axis], idx[axis] * group_size + r, *idx[axis + 1 :])),
                        te.min_value(model_dtype),
                    ),
                    axis=r,
                ),
                name="max_abs_value",
            )
            scale = te.compute(
                scale_shape,
                lambda *idx: te.max(
                    max_abs(*idx).astype(model_dtype) / max_int, min_scaling_factor
                ),
                name="scale",
            )
            return scale

        def compute_quantize_weight(bb: relax.BlockBuilder, args: relax.expr.Expr):
            # compute scaled weight
            packed_shape = (weight_shape[0], weight_shape[1] // num_elem_per_storage)
            quant = cls.quant_and_pack_fp8x4_e4m3_sm90(
                weight_shape,
                packed_shape,
                scale_shape,
                group_size,
                axis,
                model_dtype,
                storage_dtype,
                quantize_dtype,
            )
            # quant.show()

            global_var = bb.add_func(quant, "quantized_weight")
            lv_quantized_weight = bb.emit(
                relax.call_tir(
                    global_var, args, relax.TensorStructInfo(packed_shape, storage_dtype)
                )
            )
            return lv_quantized_weight

        compute_transpose = None
        if output_transpose:

            def compute_transpose(quantized_weight: te.Tensor, scale: te.Tensor):
                if len(quantized_weight.shape) != 2 or len(scale.shape) != 2:
                    raise ValueError(
                        "Does not support transpose output quantized weight with ndim != 2"
                    )

                quantized_weight = topi.transpose(quantized_weight)
                scale = topi.transpose(scale)
                return quantized_weight, scale

        return compute_scale, compute_quantize_weight, compute_transpose

    @classmethod
    def dequantize_fp8x4_e4m3(  # pylint: disable=too-many-locals
        cls,
        packed_weight_shape: List[tir.PrimExpr],
        scale_shape,
        dequant_shape,
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_elem_per_storage,
        axis: int = -1,
    ) -> Tuple[te.Tensor, te.Tensor]:
        """Group quantization for weight tensor, defined in tensor expression."""
        axis = axis if axis >= 0 else len(shape) + axis

        def compute_dequantize_weight(bb: relax.BlockBuilder, args: relax.expr.Expr):
            dequant = cls.dequant_fp8x4_e4m3_sm90(
                packed_weight_shape,
                scale_shape,
                dequant_shape,
                group_size,
                axis,
                model_dtype,
                storage_dtype,
                quantize_dtype,
            )

            global_var = bb.add_func(dequant, "dequantize_weight")
            lv_dequantized_weight = bb.emit(
                relax.call_tir(global_var, args, relax.TensorStructInfo(dequant_shape, model_dtype))
            )
            return lv_dequantized_weight

        return compute_dequantize_weight

    @classmethod
    def quant_and_pack_fp8x4_e4m3_sm90(
        cls,
        weight_shape,
        packed_shape,
        scale_shape,
        group_size,
        axis,
        model_dtype,
        storage_dtype,
        quantized_dtype,
    ):
        vector_length = 4
        vec_quantized_dtype = f"{quantized_dtype}x{vector_length}"
        vec_model_dtype = f"{model_dtype}x{vector_length}"
        num_elem_per_storage = vector_length
        # TODO(csullivan) assert on storage dtype / quantize type bytes == vector length
        assert (
            group_size % vector_length == 0
        ), f"Number of elements in a group must be divisible by fp8 vector length {vector_length}"

        @T.prim_func(private=True)
        def quant_pack(
            A: T.Buffer(weight_shape, model_dtype),
            scale: T.Buffer(scale_shape, model_dtype),
            compute: T.Buffer(
                packed_shape,
                storage_dtype,
            ),
        ):
            # with T.block("root"):
            # test = T.alloc_buffer(1, dtype=vec_model_dtype, scope="local")
            for i0, i1 in T.grid(
                T.int64(weight_shape[0]), T.int64(weight_shape[1] // vector_length)
            ):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(
                        A[v_i0, v_i1 : v_i1 + vector_length],
                        scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                    )
                    T.writes(compute[v_i0, v_i1 * vector_length])
                    compute[v_i0, v_i1] = T.reinterpret(
                        storage_dtype,
                        T.Cast(
                            vec_quantized_dtype,
                            A[v_i0, T.ramp(v_i1 * vector_length, 1, vector_length)]
                            / scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                        ),
                    )

        return quant_pack

    @classmethod
    def dequant_fp8x4_e4m3_sm90(
        cls,
        packed_weight_shape,
        scale_shape,
        out_shape,
        group_size,
        axis,
        model_dtype,
        storage_dtype,
        quantized_dtype,
    ):
        vector_length = 4
        vec_quantized_dtype = f"{quantized_dtype}x{vector_length}"
        vec_model_dtype = f"{model_dtype}x{vector_length}"
        num_elem_per_storage = vector_length

        @T.prim_func
        def dequant(
            packed_weight: T.Buffer(packed_weight_shape, storage_dtype),
            scale: T.Buffer(scale_shape, model_dtype),
            dequantize: T.Buffer(out_shape, model_dtype),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(packed_weight_shape[0]), T.int64(packed_weight_shape[1])):
                with T.block("dequantize"):
                    v_i0 = T.axis.spatial(T.int64(packed_weight_shape[0]), i0)
                    v_i1 = T.axis.spatial(T.int64(packed_weight_shape[1]), i1)
                    T.reads(
                        packed_weight[v_i0, v_i1],
                        scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                    )

                    dequantize[v_i0, T.ramp(v_i1 * vector_length, 1, vector_length)] = T.Cast(
                        vec_model_dtype,
                        T.reinterpret(vec_quantized_dtype, packed_weight[v_i0, v_i1]),
                    ) * T.Broadcast(
                        scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                        vector_length,
                    )

        return dequant

    @classmethod
    def compile_quant_and_dequant_by_scale(
        cls,
        weight_shape,
        scales_shape,
        quant_weight_shape,
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_el_per_storage,
        max_int_value,
        axis,
        target_str,
        dev,
    ):
        quant_mod = cls.create_quantize_func(
            weight_shape,
            model_dtype,
            quantize_dtype,
            storage_dtype,
            group_size,
            num_el_per_storage,
            max_int_value,
            axis,
            output_transpose=False,
        )
        # quant_mod.show()

        target = tvm.target.Target(target_str)
        with target:
            quant_mod = dl.ApplyDefaultSchedule(
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(quant_mod)
        ex_1 = relax.build(quant_mod, target=target)
        vm_1 = relax.VirtualMachine(ex_1, dev)

        dequant_mod = cls.create_dequantize_func(
            quant_weight_shape,
            scales_shape,
            weight_shape,
            model_dtype,
            quantize_dtype,
            storage_dtype,
            group_size,
            num_el_per_storage,
            axis,
        )
        # dequant_mod.show()

        with target:
            dequant_mod = dl.ApplyDefaultSchedule(
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(dequant_mod)
        dequant_mod.show()

        ex_2 = relax.build(dequant_mod, target=target)
        vm_2 = relax.VirtualMachine(ex_2, dev)

        def print_cuda(target, mod, name=None):
            if name:
                mod = mod[name]
            f = tvm.build(mod, target=target)
            cuda_src = f.imported_modules[0].get_source()
            print(cuda_src)

        print_cuda(target, dequant_mod, name="dequant")

        return vm_1["main"], vm_2["main"]


class TestFP8e4x4QuantDequantScale(BaseFP8E4M3QuantScaleOnly):
    # weight_shape = tvm.testing.parameter((32000, 4096), (4096, 14336))
    weight_shape = tvm.testing.parameter((128, 256), (128, 64))

    @tvm.testing.fixture
    def group_size(self):
        return 64

    @tvm.testing.fixture
    def axis(self):
        return 1

    @tvm.testing.fixture
    def model_dtype(self):
        return "float16"

    @tvm.testing.fixture
    def storage_dtype(self):
        return "uint32"

    @tvm.testing.fixture
    def quantize_dtype(self):
        return "e4m3_float8"

    @tvm.testing.fixture
    def num_el_per_storage(self):
        return 4

    @tvm.testing.fixture
    def max_int_value(self):
        return 448

    @tvm.testing.fixture
    def target_str(self):
        return "cuda"

    @tvm.testing.fixture
    def scale_shape(self, weight_shape, group_size, axis):
        return [
            (d + group_size - 1) // group_size if axis == i else d
            for i, d in enumerate(weight_shape)
        ]

    @tvm.testing.fixture
    def quant_weight_shape(self, weight_shape, num_el_per_storage, axis):
        return [
            (d + num_el_per_storage - 1) // num_el_per_storage if axis == i else d
            for i, d in enumerate(weight_shape)
        ]

    @tvm.testing.fixture
    def compiled_functions(
        self,
        weight_shape,
        scale_shape,
        quant_weight_shape,
        model_dtype,
        quantize_dtype,
        storage_dtype,
        group_size,
        num_el_per_storage,
        max_int_value,
        axis,
        target_str,
    ):
        dev = tvm.device(target_str, 0)
        return self.compile_quant_and_dequant_by_scale(
            weight_shape,
            scale_shape,
            quant_weight_shape,
            model_dtype,
            quantize_dtype,
            storage_dtype,
            group_size,
            num_el_per_storage,
            max_int_value,
            axis,
            target_str,
            dev,
        )

    @tvm.testing.requires_cuda_compute_version(8, 9)
    def test_main(self, weight_shape, model_dtype, target_str, compiled_functions):
        quant, dequant = compiled_functions
        dev = tvm.device(target_str, 0)

        weight_np = np.random.uniform(-100, 100, weight_shape).astype(model_dtype)
        weight = tvm.nd.array(weight_np, device=dev)
        quant_weight, scales = quant(weight)
        quant_weight_np, scales_np = quant_weight.numpy(), scales.numpy()

        dequant_weight = dequant(quant_weight, scales)
        dequant_weight_np = dequant_weight.numpy()
        tvm.testing.assert_allclose(weight_np, dequant_weight_np, atol=10, rtol=5e-2)


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("dtype", ["e5m2_float8", "e4m3_float8"])
def test_const(dtype):
    @T.prim_func
    def func(A: T.Buffer((4,), dtype)) -> None:
        A_local = T.alloc_buffer((4,), dtype=dtype, scope="local")
        for tx in T.thread_binding(0, 4, "threadIdx.x"):
            for i in T.vectorized(4):
                A_local[i] = T.float32(1.0).astype(dtype)
            A[tx] = A_local[tx]

    mod = tvm.IRModule({"main": func})
    tvm.build(mod, target="cuda")


@tvm.testing.requires_cuda_compute_version(8, 9)
@pytest.mark.parametrize("dtype", ["e5m2_float8", "e4m3_float8"])
@pytest.mark.parametrize("vec_length", [2, 4, 8, 16])
def test_copy(dtype, vec_len=4):
    @T.prim_func
    def func(
        A: T.Buffer(
            (
                4,
                vec_len,
            ),
            dtype,
        ),
        B: T.Buffer(
            (
                4,
                vec_len,
            ),
            dtype,
        ),
    ) -> None:
        for tx in T.thread_binding(0, 4, "threadIdx.x"):
            for i in T.vectorized(vec_len):
                B[tx, i] = A[tx, i]

    mod = tvm.IRModule({"main": func})
    rtmod = tvm.build(mod, target="cuda")

    print(rtmod.imported_modules[0].get_source())


if __name__ == "__main__":
    tvm.testing.main()
