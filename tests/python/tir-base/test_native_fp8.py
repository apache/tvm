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


def create_quantize_func(
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
        quantize_func = quantize_fp8x4_e4m3
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


def quantize_fp8x4_e4m3(  # pylint: disable=too-many-locals
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
            lambda *idx: te.max(max_abs(*idx).astype(model_dtype) / max_int, min_scaling_factor),
            name="scale",
        )
        return scale

    def compute_quantize_weight(bb: relax.BlockBuilder, args: relax.expr.Expr):
        # compute scaled weight
        packed_shape = (weight_shape[0], weight_shape[1] // num_elem_per_storage)
        quant = quant_and_pack_fp8x4_e4m3_sm90(
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
        # import ipdb

        # ipdb.set_trace()

        global_var = bb.add_func(quant, "quantized_weight")
        lv_quantized_weight = bb.emit(
            relax.call_tir(global_var, args, relax.TensorStructInfo(packed_shape, storage_dtype))
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


def quant_and_pack_fp8x4_e4m3_sm90(
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
        for i0, i1 in T.grid(T.int64(weight_shape[0]), T.int64(weight_shape[1])):
            with T.block("compute"):
                v_i0 = T.axis.spatial(T.int64(weight_shape[0]), i0)
                v_i1 = T.axis.spatial(T.int64(weight_shape[1] // vector_length), i1)
                T.reads(
                    A[v_i0, v_i1 : v_i1 + vector_length],
                    scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                )
                T.writes(compute[v_i0, v_i1 * vector_length])
                compute[v_i0, v_i1 * vector_length] = T.reinterpret(
                    storage_dtype,
                    T.Cast(
                        vec_quantized_dtype,
                        # Note: Using the colon here is a sugared way of writing T.ramp(v_i1, 1, vector_length)
                        # ie a vector load of A
                        A[v_i0, v_i1 : v_i1 + vector_length]
                        / scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)],
                    ),
                )

    quant_pack.show()
    return quant_pack


def dequant_fp8x4_e4m3_sm90(
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
                T.writes(dequantize[v_i0, v_i1 : v_i1 + vector_length])
                dequantize[v_i0, v_i1 : v_i1 + vector_length] = T.Cast(
                    vec_model_dtype, T.reinterpret(vec_quantized_dtype, packed_weight[v_i0, v_i1])
                ) * T.Broadcast(
                    scale[v_i0, v_i1 * T.int64(vector_length) // T.int64(group_size)], vector_length
                )

    dequant.show()

    return dequant


@tvm.testing.requires_cuda_compute_version(9)
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


@tvm.testing.requires_cuda_compute_version(9)
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


@tvm.testing.requires_cuda_compute_version(9)
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


@tvm.testing.requires_cuda_compute_version(8)
def test_weight_scale():
    weight_shape = [32000, 4096]
    group_size = 32
    axis = 1
    scale_shape = [d // group_size if axis == i else d for i, d in enumerate(weight_shape)]
    model_dtype = "float16"
    storage_dtype = "uint32"
    quantized_dtype = "e4m3_float8"

    # q_weight = fp8(weight_f16 / scale_f16)
    # q_weight = fp8x4(weight_f16x4 / scale_f16x4)
    vector_length = 4
    vec_quantized_dtype = "e4m3_float8x4"
    vec_model_dtype = "float16x4"
    num_el_per_storage = 4

    @T.prim_func
    def vectorized(
        A: T.Buffer(weight_shape, model_dtype),
        scale: T.Buffer(scale_shape, model_dtype),
        compute: T.Buffer(
            (T.int64(weight_shape[0]), T.int64(weight_shape[1] // num_el_per_storage)),
            storage_dtype,
        ),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        # test = T.alloc_buffer(1, dtype=vec_model_dtype, scope="local")
        for i0, i1 in T.grid(T.int64(weight_shape[0]), T.int64(weight_shape[1])):
            with T.block("compute"):
                v_i0 = T.axis.spatial(T.int64(weight_shape[0]), i0)
                v_i1 = T.axis.spatial(T.int64(weight_shape[1] // vector_length), i1)
                T.reads(
                    A[v_i0, v_i1 : v_i1 + vector_length], scale[v_i0, v_i1 // T.int64(group_size)]
                )
                T.writes(compute[v_i0, v_i1 * vector_length])
                compute[v_i0, v_i1 * vector_length] = T.reinterpret(
                    storage_dtype,
                    T.Cast(
                        vec_quantized_dtype,
                        # Note: Using the colon here is a sugared way of writing T.ramp(v_i1, 1, vector_length)
                        # ie a vector load of A
                        A[v_i0, v_i1 : v_i1 + vector_length]
                        / scale[v_i0, v_i1 // T.int64(group_size)],
                    ),
                )

    sch = tvm.tir.Schedule(vectorized)
    block = sch.get_block("compute")
    loops = sch.get_loops(block)
    txo, txi = sch.split(loops[0], factors=[None, 256])
    sch.bind(loops[1], "blockIdx.x")
    sch.bind(txi, "threadIdx.x")
    sch.mod.show()

    # sch = tvm.tir.Schedule(main)
    # block = sch.get_block("compute")
    # loops = sch.get_loops(block)
    # bx, tx, lanes = sch.split(loops[-1], factors=[None, 32, 4])
    # w_l = sch.cache_read(block, 0, storage_scope="local")
    # # s_l = sch.cache_read(block, 1, storage_scope="local")
    # sch.compute_at(block=w_l, loop=tx)
    # # sch.compute_at(block=s_l, loop=tx)
    # sch.bind(bx, "blockIdx.x")
    # sch.bind(tx, "threadIdx.x")
    # # sch.vectorize(lanes)
    # sch.mod.show()

    import ipdb

    ipdb.set_trace()
    target = "cuda"
    f = tvm.build(sch.mod, target=target)
    print(f.imported_modules[0].get_source())


weight_shape = tvm.testing.parameter([32000, 4096], [4096, 14336])


@tvm.testing.requires_cuda_compute_version(8)
def test_fp8_e4_quant_weight(weight_shape):
    group_size = 32
    axis = 1
    scale_shape = [d // group_size if axis == i else d for i, d in enumerate(weight_shape)]
    model_dtype = "float16"
    storage_dtype = "uint32"
    quantize_dtype = "e4m3_float8"
    num_el_per_storage = 4

    # TODO(csullivan): check this
    max_int_value = 448 if "e4m3" in quantize_dtype else 57344

    mod = create_quantize_func(
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

    target_str = "cuda"
    target = tvm.target.Target(target_str)
    dev = tvm.device(target_str, 0)
    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)

    mod.show()

    f = tvm.build(mod["compute_scale"], target=target)
    cuda_src = f.imported_modules[0].get_source()
    print(cuda_src)

    ex = relax.build(mod, target=target)

    vm = relax.VirtualMachine(ex, dev)  # pylint: disable=invalid-name

    weight_np = np.random.uniform(-100, 100, weight_shape).astype(model_dtype)
    weight = tvm.nd.array(weight_np, device=dev)
    quant_weight, scales = vm["main"](weight)
    quant_weight_np, scales_np = quant_weight.numpy(), scales.numpy()

    print(quant_weight_np, scales_np)


if __name__ == "__main__":
    tvm.testing.main()
