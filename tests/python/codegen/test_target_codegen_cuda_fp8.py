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
# ruff: noqa: F401, F841

from itertools import product

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.testing import env

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


@pytest.mark.parametrize(
    "input",
    [
        ("float8_e4m3fn", "__nv_fp8_e4m3"),
        ("float8_e5m2", "__nv_fp8_e5m2"),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_fp8_conversions(input):
    dtype, nv_dtype = input

    def _create_mod(dtype):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((64,), dtype),
                B: T.Buffer((64,), dtype),
                C: T.Buffer((64,), dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i_0 in T.thread_binding(2, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                        C[i_0 * 32 + i_1] = T.Cast(
                            dtype,
                            T.Cast("float16", A[i_0 * 32 + i_1])
                            + T.Cast("float16", B[i_0 * 32 + i_1]),
                        )

        return Module

    mod = _create_mod(dtype)
    target = "cuda"
    fadd = tvm.tirx.build(mod, target=target)

    cuda_src = fadd.imports[0].inspect_source()
    assert nv_dtype in cuda_src, f"{nv_dtype} datatype not found in generated CUDA"

    dev = tvm.cuda(0)

    a = tvm.runtime.tensor(np.random.uniform(low=0, high=5, size=64).astype(dtype), dev)
    b = tvm.runtime.tensor(np.random.uniform(low=0, high=5, size=64).astype(dtype), dev)
    c = tvm.runtime.tensor(np.zeros(64, dtype=dtype), dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype("float16"), (a.numpy() + b.numpy()).astype("float16")
    )


@pytest.mark.parametrize(
    "dtype",
    ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_fp8_packing(dtype):
    length = 64
    vector_length = 4
    native_dtype, packed_dtype = (f"{dtype}x{vector_length}", "uint32")

    def _create_mod(native_dtype, packed_dtype, length):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((length,), native_dtype),
                R: T.Buffer((length,), packed_dtype),
                B: T.Buffer((length,), native_dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i_0 in T.thread_binding(2, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                        R[i_0 * 32 + i_1] = T.reinterpret(packed_dtype, A[i_0 * 32 + i_1])
                for i_0 in T.thread_binding(2, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                        B[i_0 * 32 + i_1] = T.reinterpret(native_dtype, R[i_0 * 32 + i_1])

        return Module

    mod = _create_mod(native_dtype, packed_dtype, length)
    target = "cuda"
    f = tvm.compile(mod, target=target)
    dev = tvm.cuda(0)

    np_shape = (length, vector_length)
    a_np = np.random.uniform(low=0, high=5, size=np_shape).astype(dtype)
    a = tvm.runtime.empty(shape=(length,), dtype=native_dtype, device=dev)
    r = tvm.runtime.empty(shape=(length,), dtype=packed_dtype, device=dev)
    b = tvm.runtime.empty(shape=(length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    f(a, r, b)
    tvm.testing.assert_allclose(a.numpy().astype("float16"), b.numpy().astype("float16"))


@pytest.mark.parametrize(
    "native_dtype,promoted_dtype,numpytype",
    [
        ("float8_e4m3fn", "float32", "float8_e4m3fn"),
        ("float8_e4m3fn", "float16", "float8_e4m3fn"),
        ("float8_e4m3fnx2", "float32x2", "float8_e4m3fn"),
        ("float8_e4m3fnx2", "float16x2", "float8_e4m3fn"),
        ("float8_e4m3fnx4", "float32x4", "float8_e4m3fn"),
        # Supported via half4 vector type extension in codegen
        ("float8_e4m3fnx4", "float16x4", "float8_e4m3fn"),
        ("float8_e5m2", "float32", "float8_e5m2"),
        ("float8_e5m2", "float16", "float8_e5m2"),
        ("float8_e5m2x2", "float32x2", "float8_e5m2"),
        ("float8_e5m2x2", "float16x2", "float8_e5m2"),
        ("float8_e5m2x4", "float32x4", "float8_e5m2"),
        ("float8_e5m2x4", "float16x4", "float8_e5m2"),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_fp8_vector_conversions(native_dtype, promoted_dtype, numpytype):
    vector_length = 64

    def _create_mod(native_dtype, promoted_dtype):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((64,), native_dtype),
                B: T.Buffer((64,), native_dtype),
                C: T.Buffer((64,), native_dtype),
            ):
                T.func_attr({"tirx.noalias": True})
                for i_0 in T.thread_binding(2, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                        C[i_0 * 32 + i_1] = T.Cast(
                            native_dtype,
                            T.Cast(promoted_dtype, A[i_0 * 32 + i_1])
                            + T.Cast(promoted_dtype, B[i_0 * 32 + i_1]),
                        )

        return Module

    mod = _create_mod(native_dtype, promoted_dtype)
    target = "cuda"
    fadd = tvm.tirx.build(mod, target=target)
    cuda_src = fadd.imports[0].inspect_source()
    dev = tvm.cuda(0)

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
    a = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(low=0, high=5, size=np_shape).astype(numpytype)
    b = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.runtime.empty(shape=(vector_length,), dtype=native_dtype, device=dev)
    fadd(a, b, c)

    tvm.testing.assert_allclose(
        c.numpy().astype(promoted_base_dtype), (a_np + b_np).astype(promoted_base_dtype)
    )


bcast_length = tvm.testing.parameter(2, 4, 6, 8)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="need cuda compute >= 8.0")
def test_half_broadcast(bcast_length):
    dtype = "float16"

    def _create_mod(bcast_length, dtype):
        @I.ir_module
        class Module:
            @T.prim_func
            def main(a: T.Buffer((), dtype), vec: T.Buffer((bcast_length,), dtype)):
                for i_0 in T.thread_binding(1, thread="blockIdx.x"):
                    for i_1 in T.thread_binding(1, thread="threadIdx.x"):
                        vec[0:bcast_length] = T.broadcast(a[()], bcast_length)

        return Module

    mod = _create_mod(bcast_length, dtype)
    target = "cuda"
    func = tvm.compile(mod, target=target)
    dev = tvm.cuda(0)

    a_np = np.random.uniform(low=0, high=4, size=()).astype(dtype)
    a = tvm.runtime.tensor(a_np, device=dev)
    b = tvm.runtime.empty((bcast_length,), dtype=dtype, device=dev)

    func(a, b)

    b_np = np.full((bcast_length,), a_np)

    tvm.testing.assert_allclose(b.numpy(), b_np)


vector_length = tvm.testing.parameter(2, 4)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="need cuda compute >= 8.0")
def test_half_misaligned_vector_load(vector_length):
    dtype = "float16"
    vec_dtype = dtype + "x" + str(vector_length)
    length = 256

    @T.prim_func
    def vector_load(
        A: T.Buffer((length,), dtype), B: T.Buffer((length // vector_length,), vec_dtype)
    ):
        for b in T.thread_binding(1, thread="blockIdx.x"):
            for i in T.thread_binding(length // vector_length, thread="threadIdx.x"):
                vec_index = T.ramp((i + 1) * vector_length - 1, -1, vector_length)
                B[i] = A[vec_index]

    target = "cuda"
    f = tvm.compile(vector_load, target=target)

    dev = tvm.cuda(0)
    a_np = np.random.uniform(low=0, high=1, size=(length,)).astype(dtype)
    a = tvm.runtime.tensor(a_np, device=dev)

    b = tvm.runtime.empty((length // vector_length,), dtype=vec_dtype, device=dev)

    f(a, b)

    b_np = np.empty((length // vector_length, vector_length), dtype=dtype)

    for i in range(length // vector_length):
        start_index = (i + 1) * vector_length - 1
        b_np[i, :] = a_np[start_index - vector_length + 1 : start_index + 1][::-1]

    tvm.testing.assert_allclose(b.numpy(), b_np)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8), reason="need cuda compute >= 8.0")
def test_half4_vector_add():
    dtype = "float16"
    length = 64
    vector_length = 4
    vec_dtype = dtype + "x" + str(vector_length)

    @I.ir_module
    class Module:
        @T.prim_func
        def main(
            A: T.Buffer((64,), "float16x4"),
            B: T.Buffer((64,), "float16x4"),
            C: T.Buffer((64,), "float16x4"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i_0 in T.thread_binding(2, thread="blockIdx.x"):
                for i_1 in T.thread_binding(32, thread="threadIdx.x"):
                    C[i_0 * 32 + i_1] = A[i_0 * 32 + i_1] + B[i_0 * 32 + i_1]

    target = "cuda"
    fadd = tvm.compile(Module, target=target)
    dev = tvm.cuda(0)

    a_np = np.random.uniform(-1, 1, (length, vector_length)).astype(dtype)
    a = tvm.runtime.empty(shape=(length,), dtype=vec_dtype, device=dev)
    a.copyfrom(a_np)
    b_np = np.random.uniform(-1, 1, (length, vector_length)).astype(dtype)
    b = tvm.runtime.empty(shape=(length,), dtype=vec_dtype, device=dev)
    b.copyfrom(b_np)
    c = tvm.runtime.empty(shape=(length,), dtype=vec_dtype, device=dev)

    fadd(a, b, c)
    c_expected = a_np + b_np
    tvm.testing.assert_allclose(c.numpy(), c_expected, atol=1e-5, rtol=1e-5)


class BaseFP8E4M3QuantScaleOnly:
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
        assert axis == 1 and num_el_per_storage == 4
        rows, columns = weight_shape
        groups = scales_shape[1]
        packed_columns = quant_weight_shape[1]
        vec_model_dtype = f"{model_dtype}x4"
        vec_quantized_dtype = f"{quantize_dtype}x4"

        @T.prim_func
        def quantize(
            A: T.Buffer(weight_shape, model_dtype),
            packed: T.Buffer(quant_weight_shape, storage_dtype),
            scale: T.Buffer(scales_shape, model_dtype),
        ):
            for row in T.thread_binding(rows, thread="blockIdx.x"):
                for group in T.thread_binding(groups, thread="threadIdx.x"):
                    maximum = T.alloc_buffer((1,), model_dtype, scope="local")
                    maximum[0] = T.Cast(model_dtype, 0)
                    for k in range(group_size):
                        if group * group_size + k < columns:
                            maximum[0] = T.max(maximum[0], T.abs(A[row, group * group_size + k]))
                    scale[row, group] = T.max(
                        maximum[0] / T.Cast(model_dtype, max_int_value),
                        T.Cast(model_dtype, 1.0 / (max_int_value * 512.0)),
                    )
                    for k in range(group_size // 4):
                        packed[row, group * (group_size // 4) + k] = T.reinterpret(
                            storage_dtype,
                            T.Cast(
                                vec_quantized_dtype,
                                A[row, T.ramp(group * group_size + k * 4, 1, 4)]
                                / scale[row, group],
                            ),
                        )

        @T.prim_func
        def dequantize(
            packed: T.Buffer(quant_weight_shape, storage_dtype),
            scale: T.Buffer(scales_shape, model_dtype),
            output: T.Buffer(weight_shape, model_dtype),
        ):
            for row in T.thread_binding(rows, thread="blockIdx.x"):
                for k in T.thread_binding(packed_columns, thread="threadIdx.x"):
                    output[row, T.ramp(k * 4, 1, 4)] = T.Cast(
                        vec_model_dtype, T.reinterpret(vec_quantized_dtype, packed[row, k])
                    ) * T.Broadcast(scale[row, k * 4 // group_size], 4)

        quant_func = tvm.compile(quantize, target=target_str)
        dequant_func = tvm.compile(dequantize, target=target_str)

        def quant(weight):
            packed = tvm.runtime.empty(quant_weight_shape, storage_dtype, dev)
            scales = tvm.runtime.empty(scales_shape, model_dtype, dev)
            quant_func(weight, packed, scales)
            return packed, scales

        def dequant(packed, scales):
            output = tvm.runtime.empty(weight_shape, model_dtype, dev)
            dequant_func(packed, scales, output)
            return output

        return quant, dequant


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
        return "float8_e4m3fn"

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
        dev = tvm.cuda(0)
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

    @pytest.mark.gpu
    @pytest.mark.skipif(not env.has_cuda_compute(8, 9), reason="need cuda compute >= 8.9")
    def test_main(self, weight_shape, model_dtype, target_str, compiled_functions):
        quant, dequant = compiled_functions
        dev = tvm.cuda(0)

        weight_np = np.random.uniform(-100, 100, weight_shape).astype(model_dtype)
        weight = tvm.runtime.tensor(weight_np, device=dev)
        quant_weight, scales = quant(weight)
        quant_weight_np, scales_np = quant_weight.numpy(), scales.numpy()

        dequant_weight = dequant(quant_weight, scales)
        dequant_weight_np = dequant_weight.numpy()
        tvm.testing.assert_allclose(weight_np, dequant_weight_np, atol=10, rtol=5e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
@pytest.mark.parametrize("dtype", ["float8_e5m2", "float8_e4m3fn", "float8_e8m0fnu"])
def test_const(dtype):
    @T.prim_func
    def func(A: T.Buffer((4,), dtype)) -> None:
        A_local = T.alloc_buffer((4,), dtype=dtype, scope="local")
        for tx in T.thread_binding(0, 4, "threadIdx.x"):
            for i in T.vectorized(4):
                A_local[i] = T.float32(1.0).astype(dtype)
            A[tx] = A_local[tx]

    mod = tvm.IRModule({"main": func})
    tvm.compile(mod, target="cuda")


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8, 9), reason="need cuda compute >= 8.9")
@pytest.mark.parametrize("dtype", ["float8_e5m2", "float8_e4m3fn"])
@pytest.mark.parametrize("vec_len", [2, 4, 8, 16])
def test_copy(dtype, vec_len):
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
    rtmod = tvm.compile(mod, target="cuda")


num_experts = 8
reduce_size = 1792
spatial_size = 4096


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes to be installed")
def test_moe_gemv_shfl_down_illegal_instr():
    global num_experts
    global reduce_size
    global spatial_size

    @T.prim_func
    def moe_dequantize_gemv(
        x: T.Buffer((1, reduce_size), "float16"),
        indptr: T.Buffer((1, 2), "int32"),
        w: T.Buffer((num_experts, spatial_size, reduce_size), "float8_e4m3fn"),
        scale: T.Buffer((1,), "float32"),
        output: T.Buffer((2, spatial_size), "float16"),
    ):
        for expert in T.thread_binding(2, thread="blockIdx.y"):
            for block in T.thread_binding(spatial_size // 4, thread="blockIdx.x"):
                for spatial in T.thread_binding(4, thread="threadIdx.y"):
                    for reduction in T.thread_binding(64, thread="threadIdx.x"):
                        partial = T.alloc_buffer((1,), "float16", scope="local")
                        reduced = T.alloc_buffer((1,), "float16", scope="local")
                        partial[0] = T.float16(0)
                        for k in range(reduce_size // 64):
                            partial[0] = partial[0] + x[0, k * 64 + reduction] * (
                                T.Cast(
                                    "float16",
                                    w[indptr[0, expert], block * 4 + spatial, k * 64 + reduction],
                                )
                                * T.Cast("float16", scale[0])
                            )
                        with T.attr(
                            T.comm_reducer(lambda x, y: x + y, [T.float16(0)]), "reduce_scope", 0
                        ):
                            T.tvm_thread_allreduce(
                                T.uint32(1), partial[0], True, reduced[0], reduction
                            )
                        if reduction == 0:
                            output[expert, block * 4 + spatial] = reduced[0]

    rt_mod = tvm.compile(moe_dequantize_gemv, target="cuda")

    x_data = np.zeros((1, reduce_size), dtype=np.float16)
    indptr_data = np.zeros((1, 2), dtype=np.int32)
    weight_data = np.zeros((num_experts, spatial_size, reduce_size), dtype="float8_e4m3fn")
    scale_data = np.zeros((1,), dtype=np.float32)

    def run_and_check():
        dev = tvm.cuda(0)
        x = tvm.runtime.tensor(x_data, device=dev)
        indptr = tvm.runtime.tensor(indptr_data, device=dev)
        weight = tvm.runtime.tensor(weight_data, device=dev)
        scale = tvm.runtime.tensor(scale_data, device=dev)
        output = tvm.runtime.empty((2, spatial_size), "float16", dev)
        # Exercise shuffle reduction with spatial/reduction thread extents 4 and 64.
        rt_mod(x, indptr, weight, scale, output)
        tvm.testing.assert_allclose(output.numpy(), np.zeros((2, spatial_size)))
        dev.sync()

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("vec_length", [2, 4])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(8, 9), reason="need cuda compute >= 8.9")
def test_fp8_fp16_bf16_vectorize_arith(vec_length, dtype):
    def _create_mod(vec_length, dtype):
        num_threads = 128 // vec_length

        @I.ir_module
        class Module:
            @T.prim_func
            def main(
                A: T.Buffer((128,), "float8_e4m3fn"),
                B: T.Buffer((128,), dtype),
                C: T.Buffer((128,), dtype),
            ) -> None:
                for i_0 in T.thread_binding(num_threads, thread="threadIdx.x"):
                    for i_1 in T.vectorized(vec_length):
                        C[i_0 * vec_length + i_1] = A[i_0 * vec_length + i_1].astype(dtype) * B[
                            i_0 * vec_length + i_1
                        ] + T.bfloat16(3.0)

        return Module

    mod = _create_mod(vec_length, dtype)
    target = tvm.target.Target.from_device(tvm.cuda())
    f = tvm.tirx.build(mod, target=target)

    a_np = np.random.rand(128).astype("float8_e4m3fn")
    b_np = np.random.rand(128).astype(dtype)
    c_np = (a_np.astype(dtype) * b_np) + 3

    def run_and_check():
        device = tvm.cuda()
        a_tvm = tvm.runtime.tensor(a_np, device=device)
        b_tvm = tvm.runtime.tensor(b_np, device=device)
        c_tvm = tvm.runtime.empty((128,), dtype=dtype, device=device)
        f(a_tvm, b_tvm, c_tvm)
        actual = c_tvm.numpy()
        tvm.testing.assert_allclose(
            actual.astype(np.float32), c_np.astype(np.float32), atol=5e-1, rtol=1e-2
        )

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
