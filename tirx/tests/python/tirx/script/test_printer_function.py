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
# ruff: noqa: E501
"""TIRx function printing and roundtrip coverage."""

import sys

import pytest
from tvm_ffi.access_path import AccessPath

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import IRModule, tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import ir_builder as IB
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as TB


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_function_symbolic_tensor_param_roundtrip():
    n = tirx.Var("n", "int32")
    A = tirx.Var("A", tirx.TensorType([n + 1, n], "float32", layout=None))
    func = tirx.Function(params=[A], body=tvm_tirx.ir.Evaluate(n)).with_attr(
        "global_symbol", "main"
    )

    source = func.script()
    assert "T.Tensor((n + 1, n)" in source
    assert source.index("def main[n](") < source.index("T.evaluate(n)")
    tvm_tirx.ir.assert_structural_equal(
        tvm_tirx.script.from_source(
            source,
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
        ),
        func,
    )


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_function_compound_tensor_shape_first_use_roundtrip():
    n = tirx.Var("n", "int32")
    A = tirx.Var("A", tirx.TensorType([tirx.max(n, 1)], "float32", layout=None))
    func = tirx.Function(params=[A], body=tvm_tirx.ir.Evaluate(n)).with_attr(
        "global_symbol", "main"
    )

    source = func.script()
    assert "T.Tensor((T.max(n, 1),)" in source
    assert source.index("def main[n](") < source.index("T.evaluate(n)")
    tvm_tirx.ir.assert_structural_equal(
        tvm_tirx.script.from_source(
            source,
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
        ),
        func,
    )


def test_function_symbolic_alloc_tensor_roundtrip():
    size = tirx.Var("size", "int32")
    buf = tirx.Var("buf", tirx.TensorType([size], "float32", layout=None))
    func = tirx.Function(
        params=[],
        body=tvm_tirx.ir.SeqStmt(
            [
                tvm_tirx.ir.Bind(
                    buf,
                    tvm_tirx.ir.Call(
                        "tirx.alloc_tensor",
                        [],
                        attrs=tvm_tirx.ir.DictAttrs({}),
                        ty_args=[buf.ty],
                    ),
                ),
                tvm_tirx.ir.Evaluate(tvm_tirx.ir.TensorLoad(buf, [0])),
            ]
        ),
    )

    source = func.script()
    assert "T.alloc_tensor(ty_args=[T.Tensor((size,), T.float32)])" in source
    tvm_tirx.ir.assert_structural_equal(
        tvm_tirx.script.from_source(
            source,
            check_well_formed=False,
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
        ),
        func,
    )


def test_function():
    A = tirx.Var("A", tirx.TensorType([128, 128], "float32"))
    B = tirx.Var("B", tirx.TensorType([256, 256], "float32"))
    func = tirx.Function(
        params=[A, B],
        ret_type=None,
        body=tvm_tirx.ir.Evaluate(0),
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    pass
""",
    )


def test_function_tensor_data_use():
    A = tirx.Var("A", tirx.TensorType([128, 128], "float32"))
    B = tirx.Var("B", tirx.TensorType([256, 256], "float32"))
    func = tirx.Function(
        params=[A, B],
        ret_type=None,
        body=tvm_tirx.ir.Evaluate(A.data),
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    T.evaluate(A.data)
""",
    )


def test_function_tensor_data_argument_is_scope_hint():
    buffer_data = tirx.Var("A", tirx.TensorType([128, 128], "float32")).data
    A = tirx.Var("A", tirx.TensorType([128, 128], "float32", data=buffer_data))
    B = tirx.Var("B", tirx.TensorType([256, 256], "float32", data=buffer_data))
    func = tirx.Function(
        params=[A, B],
        ret_type=None,
        body=tvm_tirx.ir.Evaluate(0),
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    pass
""",
    )


def test_bind():
    with IRBuilder() as ib:
        with TB.function():
            v = TB.bind(TB.float32(10))
            ib.name("v", v)
            TB.evaluate(1)
    obj = ib.get()
    _assert_print(
        obj,
        """
# from tvm_tirx.script import tirx as T

@T.function(private=True)
def main():
    v: T.let[T.float32] = T.float32(10.0)
    T.evaluate(1)
""",
    )


def test_private_function():
    A = tirx.Var("A", tirx.TensorType([128, 128], "float32"))
    B = tirx.Var("B", tirx.TensorType([256, 256], "float32"))
    func = tirx.Function(
        params=[A, B],
        ret_type=None,
        body=tvm_tirx.ir.Evaluate(0),
    )
    _assert_print(
        func,
        expected="""
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function(private=True)
def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    pass
""",
    )


def test_function_different_symbol():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((128, 128), "float32"), B: TB.Tensor((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        TB.evaluate(0)

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def func(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    pass
"""
    _assert_print(main, expected_output)


def test_return_statement():
    from tvm_tirx.script import tirx as TB

    @T.function
    def func():
        return TB.int32(5)

    expected_output = """
# from tvm_tirx.script import tirx as T

@T.function
def func():
    return 5
"""
    _assert_print(func, expected_output)
    assert func.script(verbose_expr=True).strip() == expected_output.strip()


CUSTOM_FLOAT_DTYPES = [
    # Float8 variants
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    # Float6 variants
    "float6_e2m3fn",
    "float6_e3m2fn",
    # Float4 variant
    "float4_e2m1fn",
]


@pytest.mark.parametrize("dtype", CUSTOM_FLOAT_DTYPES)
def test_custom_float_types(dtype):
    from tvm_tirx.script import tirx as TB

    @T.function
    def func():
        TB.evaluate(getattr(TB, dtype)(0.0))

    expected_output = f"""
# from tvm_tirx.script import tirx as T

@T.function
def func():
    T.evaluate(T.{dtype}(0.0))
"""
    _assert_print(func, expected_output)


def test_predicated_load_store():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((128, 128), "float32"), B: TB.Tensor((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        a_load = TB.meta_var(
            TB.call_intrin(
                "tirx.masked_load",
                A,
                0,
                TB.Ramp(0, 4, 4),
                TB.Broadcast(TB.bool(False), 4),
                ty="float32x4",
            )
        )
        TB.evaluate(
            TB.call_intrin(
                "tirx.masked_store",
                A,
                a_load,
                0,
                TB.Ramp(0, 2, 4),
                TB.Broadcast(TB.bool(False), 4),
                ty="void",
            )
        )

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def func(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    a_load: T.let[T.float32x4] = T.masked_load(A, 0, T.Ramp(0, 4, 4), T.Broadcast(T.bool(False), 4))
    T.masked_store(A, a_load, 0, T.Ramp(0, 2, 4), T.Broadcast(T.bool(False), 4))
"""
    _assert_print(main, expected_output)


def test_predicated_tensor_load_store():
    a = tirx.Var("a", "handle")
    b = tirx.Var("b", "handle")
    tensors = {
        a: tirx.Var("A", tirx.TensorType([128, 128], "float32")),
        b: tirx.Var("B", tirx.TensorType([256, 256], "float32")),
    }
    tensor_load = tirx.call_intrin(
        "tirx.masked_load",
        tensors[b],
        0,
        tvm_tirx.ir.prim.vector_expr.Ramp(0, 4, 4),
        tvm_tirx.ir.prim.vector_expr.Broadcast(tvm_tirx.ir.IntImm("bool", 0), 4),
        ty="float32x4",
    )
    body = tvm_tirx.ir.Evaluate(
        tirx.call_intrin(
            "tirx.masked_store",
            tensors[a],
            tensor_load,
            0,
            tvm_tirx.ir.prim.vector_expr.Ramp(0, 2, 4),
            tvm_tirx.ir.prim.vector_expr.Broadcast(tvm_tirx.ir.IntImm("bool", 0), 4),
            ty="void",
        )
    )
    func = tirx.Function(
        params=[tensors[a], tensors[b]],
        ret_type=None,
        body=body,
    )

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function(private=True)
def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    T.masked_store(A, T.masked_load(B, 0, T.Ramp(0, 4, 4), T.Broadcast(T.bool(False), 4)), 0, T.Ramp(0, 2, 4), T.Broadcast(T.bool(False), 4))
"""
    _assert_print(func, expected_output)


def test_predicated_scalable_load_store():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((128, 128), "float32"), B: TB.Tensor((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        mask = TB.meta_var(TB.get_active_lane_mask(0, 13, ty="uint1xvscalex4"))
        a_load = TB.meta_var(
            TB.call_intrin(
                "tirx.masked_load",
                A,
                0,
                TB.Ramp(0, 4, TB.vscale() * 4),
                mask,
                ty="float32xvscalex4",
            )
        )
        TB.evaluate(
            TB.call_intrin(
                "tirx.masked_store", A, a_load, 0, TB.Ramp(0, 2, TB.vscale() * 4), mask, ty="void"
            )
        )

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def func(A: T.Tensor((128, 128), T.float32), B: T.Tensor((256, 256), T.float32)):
    mask: T.let[T.PrimType("uint1xvscalex4")] = T.get_active_lane_mask(0, 13, ty=T.PrimType("uint1xvscalex4"))
    a_load: T.let[T.PrimType("float32xvscalex4")] = T.masked_load(A, 0, T.Ramp(0, 4, T.vscale() * 4), mask)
    T.masked_store(A, a_load, 0, T.Ramp(0, 2, T.vscale() * 4), mask)
"""
    _assert_print(main, expected_output)


def test_masked_load_prevents_scalar_allocation_init_fusion():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main():
        A = TB.alloc_tensor(ty_args=[TB.Tensor((1,), "float32x4")])
        A[0] = TB.masked_load(A, 0, TB.Broadcast(TB.bool(True), 4), ty="float32x4")

    source = main.script()
    assert "A = T.alloc_tensor" in source
    assert "A[0] = T.masked_load" in source
    tvm_tirx.ir.assert_structural_equal(
        tvm_tirx.script.from_source(
            source,
            extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
        ),
        main,
    )


def test_vload_with_explicit_scalable_data_type():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((128,), "float32"), B: TB.Tensor((128,), "float32")):
        B[0 : TB.vscale() * 4] = A[TB.Ramp(0, 1, TB.vscale() * 4)]

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
    B[0:T.vscale() * 4] = A[T.Ramp(0, 1, T.vscale() * 4)]
"""
    _assert_print(main, expected_output)


def test_vectorize_llvm_pure_intrin():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((4,), "float32"), B: TB.Tensor((4,), "float32")):
        A[TB.Ramp(0, 1, 4)] = TB.call_llvm_pure_intrin(
            "llvm.sqrt", B[TB.Ramp(0, 1, 4)], ty="float32x4"
        )

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
    A[0:4] = T.call_llvm_pure_intrin("llvm.sqrt", B[T.Ramp(0, 1, 4)], ty=T.float32x4)
"""
    _assert_print(main, expected_output)


def test_func_with_loop_jumps():
    from tvm_tirx.script import tirx as TB

    @T.function
    def main(A: TB.Tensor((4,), "float32"), B: TB.Tensor((4,), "float32")):
        for i in range(1000):
            if i % 13 == 0:
                A[1] = A[1] + 1
                continue
            if A[0] >= B[0]:
                break

    expected_output = """
from __future__ import annotations

# from tvm_tirx.script import tirx as T

@T.function
def main(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
    for i in range(1000):
        if i % 13 == 0:
            A[1] = A[1] + T.float32(1.0)
            continue
        if A[0] >= B[0]:
            break
"""
    _assert_print(main, expected_output)


def opt_gemm_lower():
    """Representative vector GEMM lowering with base and offset MAC accesses."""

    @tvm_tirx.script.ir_module
    class Module:
        @T.function
        def mmult(
            A_1: T.Tensor([16384], "float32", align=64),
            B_1: T.Tensor([1024, 1024], "float32", align=64),
            C_1: T.Tensor([16384], "float32", align=64),
        ) -> None:
            # function attr dict
            T.func_attr({"tirx.noalias": True})
            # body
            packedB = T.alloc_tensor(ty_args=[T.Tensor((32768,), "float32")])
            for x in T.parallel(0, 32):
                for y in T.serial(0, 1024):
                    packedB[T.ramp(((x * 32768) + (y * 32)), 1, 32)] = B_1[y, T.ramp(x * 32, 1, 32)]
            for x_outer in T.parallel(0, 32):
                C_global = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
                for y_outer in T.serial(0, 32):
                    for x_c_init in T.serial(0, 32):
                        C_global[T.ramp((x_c_init * 32), 1, 32)] = T.broadcast(T.float32(0), 32)
                    for k_outer in T.serial(0, 256):
                        for x_c in T.serial(0, 32):
                            C_global[T.ramp((x_c * 32), 1, 32)] = C_global[
                                T.ramp((x_c * 32), 1, 32)
                            ] + (
                                T.broadcast(
                                    A_1[(((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4))],
                                    32,
                                )
                                * packedB[T.ramp(((y_outer * 32768) + (k_outer * 128)), 1, 32)]
                            )
                            C_global[T.ramp((x_c * 32), 1, 32)] = C_global[
                                T.ramp((x_c * 32), 1, 32)
                            ] + (
                                T.broadcast(
                                    A_1[
                                        ((((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4)) + 3),
                                    ],
                                    32,
                                )
                                * packedB[
                                    T.ramp((((y_outer * 32768) + (k_outer * 128)) + 96), 1, 32)
                                ]
                            )
                    for x_inner in T.serial(0, 32):
                        for y_inner in T.serial(0, 32):
                            C_1[
                                (
                                    (((x_outer * 32768) + (x_inner * 1024)) + (y_outer * 32))
                                    + y_inner
                                )
                            ] = C_global[((x_inner * 32) + y_inner)]

    return Module


def opt_conv_tensorcore_lower():
    """Representative WMMA lowering with scalar/vector copies and zero/nonzero fragments."""

    @T.function
    def func(
        A: T.Tensor((16, 14, 14, 16, 16, 16), "float16"),
        W: T.Tensor((3, 3, 16, 32, 16, 16), "float16"),
        Conv: T.Tensor((16, 14, 14, 32, 16, 16), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
        # body
        A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor([12845056], "float16", data=A.data)])
        W_1 = T.decl_tensor(W.data, ty_args=[T.Tensor([1179648], "float16", data=W.data)])
        Conv_1 = T.decl_tensor(Conv.data, ty_args=[T.Tensor([25690112], "float32", data=Conv.data)])
        bz = T.launch_thread("blockIdx.z", 196)
        Conv_wmma_accumulator = T.alloc_tensor(
            ty_args=[T.Tensor((2048,), "float32", scope="wmma.accumulator")]
        )
        Apad_shared = T.alloc_tensor(ty_args=[T.Tensor((12288,), "float16", scope="shared")])
        W_shared = T.alloc_tensor(ty_args=[T.Tensor((12288,), "float16", scope="shared")])
        Apad_shared_wmma_matrix_a = T.alloc_tensor(
            ty_args=[T.Tensor((512,), "float16", scope="wmma.matrix_a")]
        )
        W_shared_wmma_matrix_b = T.alloc_tensor(
            ty_args=[T.Tensor((1024,), "float16", scope="wmma.matrix_b")]
        )
        bx = T.launch_thread("blockIdx.x", 2)
        by = T.launch_thread("blockIdx.y", 4)
        ty = T.launch_thread("threadIdx.y", 4)
        tz = T.launch_thread("threadIdx.z", 2)
        T.evaluate(T.gpu_fill_fragment(Conv_wmma_accumulator.data, 16, 16, 16, 0, T.float32(0)))
        T.evaluate(T.gpu_fill_fragment(Conv_wmma_accumulator.data, 16, 16, 16, 7, T.float32(0)))
        for ic_outer in T.serial(0, 8):
            for kh in T.serial(0, 3):
                for ax2 in T.serial(0, 3):
                    with T.launch_thread("threadIdx.x", 32) as tx:
                        Apad_shared[((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx)] = (
                            T.if_then_else(
                                (
                                    (
                                        (
                                            (1 <= (T.floordiv(bz, 14) + kh))
                                            and ((T.floordiv(bz, 14) + kh) < 15)
                                        )
                                        and (1 <= (ax2 + T.floormod(bz, 14)))
                                    )
                                    and ((ax2 + T.floormod(bz, 14)) < 15)
                                ),
                                A_1[
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                ((bx * 6422528) + (ty * 1605632))
                                                                + (tz * 802816)
                                                            )
                                                            + (kh * 57344)
                                                        )
                                                        + (bz * 4096)
                                                    )
                                                    + (ax2 * 4096)
                                                )
                                                + (ic_outer * 512)
                                            )
                                            + tx
                                        )
                                        - 61440
                                    ),
                                ],
                                T.float16(0),
                            )
                        )
                    tx = T.launch_thread("threadIdx.x", 32)
                    Apad_shared[(((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 480)] = (
                        T.if_then_else(
                            (
                                (
                                    (
                                        (1 <= (T.floordiv(bz, 14) + kh))
                                        and ((T.floordiv(bz, 14) + kh) < 15)
                                    )
                                    and (1 <= (ax2 + T.floormod(bz, 14)))
                                )
                                and ((ax2 + T.floormod(bz, 14)) < 15)
                            ),
                            A_1[
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            ((bx * 6422528) + (ty * 1605632))
                                                            + (tz * 802816)
                                                        )
                                                        + (kh * 57344)
                                                    )
                                                    + (bz * 4096)
                                                )
                                                + (ax2 * 4096)
                                            )
                                            + (ic_outer * 512)
                                        )
                                        + tx
                                    )
                                    - 60960
                                ),
                            ],
                            T.float16(0),
                        )
                    )
                with T.launch_thread("threadIdx.x", 32) as tx:
                    W_shared[T.ramp((((ty * 512) + (tz * 256)) + (tx * 8)), 1, 8)] = W_1[
                        T.ramp(
                            (
                                (
                                    (
                                        (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                        + (ty * 512)
                                    )
                                    + (tz * 256)
                                )
                                + (tx * 8)
                            ),
                            1,
                            8,
                        )
                    ]
                with T.launch_thread("threadIdx.x", 32) as tx:
                    W_shared[T.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 10240), 1, 8)] = W_1[
                        T.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 270336
                            ),
                            1,
                            8,
                        )
                    ]
                for ic_inner in T.serial(0, 2):
                    for kw in T.serial(0, 3):
                        T.evaluate(
                            T.gpu_load_matrix_sync(
                                Apad_shared_wmma_matrix_a.data,
                                16,
                                16,
                                16,
                                0,
                                T.ptr_byte_offset(
                                    Apad_shared.data,
                                    (((ty * 3072) + (kw * 512)) + (ic_inner * 256)) * 2,
                                    ty=Apad_shared.data.ty,
                                ),
                                16,
                                "row_major",
                            )
                        )
                        T.evaluate(
                            T.gpu_load_matrix_sync(
                                Apad_shared_wmma_matrix_a.data,
                                16,
                                16,
                                16,
                                1,
                                T.ptr_byte_offset(
                                    Apad_shared.data,
                                    ((((ty * 3072) + (kw * 512)) + (ic_inner * 256)) + 1536) * 2,
                                    ty=Apad_shared.data.ty,
                                ),
                                16,
                                "row_major",
                            )
                        )
                        T.evaluate(
                            T.gpu_load_matrix_sync(
                                W_shared_wmma_matrix_b.data,
                                16,
                                16,
                                16,
                                0,
                                T.ptr_byte_offset(
                                    W_shared.data,
                                    (((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) * 2,
                                    ty=W_shared.data.ty,
                                ),
                                16,
                                "row_major",
                            )
                        )
                        T.evaluate(
                            T.gpu_load_matrix_sync(
                                W_shared_wmma_matrix_b.data,
                                16,
                                16,
                                16,
                                3,
                                T.ptr_byte_offset(
                                    W_shared.data,
                                    ((((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) + 768) * 2,
                                    ty=W_shared.data.ty,
                                ),
                                16,
                                "row_major",
                            )
                        )
                        T.evaluate(
                            T.gpu_mma_sync(
                                Conv_wmma_accumulator.data,
                                0,
                                Apad_shared_wmma_matrix_a.data,
                                0,
                                W_shared_wmma_matrix_b.data,
                                0,
                                Conv_wmma_accumulator.data,
                                0,
                            )
                        )
                        T.evaluate(
                            T.gpu_mma_sync(
                                Conv_wmma_accumulator.data,
                                7,
                                Apad_shared_wmma_matrix_a.data,
                                1,
                                W_shared_wmma_matrix_b.data,
                                3,
                                Conv_wmma_accumulator.data,
                                7,
                            )
                        )
        T.evaluate(
            T.gpu_store_matrix_sync(
                Conv_wmma_accumulator.data,
                16,
                16,
                16,
                0,
                T.ptr_byte_offset(
                    Conv_1.data,
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    )
                    * 4,
                    ty=Conv_1.data.ty,
                ),
                16,
                "row_major",
            )
        )
        T.evaluate(
            T.gpu_store_matrix_sync(
                Conv_wmma_accumulator.data,
                16,
                16,
                16,
                7,
                T.ptr_byte_offset(
                    Conv_1.data,
                    (
                        (
                            ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                            + (tz * 1024)
                        )
                        + 1606400
                    )
                    * 4,
                    ty=Conv_1.data.ty,
                ),
                16,
                "row_major",
            )
        )

    return func


def opt_conv_tensorcore_mod_host():
    """Representative packed host ABI checks and device/kernel calls."""

    @T.function
    def opt_conv_tensorcore_mod_host(
        args: T.handle,
        arg_type_ids: T.Tensor((3,), "int32"),
        num_args: T.int32,
        out_ret_value: T.handle,
        out_ret_tcode: T.handle,
        resource_handle: T.handle,
    ) -> T.int32:
        # function attr dict
        T.func_attr(
            {
                "tirx.noalias": True,
                "global_symbol": "default_function",
                "tirx.is_entry_func": True,
                "calling_conv": 1,
            }
        )
        # body
        stack_tcode_data: T.let[T.handle("int32")] = T.stack_alloca("arg_tcode", 10)
        stack_tcode = T.decl_tensor(
            stack_tcode_data, ty_args=[T.Tensor([9], "int32", data=stack_tcode_data)]
        )
        stack_value: T.let[T.handle] = T.stack_alloca("arg_value", 10)
        assert num_args == 3, "default_function: num_args should be 3"
        arg0: T.let[T.handle] = T.abi_field_get(args, 0, 12, ty="handle")
        arg0_code: T.let[T.int32] = arg_type_ids[0]
        arg1: T.let[T.handle] = T.abi_field_get(args, 1, 12, ty="handle")
        arg2: T.let[T.handle] = T.abi_field_get(args, 2, 12, ty="handle")

        A: T.let[T.handle] = T.abi_field_get(arg0, 0, 1, ty="handle")
        A_tensor = T.decl_tensor(A, ty_args=[T.Tensor([1], "float16", data=A)])
        T.assume_aligned(A_tensor, 128)
        arg0_shape_data: T.let[T.handle("int64")] = T.abi_field_get(
            arg0, 0, 2, ty=T.handle("int64").ty
        )
        arg0_shape = T.decl_tensor(
            arg0_shape_data, ty_args=[T.Tensor([6], "int64", data=arg0_shape_data)]
        )
        arg0_strides_data: T.let[T.handle("int64")] = T.abi_field_get(
            arg0, 0, 3, ty=T.handle("int64").ty
        )
        arg0_strides = T.decl_tensor(
            arg0_strides_data, ty_args=[T.Tensor([6], "int64", data=arg0_strides_data)]
        )

        dev_id: T.let[T.int32] = T.abi_field_get(arg0, 0, 9, ty="int32")

        W: T.let[T.handle] = T.abi_field_get(arg1, 0, 1, ty="handle")
        W_tensor = T.decl_tensor(W, ty_args=[T.Tensor([1], "float16", data=W)])
        T.assume_aligned(W_tensor, 128)

        Conv: T.let[T.handle] = T.abi_field_get(arg2, 0, 1, ty="handle")
        Conv_tensor = T.decl_tensor(Conv, ty_args=[T.Tensor([1], "float32", data=Conv)])
        T.assume_aligned(Conv_tensor, 128)

        assert (((arg0_code == 3) or (arg0_code == 13)) or (arg0_code == 7)) or (arg0_code == 4), (
            "default_function: Expect arg[0] to be pointer"
        )
        assert 6 == T.abi_field_get(arg0, 0, 4, ty="int32"), "arg0.ndim is expected to equal 6"
        assert (
            (T.abi_field_get(arg0, 0, 5, ty="uint8") == T.uint8(2))
            and (T.abi_field_get(arg0, 0, 6, ty="uint8") == T.uint8(16))
        ) and (T.abi_field_get(arg0, 0, 7, ty="uint16") == T.uint16(1)), (
            "arg0.dtype is expected to be float16"
        )
        assert 16 == T.cast(arg0_shape[0], ty="int32"), (
            "Argument arg0.shape[0] has an unsatisfied constraint"
        )
        assert 14 == T.cast(arg0_shape[1], ty="int32"), (
            "Argument arg0.shape[1] has an unsatisfied constraint"
        )
        if not (T.isnullptr(arg0_strides.data)):
            assert (
                (
                    (
                        (
                            (1 == T.cast(arg0_strides[5], ty="int32"))
                            and (16 == T.cast(arg0_strides[4], ty="int32"))
                        )
                        and (256 == T.cast(arg0_strides[3], ty="int32"))
                    )
                    and (4096 == T.cast(arg0_strides[2], ty="int32"))
                )
                and (57344 == T.cast(arg0_strides[1], ty="int32"))
            ) and (802816 == T.cast(arg0_strides[0], ty="int32")), (
                "arg0.strides: expected to be compact array"
            )
            T.evaluate(0)
        assert T.uint64(0) == T.abi_field_get(arg0, 0, 8, ty="uint64"), (
            "Argument arg0.byte_offset has an unsatisfied constraint"
        )
        assert 2 == T.abi_field_get(arg0, 0, 10, ty="int32"), (
            "Argument arg0.device_type has an unsatisfied constraint"
        )
        assert (
            (T.abi_field_get(arg2, 0, 5, ty="uint8") == T.uint8(2))
            and (T.abi_field_get(arg2, 0, 6, ty="uint8") == T.uint8(32))
        ) and (T.abi_field_get(arg2, 0, 7, ty="uint16") == T.uint16(1)), (
            "arg2.dtype is expected to be float32"
        )
        assert dev_id == T.abi_field_get(arg2, 0, 9, ty="int32"), (
            "Argument arg2.device_id has an unsatisfied constraint"
        )
        set_device_status = T.call_extern("TVMBackendSetDevice", 2, dev_id, ty="int32")
        if set_device_status != 0:
            T.evaluate(T.throw_last_error())
        with T.compute_scope("default_function_compute_"):
            T.evaluate(T.abi_field_set(stack_value, 0, 12, A))
            stack_tcode[0] = 3
            T.evaluate(T.abi_field_set(stack_value, 1, 12, W))
            stack_tcode[1] = 3
            T.evaluate(T.abi_field_set(stack_value, 2, 12, Conv))
            stack_tcode[2] = 3
            T.evaluate(T.abi_field_set(stack_value, 3, 12, T.cast(196, ty="int64")))
            stack_tcode[3] = 0
            T.evaluate(T.abi_field_set(stack_value, 4, 12, T.cast(2, ty="int64")))
            stack_tcode[4] = 0
            T.evaluate(T.abi_field_set(stack_value, 5, 12, T.cast(4, ty="int64")))
            stack_tcode[5] = 0
            T.evaluate(T.abi_field_set(stack_value, 6, 12, T.cast(4, ty="int64")))
            stack_tcode[6] = 0
            T.evaluate(T.abi_field_set(stack_value, 7, 12, T.cast(2, ty="int64")))
            stack_tcode[7] = 0
            T.evaluate(T.abi_field_set(stack_value, 8, 12, T.cast(32, ty="int64")))
            stack_tcode[8] = 0
            T.evaluate(T.call_packed_lowered("default_function_kernel0", stack_value, 0, 9))

    return opt_conv_tensorcore_mod_host


def select():
    @T.function
    def select(A: T.Tensor((), "float32")) -> None:
        A[()] = T.Select(True, 1, 2)

    return select


def minmax():
    @T.function
    def minmax(A: T.Tensor((), "float32")) -> None:
        A[()] = T.min(1, 2)
        A[()] = T.max(1, 2)

    return minmax


def constant_folding():
    @T.function
    def constant_folding(A: T.Tensor((), "float32")) -> None:
        A[()] = T.min(2.2, 5.2)
        A[()] = T.max(T.float32(2.2), T.float32(T.float32(5.2)))
        A[()] = T.min(2.2, 5.0)

    return constant_folding


def simplify_bracket():
    # uninitialized variables
    a = T.dynamic("a", "int32")
    b = T.dynamic("b", "int32")
    c = T.dynamic("c", "int32")
    d = T.dynamic("d", "int32")

    @T.function(check_well_formed=False)
    def simplify_bracket() -> None:
        T.evaluate(a + b * (c + d))

    return simplify_bracket


def function_with_allocate_annotations():
    @T.function
    def function_with_allocate_annotations(
        placeholder_29: T.Tensor([802816], dtype="uint8", align=64),
        T_cast_7: T.Tensor([200704], dtype="int16", align=64),
    ) -> None:
        # function attr dict
        T.func_attr(
            {"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tirx.noalias": True}
        )

        # body
        tensor_2 = T.alloc_tensor(
            ty_args=[T.Tensor((200704,), "uint8")], annotations={"attr1_key": "attr1_value"}
        )
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_2)] = T.max(
                        tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_2)],
                        T.if_then_else(
                            (
                                (((ax0_ax1_fused_4 * 2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112)
                                and (((ax2_4 * 2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)
                            ),
                            placeholder_29[
                                (
                                    (
                                        (
                                            (
                                                (ax0_ax1_fused_4 * 14336)
                                                + (T.floordiv(rv0_rv1_fused_1, 3) * 7168)
                                            )
                                            + (ax2_4 * 128)
                                        )
                                        + (T.floormod(rv0_rv1_fused_1, 3) * 64)
                                    )
                                    + ax3_2
                                )
                            ],
                            T.uint8(0),
                        ),
                    )
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5 * 3584) + (ax2_5 * 64)) + ax3_3)] = T.cast(
                    tensor_2[(((ax0_ax1_fused_5 * 3584) + (ax2_5 * 64)) + ax3_3)], ty="int16"
                )

    return function_with_allocate_annotations


def comm_reducer_single_reduce_group():
    @T.function
    def comm_reducer_single_reduce_group(
        A: T.Tensor([16384], dtype="float32"), b: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i in T.serial(0, 128):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            reduce_temp0 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            T.evaluate(
                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                    (T.float32(0),),
                    (A[i * 128 + threadIdx_x],),
                    True,
                    (reduce_temp0[0],),
                    (threadIdx_x,),
                )
            )

    return comm_reducer_single_reduce_group


def comm_reducer_multiple_reduce_groups():
    @T.function
    def comm_reducer_multiple_reduce_groups(
        A: T.Tensor([16384], dtype="float32"), b: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i in T.serial(0, 128):
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            reduce_temp0 = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
            reduce_temp1 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            T.evaluate(
                T.gpu_thread_allreduce(
                    T.Lambda(
                        [T.int32, T.float32, T.int32, T.float32],
                        lambda x0, x1, y0, y1: (
                            T.Select(x1 >= y1, x0, y0),
                            T.Select(x1 >= y1, x1, y1),
                        ),
                    ),
                    (T.int32(-1), T.min_value(ty="float32")),
                    (i * 128 + threadIdx_x, A[i * 128 + threadIdx_x]),
                    True,
                    (reduce_temp0[0], reduce_temp1[0]),
                    (threadIdx_x,),
                )
            )

    return comm_reducer_multiple_reduce_groups


def func_div_mod():
    # not well-formed: free variables
    a = T.dynamic("a", "int32")
    b = T.dynamic("b", "int32")

    @T.function(check_well_formed=False)
    def func_div_mod():
        T.evaluate(a // b)
        T.evaluate(a % b)
        T.evaluate(T.truncmod(a, b))

    return func_div_mod


def test_div_mod():
    func = func_div_mod()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func, True)

    assert isinstance(func.body[0].value, tvm_tirx.tirx.FloorDiv)
    assert isinstance(func.body[1].value, tvm_tirx.tirx.FloorMod)
    assert isinstance(func.body[2].value, tvm_tirx.tirx.Mod)


def func_with_target_spec_by_config():
    @T.function
    def func_with_target_spec_by_config() -> None:
        T.func_attr(
            {
                "kTarget": T.target(
                    {
                        "max_num_threads": 1024,
                        "arch": "sm_70",
                        "thread_warp_size": 32,
                        "kind": "cuda",
                        "tag": "",
                        "keys": ["cuda", "gpu"],
                        "host": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}),
                    }
                )
            }
        )
        T.evaluate(0)

    return func_with_target_spec_by_config


def func_with_target_spec_by_str():
    @T.function
    def func_with_target_spec_by_str() -> None:
        T.func_attr({"kTarget": T.target("nvidia/nvidia-a100")})
        T.evaluate(0)

    return func_with_target_spec_by_str


def func_with_target_and_host_spec_by_str():
    @T.function
    def func():
        T.func_attr({"target": T.target("nvidia/nvidia-a100", host="llvm")})
        T.evaluate(0)

    return func


def func_T_ptr_let_statement():
    @T.function
    def func_T_ptr_let_statement(
        args: T.handle, arg_type_ids_handle: T.handle("int32"), num_args: T.int32
    ) -> None:
        # The T.Ptr declaration in the parameter list should parse
        # correctly, and should be usable as the data pointer in a tensor.
        arg_type_ids = T.decl_tensor(
            arg_type_ids_handle, ty_args=[T.Tensor([2], "int32", data=arg_type_ids_handle)]
        )

        arg0: T.let[T.handle] = T.abi_field_get(args, 0, 12, ty="handle")
        arg1: T.let[T.handle] = T.abi_field_get(args, 1, 12, ty="handle")

        # The ABI field is an opaque pointer.  Retag it explicitly before
        # binding it to the tensor's exact element pointer type.
        A_data: T.let[T.handle("float32")] = T.reinterpret(
            T.abi_field_get(arg0, 0, 1, ty="handle"), ty=T.handle("float32").ty
        )

        # The tensor declaration has a data pointer defined earlier in
        # this function.  It should only be defined after the data pointer
        # has been defined, and should not be hoisted into the header of
        # the function as other tensor_decl statements can be.
        A = T.decl_tensor(A_data, ty_args=[T.Tensor([1024], "float32", data=A_data)])
        B_data: T.let[T.handle("float32")] = T.reinterpret(
            T.abi_field_get(arg1, 0, 1, ty="handle"), ty=T.handle("float32").ty
        )
        B = T.decl_tensor(B_data, ty_args=[T.Tensor([1024], "float32", data=B_data)])

        B[0] = A[0]

    return func_T_ptr_let_statement


def func_T_ptr_allocate():
    @T.function
    def func_T_ptr_allocate() -> None:
        A = T.alloc_tensor(ty_args=[T.Tensor((1024,), "float32")])
        A[0] = 0.0

    return func_T_ptr_allocate


def string_annotation_escaping():
    @T.function
    def string_annotation_of_special_chars():
        T.func_attr(
            {
                "key1": '"\'hello\t\r"',
                "key2": """
            %1 = add i32 %0, %0
            %2 = add i32 %0, %1
            %3 = add i32 %1, %2
            """,
            }
        )
        T.evaluate(0)

    return string_annotation_of_special_chars


def pointer_type():
    @T.function
    def func_with_ptr_type_annotations(x: T.handle("int32"), y: T.handle("int32", "shared")):
        xx = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32")])
        yy = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="shared")])
        a: T.let[T.handle("int32")] = T.address_of(xx[0])
        b: T.let[T.handle("int32", "shared")] = T.address_of(yy[0])
        T.evaluate(T.call_extern("copy", a, b, ty=""))

    return func_with_ptr_type_annotations


def tensor_ramp_access_as_slice_index():
    @T.function
    def tensor_ramp_access(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
    ) -> None:
        for i in range(128):
            A[i : i + 1 : 1] = i
        for i in range(4):
            B[i * 32 : i * 32 + 32] = A[T.Ramp(i * 32, 1, 32)] + T.broadcast(1.0, 32)
        for i in range(4):
            C[i : i + 128 : 4] = B[T.Ramp(i, 4, 32)] + T.broadcast(1.0, 32)

    return tensor_ramp_access


def ramp_int64():
    @T.function
    def func() -> None:
        T.evaluate(T.Ramp(T.int64(0), 1, 3))

    return func


def scalable_vectors():
    @T.function
    def func(A: T.Tensor((200,), "float32")):
        A[T.Ramp(11, 2, 4 * tirx.vscale())] = T.Broadcast(125, 4 * tirx.vscale())

    return func


def predicated_tensor_load_store():
    @T.function
    def func(A: T.Tensor((4,), "float32"), B: T.Tensor((8,), "float32")):
        for i_0 in range(4):
            load_a = T.meta_var(
                T.call_intrin(
                    "tirx.masked_load",
                    A,
                    T.Ramp(i_0, 1, 4),
                    T.Broadcast(T.bool(True), 4),
                    ty="float32x4",
                )
            )
            T.evaluate(
                T.call_intrin(
                    "tirx.masked_store",
                    B,
                    load_a,
                    T.Ramp(0, 2, 4),
                    T.Broadcast(T.bool(True), 4),
                    ty="void",
                )
            )

    return func


def let_expression():
    x = T.dynamic("x", "int32")

    @T.function
    def func():
        T.evaluate(T.Let(x + 1, where={x: 1}))

    return func


def test_void_ptr_vs_handle():
    """An untyped handle is the canonical void-pointer type."""

    # Generates PtrType(PrimType::Void())
    @T.function
    def void_ptr(out_ret_value: T.handle("void")):
        T.evaluate(out_ret_value)

    # Generates PtrType::VoidPtrType()
    @T.function
    def handle(out_ret_value: T.handle):
        T.evaluate(out_ret_value)

    tvm_tirx.ir.assert_structural_equal(void_ptr.params[0].ty, handle.params[0].ty)
    script = void_ptr.script()
    assert "out_ret_value: T.Ptr(T.void)" in script
    assert 'T.handle("void")' not in script
    tvm_tirx.ir.assert_structural_equal(
        void_ptr,
        tvm_tirx.script.from_source(
            script,
            extra_vars={
                "I": tvm_tirx.script.ir,
                "T": tvm_tirx.script.tirx,
            },
        ),
    )

    @T.function
    def scoped_void_ptr(out_ret_value: T.handle("void", "shared")):
        T.evaluate(out_ret_value)

    scoped_script = scoped_void_ptr.script()
    assert 'out_ret_value: T.Ptr(T.void, "shared")' in scoped_script
    assert 'T.handle("void"' not in scoped_script
    tvm_tirx.ir.assert_structural_equal(
        scoped_void_ptr,
        tvm_tirx.script.from_source(
            scoped_script,
            extra_vars={
                "I": tvm_tirx.script.ir,
                "T": tvm_tirx.script.tirx,
            },
        ),
    )


def void_ptr():
    @T.function
    def func(out_ret_value: T.handle("void")):
        T.evaluate(out_ret_value)

    return func


def decl_tensor():
    @T.function
    def func(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
        A_flattened = T.decl_tensor(A.data, ty_args=[T.Tensor((256,), "float32", data=A.data)])
        B_flattened = T.decl_tensor(B.data, ty_args=[T.Tensor((256,), "float32", data=B.data)])
        C_alias = T.decl_tensor(
            A_flattened.data, ty_args=[T.Tensor((256,), "float32", data=A_flattened.data)]
        )
        for i in range(256):
            B_flattened[i] = A_flattened[i] + C_alias[i] + T.float32(1.0)

    return func


def allocate_and_decl_tensor():
    @T.function
    def func(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")) -> None:
        D = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(4):
            C = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32")])
            for j in range(4):
                C[j] = A[i * 4 + j] + T.float32(1.0)
            for j in range(4):
                D[j] = C[j]
            for j in range(4):
                B[i * 4 + j] = D[j]

    return func


def alloc_tensor_example():
    @T.function
    def func(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32")])
        for i in range(128):
            B[i] = A[i] * T.float32(2)
        for i in range(128):
            C[i] = B[i] + T.float32(1)

    return func


def minimal_i32_literal():
    @T.function
    def func() -> None:
        T.evaluate(T.int32(-2147483648))
        T.evaluate(-T.int64(2147483648))

    return func


def bool_primitive():
    @T.function
    def func() -> None:
        T.evaluate(T.bool(True))

    return func


def bool_cast():
    # uninitialized var
    a = T.dynamic("a", "bool")

    @T.function(check_well_formed=False)
    def func() -> None:
        T.evaluate(T.bool(T.int32(0)))
        T.evaluate(a == T.bool(False))

    return func


def nested_boolean_expressions():
    expressions = {
        "and_lhs_and": lambda i, j, k: tirx.all(tirx.all(i, j), k),
        "and_rhs_and": lambda i, j, k: tirx.all(i, tirx.all(j, k)),
        "and_lhs_or": lambda i, j, k: tirx.all(tirx.any(i, j), k),
        "and_rhs_or": lambda i, j, k: tirx.all(i, tirx.any(j, k)),
        "or_lhs_and": lambda i, j, k: tirx.any(tirx.all(i, j), k),
        "or_rhs_and": lambda i, j, k: tirx.any(i, tirx.all(j, k)),
        "or_lhs_or": lambda i, j, k: tirx.any(tirx.any(i, j), k),
        "or_rhs_or": lambda i, j, k: tirx.any(i, tirx.any(j, k)),
        "and_of_ors": lambda i, j, k: tirx.all(
            tirx.any(i, j), tirx.any(j, k), tirx.any(i, k), i, j, k
        ),
        "or_of_ands": lambda i, j, k: tirx.any(
            tirx.all(i, j), tirx.all(j, k), tirx.all(i, k), i, j, k
        ),
    }

    def make_ir_generator(name, expression):
        def inner():
            @T.function
            def func(A: T.Tensor(1, "bool"), i: T.bool, j: T.bool, k: T.bool):
                A[0] = expression(i, j, k)

            return func

        inner.__name__ = f"nested_boolean_expr_{name}"
        return inner

    for name, expression in expressions.items():
        generator = make_ir_generator(name, expression)

        yield generator


def intrinsic_pow():
    @T.function
    def func():
        T.pow(T.float32(1), T.float32(1))

    return func


def tvm_shfl_builtins():
    @T.function
    def func(
        A: T.handle("float32"),
        B: T.handle("float32"),
        C: T.handle("float32"),
    ):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        A_warp = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        B_warp = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        red_buf0 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        A_warp_1 = T.decl_tensor(A_warp.data, ty_args=[T.Tensor((32,), "float32", scope="local")])
        A_1 = T.decl_tensor(A, ty_args=[T.Tensor((32,), "float32", data=A)])  # A is a handle param
        A_warp_1[0] = A_1[threadIdx_x]
        B_warp_1 = T.decl_tensor(B_warp.data, ty_args=[T.Tensor((32,), "float32", scope="local")])
        T.gpu_storage_sync("warp")
        B_warp_1[0] = T.gpu_warp_shuffle(
            T.gpu_warp_activemask(), A_warp_1[0], threadIdx_x % 4 * 8 + threadIdx_x // 4, 32, 32
        ) + T.float32(1)
        red_buf0_1 = T.decl_tensor(
            red_buf0.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
        )
        mask = T.alloc_tensor(ty_args=[T.Tensor((1,), "uint32", scope="local")])
        t0 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        red_buf0_1[0] = A_warp_1[0]
        mask_1 = T.decl_tensor(mask.data, ty_args=[T.Tensor((1,), "uint32", scope="local")])
        mask_1[0] = T.gpu_warp_activemask()
        t0_1 = T.decl_tensor(t0.data, ty_args=[T.Tensor((1,), "float32", scope="local")])
        t0_1[0] = T.gpu_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
        red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
        t0_1[0] = T.gpu_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
        red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
        t0_1[0] = T.gpu_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
        red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
        t0_1[0] = T.gpu_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
        red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
        t0_1[0] = T.gpu_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
        red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
        red_buf0_1[0] = T.gpu_warp_shuffle(mask_1[0], red_buf0_1[0], 0, 32, 32)
        # NOTE(Zihao): test gpu_warp_shuffle_up
        red_buf0_1[0] = T.gpu_warp_shuffle_up(mask_1[0], red_buf0_1[0], 0, 32, 32)
        if threadIdx_x == 0:
            C_1 = T.decl_tensor(C, ty_args=[T.Tensor((1,), "float32", data=C)])
            C_1[0] = red_buf0_1[0]
        B_1 = T.decl_tensor(B, ty_args=[T.Tensor((32,), "float32", data=B)])
        B_1[threadIdx_x] = B_warp_1[0]

    return func


def make_packed_api_result():
    @T.function
    def func(A: T.Tensor(64, "float32")):
        T.func_attr({"global_symbol": "main", "target": T.target("cuda")})
        bx = T.launch_thread("blockIdx.x", 64)
        T.evaluate(A[bx])

    mod = tvm_tirx.IRModule.from_expr(func)
    return tvm_tirx.tirx.transform.MakePackedAPI()(mod)


def tvm_struct_set_generated_in_cpp():
    """Ensure same dtype for abi_field_set in Python/C++

    The TVMStructSet method in C++, used internally by
    LowerTVMBuiltin, and the Python method `T.abi_field_set`, used
    when parsing TVMScript should use the same dtype "int32".
    """

    @I.ir_module
    class Module:
        @T.function
        def tir_packed_call(A: T.Tensor(16, "float32")):
            with T.device_context(0, 0):
                T.evaluate(
                    T.call_cpacked(
                        "tvm_test_cpacked",
                        T.stack_make_dltensor(
                            A.data,
                            T.stack_make_shape(16),
                            T.reinterpret(T.uint64(0), ty="handle"),
                            T.uint32(1),
                            T.Cast("float32", 0),
                            0,
                        ),
                    )
                )

    return tvm_tirx.tirx.transform.LowerTVMBuiltin()(Module)


def undefined_data_ptr_in_decl_tensor():
    """The T.decl_tensor syntax should not introduce an Allocate

    While T.decl_tensor can be used to represent an
    Allocate/DeclTensor pair, performing a round-trip through
    TVMScript should not introduce an Allocate node.
    """

    # uninitialized var
    @T.function(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_tensor(data_ptr, ty_args=[T.Tensor([1], "float32", data=data_ptr)])
        T.evaluate(buf[0])

    return func


def op_of_literal():
    op_list = [
        (T.exp, 0),
        (T.exp2, 0),
        (T.exp10, 0),
        (T.erf, 0.0),
        (T.tanh, 0.0),
        (T.sigmoid, 0.0),
        (T.log, 0.0),
        (T.log2, 0.0),
        (T.log1p, 0.0),
        (T.tan, 0.0),
        (T.cos, 0.0),
        (T.acos, 0.0),
        (T.acosh, 0.0),
        (T.sin, 0.0),
        (T.sinh, 0.0),
        (T.asin, 0.0),
        (T.asinh, 0.0),
        (T.atan, 0.0),
        (T.atanh, 0.0),
        (T.atan2, (1.0, 0.0)),
        (T.sqrt, 0.0),
        (T.rsqrt, 1.0),
        (T.nextafter, (0.0, 1.0)),
        (T.hypot, (1.0, 1.0)),
        (T.copysign, (1.0, 1.0)),
        (T.popcount, 0),
        (T.fmod, (1.0, 1.0)),
    ]

    def make_ir_generator(op, arg):
        def inner():
            call_expr = op(*arg) if isinstance(arg, tuple) else op(arg)

            @T.function
            def func():
                T.evaluate(call_expr)

            return func

        inner.__name__ = f"{op.__name__}_of_literal"
        return inner

    for op, arg in op_list:
        yield make_ir_generator(op, arg)


def test_address_of_tensor():
    @T.function
    def func(A: T.Tensor((128, 128), "float32")):
        T.evaluate(T.address_of(A))

    assert "T.address_of(A[0, 0])" in func.script()


@T.function
def _func():
    T.evaluate(-1)
    T.evaluate(1)
    T.evaluate(2)
    T.evaluate(3)
    T.evaluate(4)
    T.evaluate(5)
    T.evaluate(6)
    T.evaluate(7)


def test_annotation_multi_access_paths():
    result = _func.with_attr("global_symbol", "main").script(
        path_to_annotate={
            AccessPath.root().attr("body").attr("seq").array_item(1): "annotation 1",
            AccessPath.root().attr("body").attr("seq").array_item(3): "annotation 3",
            AccessPath.root().attr("body").attr("seq").array_item(5): "annotation 5",
            AccessPath.root().attr("body").attr("seq").array_item(7): "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm_tirx.script import tirx as T

@T.function
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_annotate_from_multi_obj():
    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.seq[1]: "annotation 1",
            _func.body.seq[3]: "annotation 3",
            _func.body.seq[5]: "annotation 5",
            _func.body.seq[7]: "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm_tirx.script import tirx as T

@T.function
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_disable_concise_scoping_when_scope_annotated():
    @T.function
    def _func():
        x: T.int32 = 1
        y: T.int32 = x + 1
        T.evaluate(y - 1)

    # Explicit scalar declarations lower to allocation and store statements.
    # Keep the annotation attached to the allocation for y.
    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.seq[2]: "annotation 1",
        }
    )
    assert (
        result
        == """# from tvm_tirx.script import tirx as T

@T.function
def main():
    x = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
    x[0] = 1
    y = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])  # annotation 1
    y[0] = x[0] + 1
    T.evaluate(y[0] - 1)"""
    )


def _assert_module_representations(obj, expected):
    assert str(obj).strip() == expected.strip()
    assert repr(obj).strip() == expected.strip()
    if isinstance(obj, IRModule):
        assert obj.script().strip() == expected.strip()


def test_ir_module():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with IB.ir_module():
            with TB.function():
                TB.func_name_("foo")
    mod = ib.get()
    _assert_module_representations(
        mod,
        """
# from tvm_tirx.script import tirx as T

@T.ir_module
class Module:
    @T.function
    def foo():
        pass
""",
    )


@pytest.mark.parametrize(
    "ir_generator",
    [
        opt_gemm_lower,
        opt_conv_tensorcore_lower,
        opt_conv_tensorcore_mod_host,
        comm_reducer_single_reduce_group,
        comm_reducer_multiple_reduce_groups,
        tvm_shfl_builtins,
        make_packed_api_result,
        tvm_struct_set_generated_in_cpp,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_lowered(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        select,
        minmax,
        constant_folding,
        simplify_bracket,
        ramp_int64,
        let_expression,
        minimal_i32_literal,
        bool_primitive,
        bool_cast,
        *nested_boolean_expressions(),
        intrinsic_pow,
        *op_of_literal(),
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_expressions(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        function_with_allocate_annotations,
        func_T_ptr_let_statement,
        func_T_ptr_allocate,
        pointer_type,
        tensor_ramp_access_as_slice_index,
        scalable_vectors,
        predicated_tensor_load_store,
        void_ptr,
        decl_tensor,
        allocate_and_decl_tensor,
        alloc_tensor_example,
        undefined_data_ptr_in_decl_tensor,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_tensors(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        func_with_target_spec_by_config,
        func_with_target_spec_by_str,
        func_with_target_and_host_spec_by_str,
        string_annotation_escaping,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_metadata(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


def test_shared_tensor_metadata():
    tensor = tvm_tirx.runtime.tensor([1, 2, 3])
    module = IRModule(attrs={"first": tensor, "remaining": [tensor, tensor]})
    source = module.script(show_meta=True)
    assert source.count('metadata["ffi.Tensor"][0]') == 3
    assert 'metadata["ffi.Tensor"][1]' not in source
