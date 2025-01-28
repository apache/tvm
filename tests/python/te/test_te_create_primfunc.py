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
# pylint: disable=missing-function-docstring,missing-module-docstring
import numpy as np
import tvm
import tvm.testing
from tvm import te, tir, topi, relay
from tvm.script import tir as T
import pytest


def test_unique_name_complete_block():
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16, 16), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((16, 16), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_prim_func([A, C])
    s = tir.Schedule(func, debug_mask="all")
    assert isinstance(s.get_sref(s.get_block("main")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("main_1")), tir.schedule.StmtSRef)


def test_unique_name_reduction_block():
    k1 = te.reduce_axis((0, 16), "k1")
    k2 = te.reduce_axis((0, 16), "k2")
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16,), lambda i: te.sum(A[i, k1], axis=k1), name="sum")
    C = te.compute((), lambda: te.sum(B[k2], axis=k2), name="sum")
    func = te.create_prim_func([A, C])
    s = tir.Schedule(func, debug_mask="all")
    assert isinstance(s.get_sref(s.get_block("sum")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("sum_1")), tir.schedule.StmtSRef)


def _check_workload(te_workload, tir_workload, index_dtype_override=None, do_simplify=False):
    func = te.create_prim_func(te_workload(), index_dtype_override)
    if do_simplify:
        simplify = tir.transform.Simplify()
        func = simplify(tvm.IRModule.from_expr(func))["main"]
        tir_workload = simplify(tvm.IRModule.from_expr(tir_workload))["main"]
    tvm.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tir.Schedule(func, debug_mask="all")
    assert s


def te_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [A, B, C]


@T.prim_func
def tir_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i0, j0, k0 in T.grid(128, 128, 128):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


@T.prim_func
def tir_matmul_int64(
    A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    C: T.Buffer((T.int64(128), T.int64(128)), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i0, j0, k0 in T.grid(T.int64(128), T.int64(128), T.int64(128)):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_matmul():
    _check_workload(te_matmul, tir_matmul)


def test_matmul_int64():
    _check_workload(te_matmul, tir_matmul_int64, index_dtype_override="int64")


def te_element_wise():
    A = te.placeholder((128, 128), name="A")
    B = te.compute((128, 128), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((128, 128), lambda x, y: B[x, y] + 1, name="C")
    return [A, C]


@T.prim_func
def tir_element_wise(a: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    B = T.alloc_buffer((128, 128))

    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            B[i, j] = A[i, j] * 2.0
    for i0, j0 in T.grid(128, 128):
        with T.block():
            i, j = T.axis.remap("SS", [i0, j0])
            C[i, j] = B[i, j] + 1.0


def test_element_wise():
    _check_workload(te_element_wise, tir_element_wise)


def te_conv2d():
    batch = 16
    in_channel = 16
    out_channel = 32
    size = 14
    kernel = 3

    A = te.placeholder((batch, in_channel, size, size), name="A")
    W = te.placeholder((in_channel, kernel, kernel, out_channel), name="W")
    Apad = te.compute(
        (batch, in_channel, size + 2, size + 2),
        lambda nn, cc, yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= 1, yy - 1 < size, xx >= 1, xx - 1 < size),
            A[nn, cc, yy - 1, xx - 1],
            0.0,
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    B = te.compute(
        (batch, out_channel, size, size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff], axis=[rc, ry, rx]
        ),
        name="B",
    )
    return [A, W, B]


@T.prim_func
def tir_conv2d(a: T.handle, w: T.handle, b: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, [16, 16, 14, 14])
    W = T.match_buffer(w, [16, 3, 3, 32])
    B = T.match_buffer(b, [16, 32, 14, 14])
    Apad = T.alloc_buffer([16, 16, 16, 16])

    for n, c, y, x in T.grid(16, 16, 16, 16):
        with T.block("Apad"):
            nn, cc, yy, xx = T.axis.remap("SSSS", [n, c, y, x])
            Apad[nn, cc, yy, xx] = T.if_then_else(
                1 <= yy and yy < 15 and 1 <= xx and xx < 15,
                A[nn, cc, yy - 1, xx - 1],
                0.0,
                dtype="float32",
            )
    for n, f, y, x, kc, ky, kx in T.grid(16, 32, 14, 14, 16, 3, 3):
        with T.block("B"):
            nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [n, f, y, x, kc, ky, kx])
            with T.init():
                B[nn, ff, yy, xx] = 0.0
            B[nn, ff, yy, xx] += Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff]


def test_conv2d():
    _check_workload(te_conv2d, tir_conv2d)


def te_multi_output():
    n = te.var("n")
    m = te.var("m")
    A0 = te.placeholder((m, n), name="A0")
    A1 = te.placeholder((m, n), name="A1")
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name="B")
    return [A0, A1, B0, B1]


@T.prim_func
def tir_multi_output(a0: T.handle, a1: T.handle, b0: T.handle, b1: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    m = T.int32()
    n = T.int32()
    A0 = T.match_buffer(a0, (m, n))
    A1 = T.match_buffer(a1, (m, n))
    B0 = T.match_buffer(b0, (m, n))
    B1 = T.match_buffer(b1, (m, n))

    for i0, i1 in T.grid(m, n):
        with T.block("B.v0"):
            i, j = T.axis.remap("SS", [i0, i1])
            B0[i, j] = A0[i, j] + 2.0
        with T.block("B.v1"):
            i, j = T.axis.remap("SS", [i0, i1])
            B1[i, j] = A1[i, j] * 3.0


def test_multi_output():
    _check_workload(te_multi_output, tir_multi_output)


def te_extern():
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.extern(
        (128, 128),
        [A, B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], 0, 0
        ),
        name="C",
    )
    return [A, B, C]


@T.prim_func
def tir_extern(a: T.handle, b: T.handle, c: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    off1 = te.var("elem_offset")
    off2 = te.var("elem_offset_1")
    off3 = te.var("elem_offset_2")
    A = T.match_buffer(a, (128, 128), elem_offset=off1)
    B = T.match_buffer(b, (128, 128), elem_offset=off2)
    C = T.match_buffer(c, (128, 128), elem_offset=off3)
    # body
    with T.block("C"):
        T.reads()
        T.writes()
        T.evaluate(
            T.tvm_call_packed(
                "tvm.contrib.cblas.matmul",
                T.tvm_stack_make_array(
                    A.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    off1,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    B.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    off2,
                    dtype="handle",
                ),
                T.tvm_stack_make_array(
                    C.data,
                    T.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    off3,
                    dtype="handle",
                ),
                0,
                0,
                dtype="int32",
            )
        )


def test_extern():
    _check_workload(te_extern, tir_extern)


def te_reordered_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [C, A, B]


@T.prim_func
def tir_reordered_matmul(c: T.handle, a: T.handle, b: T.handle) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i0, j0, k0 in T.grid(128, 128, 128):
        with T.block():
            i, j, k = T.axis.remap("SSR", [i0, j0, k0])
            with T.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_arg_order():
    _check_workload(te_reordered_matmul, tir_reordered_matmul)


def te_scan():
    m = te.var("m")
    n = te.var("n")
    X = te.placeholder((m, n), name="X")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: X[0, i])
    s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
    s_scan = tvm.te.scan(s_init, s_update, s_state, inputs=[X])
    return [X, s_scan]


def test_error_reporting():
    try:
        te.create_prim_func(te_scan())
        assert False
    except TypeError as e:
        error_message = str(e)
        assert error_message.find("Unsupported Operation: ScanOp.") != -1
        return
    assert False


def test_constant():
    M = 11
    A = te.placeholder((M,), name="A")
    B = te.compute(tuple(), lambda: 2, name="B")
    # Manually craft ProducerLoad because `B[]` is not allowed.
    C = te.compute(
        (M,), lambda x: A[x] + tvm.tir.expr.ProducerLoad(B, []), name="C", tag="broadcast"
    )

    func = te.create_prim_func([C, A])
    func = tvm.build(func)
    a_np = np.random.uniform(size=(M,)).astype(A.dtype)
    c = tvm.nd.array(np.zeros(M, dtype=C.dtype))
    x = func(c, tvm.nd.array(a_np))
    tvm.testing.assert_allclose(a_np + 2, c.numpy())


def test_data_dependent_access():
    A = te.placeholder((10,), name="A")
    B = te.placeholder((10,), name="B", dtype="int32")
    C = te.compute((10,), lambda i: A[B[i]])

    func = te.create_prim_func([C, A, B])
    func = tvm.build(func)

    a_np = np.random.uniform(size=(10,)).astype(A.dtype)
    b_np = np.arange(10, dtype=B.dtype)
    c = tvm.nd.array(np.zeros(10, dtype=C.dtype))
    func(c, tvm.nd.array(a_np), tvm.nd.array(b_np))
    tvm.testing.assert_allclose(a_np[b_np], c.numpy())


def test_select_simplify():
    placeholder = te.placeholder([1, 128, 10, 10, 4], dtype="float32")
    tensor = topi.nn.adaptive_pool(placeholder, [1, 1], "avg", "NCHW4c")
    result = te.create_prim_func([placeholder, tensor])
    script_func = result.script()
    # There should be no Select
    assert script_func.find("Select") == -1
    # There should be no undefined vars
    assert script_func.find("Var") == -1


def test_tensor_attr():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute(
        (128, 128),
        lambda x, y: te.sum(A[x, k] * B[y, k], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    func = te.create_prim_func([A, B, C])
    rt_func = tvm.script.from_source(func.script())
    tvm.ir.assert_structural_equal(func, rt_func)


@T.prim_func
def expected_layout_attr(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    D: T.Buffer((128, 128), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
    C = T.alloc_buffer([128, 128], dtype="float32")
    for i0, i1, i2 in T.grid(128, 128, 128):
        with T.block("C"):
            x, y, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C[x, y] = T.float32(0)
            C[x, y] = C[x, y] + A[x, k] * B[y, k]
    for i0, i1 in T.grid(128, 128):
        with T.block("D"):
            T.block_attr({"layout_free_placeholders": [C]})
            x, y = T.axis.remap("SS", [i0, i1])
            D[x, y] = C[x, y] + T.float32(1)


@T.prim_func
def expected_layout_attr_int64(
    A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    D: T.Buffer((T.int64(128), T.int64(128)), "float32"),
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
    C = T.alloc_buffer([T.int64(128), T.int64(128)], dtype="float32")
    for x, y, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
        with T.block("C"):
            v_x, v_y, v_k = T.axis.remap("SSR", [x, y, k])
            T.reads(A[v_x, v_k], B[v_y, v_k])
            T.writes(C[v_x, v_y])
            with T.init():
                C[v_x, v_y] = T.float32(0)
            C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_y, v_k]
    for x, y in T.grid(T.int64(128), T.int64(128)):
        with T.block("D"):
            T.block_attr({"layout_free_placeholders": [C]})
            v_x, v_y = T.axis.remap("SS", [x, y])
            T.reads(C[v_x, v_y])
            T.writes(D[v_x, v_y])
            D[v_x, v_y] = C[v_x, v_y] + T.float32(1)


@pytest.mark.parametrize(
    "index_dtype_override, expected",
    [(None, expected_layout_attr), ("int64", expected_layout_attr_int64)],
)
def test_tensor_layout_attr(index_dtype_override, expected):
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute(
        (128, 128),
        lambda x, y: te.sum(A[x, k] * B[y, k], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    D = te.compute(
        (128, 128),
        lambda x, y: C[x, y] + 1,
        name="D",
        attrs={"layout_free_placeholders": [C]},
    )
    func = te.create_prim_func([A, B, D], index_dtype_override=index_dtype_override)
    tvm.ir.assert_structural_equal(func, expected)


def te_argmax_idx_val():
    def f_combine(x, y):
        lhs = tvm.tir.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm.tir.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def f_identity(dtype0: tvm.DataType, dtype1: tvm.DataType):
        return tvm.tir.const(-1, dtype0), tvm.te.min_value(dtype1)

    argmax = te.comm_reducer(f_combine, f_identity, name="argmax")

    m = te.var("m")
    n = te.var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="float32")
    k = te.reduce_axis((0, n), "k")
    max_idx, max_val = te.compute(
        (m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="argmax"
    )
    return [idx, val, max_idx, max_val]


@T.prim_func
def tir_argmax_idx_val(
    var_idx: T.handle, var_val: T.handle, var_argmax_v0: T.handle, var_argmax_v1: T.handle
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    m = T.int32()
    n = T.int32()
    idx = T.match_buffer(var_idx, [m, n], dtype="int32")
    val = T.match_buffer(var_val, [m, n], dtype="float32")
    argmax_v0 = T.match_buffer(var_argmax_v0, [m], dtype="int32")
    argmax_v1 = T.match_buffer(var_argmax_v1, [m], dtype="float32")
    for i0, i1 in T.grid(m, n):
        with T.block("argmax"):
            i, k = T.axis.remap("SR", [i0, i1])
            T.reads(val[i, k], idx[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = T.int32(-1)
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def te_argmax_val_idx():
    def f_combine(x, y):
        lhs = tvm.tir.Select((x[0] >= y[0]), x[0], y[0])
        rhs = tvm.tir.Select((x[0] >= y[0]), x[1], y[1])
        return lhs, rhs

    def f_identity(dtype0: tvm.DataType, dtype1: tvm.DataType):
        return tvm.te.min_value(dtype0), tvm.tir.const(-1, dtype1)

    argmax = te.comm_reducer(f_combine, f_identity, name="argmax")

    m = te.var("m")
    n = te.var("n")
    val = te.placeholder((m, n), name="val", dtype="float32")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    k = te.reduce_axis((0, n), "k")
    max_val, max_idx = te.compute(
        (m,), lambda i: argmax((val[i, k], idx[i, k]), axis=k), name="argmax"
    )
    return [val, idx, max_val, max_idx]


@T.prim_func
def tir_argmax_val_idx(
    var_val: T.handle, var_idx: T.handle, var_argmax_v0: T.handle, var_argmax_v1: T.handle
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    m = T.int32()
    n = T.int32()
    val = T.match_buffer(var_val, [m, n], dtype="float32")
    idx = T.match_buffer(var_idx, [m, n], dtype="int32")
    argmax_v0 = T.match_buffer(var_argmax_v0, [m], dtype="float32")
    argmax_v1 = T.match_buffer(var_argmax_v1, [m], dtype="int32")
    for i0, i1 in T.grid(m, n):
        with T.block("argmax"):
            i, k = T.axis.remap("SR", [i0, i1])
            T.reads(val[i, k], idx[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = T.min_value("float32")
                argmax_v1[i] = T.int32(-1)
            v_argmax_v0: T.float32 = T.Select(argmax_v0[i] >= val[i, k], argmax_v0[i], val[i, k])
            v_argmax_v1: T.int32 = T.Select(argmax_v0[i] >= val[i, k], argmax_v1[i], idx[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def test_argmax_idx_val():
    _check_workload(te_argmax_idx_val, tir_argmax_idx_val)


def test_argmax_val_idx():
    _check_workload(te_argmax_val_idx, tir_argmax_val_idx)


def test_int64_indices():
    n = te.var("n", "int64")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    prim_func = te.create_prim_func([A, B])
    loop = prim_func.body.block.body
    assert loop.loop_var.dtype == "int64"
    assert loop.min.dtype == "int64"
    assert loop.extent.dtype == "int64"


def test_zero_dim_add():
    def te_func():
        a = te.placeholder((), name="a", dtype="int32")
        b = te.placeholder((), name="b", dtype="int32")
        c = te.compute(a.shape, lambda *i: a(*i) + b(*i), name="c")
        return [a, b, c]

    @T.prim_func
    def expected(
        a: T.Buffer((), "int32"),
        b: T.Buffer((), "int32"),
        c: T.Buffer((), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        with T.block("root"):
            T.reads()
            T.writes()
            with T.block("c"):
                vi = T.axis.spatial(1, 0)
                T.reads(a[()], b[()])
                T.writes(c[()])
                c[()] = a[()] + b[()]

    _check_workload(te_func, expected)


def te_reshape():
    # The following is possible to be generated by TOPI. So we test this case.
    A = te.placeholder((tvm.tir.IntImm("int64", 2), tvm.tir.IntImm("int64", 4)), name="A")
    B = topi.reshape(A, (4, 2))
    return [A, B]


@T.prim_func
def tir_reshape(
    A: T.Buffer((T.int64(2), T.int64(4)), "float32"),
    T_reshape: T.Buffer((T.int64(4), T.int64(2)), "float32"),
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i0, i1 in T.grid(T.int64(4), T.int64(2)):
        with T.block("T_reshape"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(
                A[
                    (ax0 * T.int64(2) + ax1) % T.int64(8) // T.int64(4),
                    (ax0 * T.int64(2) + ax1) % T.int64(4),
                ]
            )
            T.writes(T_reshape[ax0, ax1])
            T_reshape[ax0, ax1] = A[
                (ax0 * T.int64(2) + ax1) % T.int64(8) // T.int64(4),
                (ax0 * T.int64(2) + ax1) % T.int64(4),
            ]


def test_reshape():
    _check_workload(te_reshape, tir_reshape, index_dtype_override="int64")


@T.prim_func
def argmax_expected(
    p0: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "uint8"),
    p0_red: T.Buffer((T.int64(1), T.int64(56), T.int64(56)), "int32"),
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    p0_red_temp_v0 = T.alloc_buffer([T.int64(1), T.int64(56), T.int64(56)], dtype="int32")
    p0_red_temp_v1 = T.alloc_buffer([T.int64(1), T.int64(56), T.int64(56)], dtype="uint8")
    for ax0, ax1, ax2, k1 in T.grid(T.int64(1), T.int64(56), T.int64(56), T.int64(64)):
        with T.block("p0_red_temp"):
            v_ax0, v_ax1, v_ax2, v_k1 = T.axis.remap("SSSR", [ax0, ax1, ax2, k1])
            T.reads(p0[v_ax0, v_k1, v_ax1, v_ax2])
            T.writes(p0_red_temp_v0[v_ax0, v_ax1, v_ax2], p0_red_temp_v1[v_ax0, v_ax1, v_ax2])
            with T.init():
                p0_red_temp_v0[v_ax0, v_ax1, v_ax2] = -1
                p0_red_temp_v1[v_ax0, v_ax1, v_ax2] = T.uint8(0)
            v_p0_red_temp_v0: T.int64 = T.Select(
                p0_red_temp_v1[v_ax0, v_ax1, v_ax2] > p0[v_ax0, v_k1, v_ax1, v_ax2]
                or (
                    p0_red_temp_v1[v_ax0, v_ax1, v_ax2] == p0[v_ax0, v_k1, v_ax1, v_ax2]
                    and T.Cast("int64", p0_red_temp_v0[v_ax0, v_ax1, v_ax2]) < v_k1
                ),
                T.Cast("int64", p0_red_temp_v0[v_ax0, v_ax1, v_ax2]),
                v_k1,
            )
            v_p0_red_temp_v1: T.uint8 = T.Select(
                p0_red_temp_v1[v_ax0, v_ax1, v_ax2] > p0[v_ax0, v_k1, v_ax1, v_ax2],
                p0_red_temp_v1[v_ax0, v_ax1, v_ax2],
                p0[v_ax0, v_k1, v_ax1, v_ax2],
            )
            p0_red_temp_v0[v_ax0, v_ax1, v_ax2] = T.Cast("int32", v_p0_red_temp_v0)
            p0_red_temp_v1[v_ax0, v_ax1, v_ax2] = v_p0_red_temp_v1
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
        with T.block("p0_red"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(p0_red_temp_v0[v_ax0, v_ax1, v_ax2])
            T.writes(p0_red[v_ax0, v_ax1, v_ax2])
            p0_red[v_ax0, v_ax1, v_ax2] = p0_red_temp_v0[v_ax0, v_ax1, v_ax2]


def test_argmax():
    data = relay.var("data", shape=(1, 64, 56, 56), dtype="uint8")
    mod = tvm.IRModule.from_expr(relay.argmax(data, axis=1))

    target = tvm.target.Target("llvm")

    opt_mod, _ = relay.optimize(mod, params={}, target=target)

    prim_func = relay.backend.te_compiler.lower_to_primfunc(opt_mod["main"].body.op, target)

    tvm.ir.assert_structural_equal(prim_func, argmax_expected)


def te_resize2d_symbolic():
    oh = tir.Var("oh", "int64")
    ow = tir.Var("ow", "int64")
    roi = (0.0, 0.0, 0.0, 0.0)
    A = te.placeholder((2, 3, 128, 128), "float32", name="A")
    B = topi.image.resize2d(
        A,
        roi,
        size=(oh, ow),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="round",
    )
    return [A, B]


@T.prim_func
def tir_resize2d_symbolic(
    A: T.Buffer((T.int64(2), T.int64(3), T.int64(128), T.int64(128)), "float32"),
    var_resize: T.handle,
):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    oh = T.int64()
    ow = T.int64()
    resize = T.match_buffer(var_resize, [T.int64(2), T.int64(3), oh, ow], dtype="float32")
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), oh, ow):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, T.int64(0) : T.int64(128), T.int64(0) : T.int64(128)])
            T.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = A[
                v_i0,
                v_i1,
                T.max(
                    T.min(
                        T.Cast(
                            "int64",
                            T.round(
                                T.float32(128) / T.Cast("float32", oh) * T.Cast("float32", v_i2),
                                dtype="float32",
                            ),
                        ),
                        T.int64(127),
                    ),
                    T.int64(0),
                ),
                T.max(
                    T.min(
                        T.Cast(
                            "int64",
                            T.round(
                                T.float32(128) / T.Cast("float32", ow) * T.Cast("float32", v_i3),
                                dtype="float32",
                            ),
                        ),
                        T.int64(127),
                    ),
                    T.int64(0),
                ),
            ]


def test_resize2d_symbolic():
    _check_workload(te_resize2d_symbolic, tir_resize2d_symbolic, index_dtype_override="int64")


def test_extern_with_explicit_buffer_access():
    def te_extern():
        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")
        P = te.placeholder((1,), name="P")
        C = te.extern(
            (128, 128),
            [A, B, P],
            lambda ins, outs: tvm.tir.call_extern(
                "", "myfunc", ins[0].data, ins[1].data, outs[0].data, ins[2][0]
            ),
            name="C",
        )
        return [A, B, P, C]

    @T.prim_func
    def tir_extern(var_A: T.handle, var_B: T.handle, var_P: T.handle, var_C: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(var_A, [128, 128], dtype="float32", offset_factor=1)
        B = T.match_buffer(var_B, [128, 128], dtype="float32", offset_factor=1)
        P = T.match_buffer(var_P, [1], dtype="float32", offset_factor=1)
        C = T.match_buffer(var_C, [128, 128], dtype="float32", offset_factor=1)
        with T.block("C"):
            T.reads()
            T.writes()
            T.call_extern("myfunc", A.data, B.data, C.data, P[0], dtype="")

    _check_workload(te_extern, tir_extern)


def te_slice_with_var_input():
    idx = te.var("idx", dtype="int64")
    m = te.var("m", dtype="int64")
    n = te.var("n", dtype="int64")
    tensor = te.placeholder((m, n), name="tensor")
    slice0 = te.compute((idx, n), lambda i, j: tensor[i, j], name="slice")
    return [tensor, idx, slice0]


@T.prim_func
def tir_slice_with_var_input(var_tensor: T.handle, idx: T.int64, var_slice: T.handle):
    T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
    m, n = T.int64(), T.int64()
    tensor = T.match_buffer(var_tensor, (m, n))
    slice = T.match_buffer(var_slice, (idx, n))
    # with T.block("root"):
    for i, j in T.grid(idx, n):
        with T.block("slice"):
            v_i = T.axis.spatial(idx, i)
            v_j = T.axis.spatial(n, j)
            T.reads(tensor[v_i, v_j])
            T.writes(slice[v_i, v_j])
            slice[v_i, v_j] = tensor[v_i, v_j]


def test_with_var_input():
    _check_workload(te_slice_with_var_input, tir_slice_with_var_input, index_dtype_override="int64")


def test_loop_aware_initial_value():
    """Test initial value aware of spatial iter position"""

    @T.prim_func
    def tir_workload(var_a: T.handle, var_b: T.handle, var_sum_red: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
        a = T.match_buffer(var_a, (5, 5))
        b = T.match_buffer(var_b, (5,))
        sum_red = T.match_buffer(var_sum_red, (5,))
        for i, ax in T.grid(5, 5):
            with T.block("sum_red"):
                v_i, v_ax = T.axis.remap("SR", [i, ax])
                T.reads(b[v_i], a[v_i, v_ax])
                T.writes(sum_red[v_i])
                with T.init():
                    sum_red[v_i] = b[v_i]
                sum_red[v_i] = sum_red[v_i] + a[v_i, v_ax]

    def te_workload():
        data = te.placeholder((5, 5), "float32", "a")
        init = te.placeholder((5,), "float32", "b")
        ax = te.reduce_axis((0, 5), "ax")
        sum_red = te.compute(
            (5,),
            lambda i: te.comm_reducer(
                lambda x, y: x + y,
                lambda t: init[i],
            )(data[i, ax], axis=[ax]),
            name="sum_red",
        )
        return [data, init, sum_red]

    _check_workload(te_workload, tir_workload)


def test_loop_aware_reducer_combiner():
    """Test combiner aware of spatial iter position"""

    @T.prim_func
    def tir_workload(var_a: T.handle, var_b: T.handle, var_sum_red: T.handle):
        T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
        a = T.match_buffer(var_a, (5, 5))
        b = T.match_buffer(var_b, (5,))
        sum_red = T.match_buffer(var_sum_red, (5,))
        for i, ax in T.grid(5, 5):
            with T.block("sum_red"):
                v_i = T.axis.spatial(5, i)
                v_ax = T.axis.reduce(5, ax)
                T.reads(a[v_i, 0:5])
                T.writes(sum_red[v_i])
                with T.init():
                    sum_red[v_i] = T.float32(0.0)
                sum_red[v_i] = T.if_then_else(
                    a[v_i, sum_red[v_i]] < a[v_i, v_ax], sum_red[v_i], T.Cast("float32", v_ax)
                )

    def te_workload():
        data = te.placeholder((5, 5), "float32", "a")
        init = te.placeholder((5,), "float32", "b")
        ax = te.reduce_axis((0, 5), "ax")
        sum_red = te.compute(
            (5,),
            lambda i: te.comm_reducer(
                lambda x, y: te.if_then_else(data[i, x] < y, x, ax),
                lambda _: te.const(0, "float32"),
            )(data[i, ax], axis=[ax]),
            name="sum_red",
        )
        return [data, init, sum_red]

    _check_workload(te_workload, tir_workload)


def test_adaptive_pooling_window():
    @T.prim_func
    def tir_workload(
        x: T.Buffer((1, 1024, 16, 40), "float32"),
        adaptive_pool_avg: T.Buffer((1, 1024, 12, 30), "float32"),
    ):
        T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
        # fmt: off
        adaptive_pool_sum = T.alloc_buffer((1, 1024, 12, 30))
        for ax0, ax1, ax2, ax3 in T.grid(1, 1024, 12, 30):
            with T.block("adaptive_pool_sum_1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(x[v_ax0, v_ax1, v_ax2 * 16 // 12:v_ax2 * 16 // 12 + ((v_ax2 % 3 * 4 + 16) // 12 + 1), v_ax3 * 40 // 30:v_ax3 * 40 // 30 + ((v_ax3 % 3 * 10 + 40) // 30 + 1)])
                T.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                for rv0, rv1 in T.grid(T.Select((v_ax2 * 4 + 4) % 12 == 0, (v_ax2 * 16 + 16) // 12, (v_ax2 * 16 + 16) // 12 + 1) - v_ax2 * 16 // 12, T.Select((v_ax3 * 10 + 10) % 30 == 0, (v_ax3 * 40 + 40) // 30, (v_ax3 * 40 + 40) // 30 + 1) - v_ax3 * 40 // 30):
                    with T.block("adaptive_pool_sum"):
                        v_ax0_1 = T.axis.spatial((v_ax0, v_ax0 + 1), v_ax0)
                        v_ax1_1 = T.axis.spatial((v_ax1, v_ax1 + 1), v_ax1)
                        v_ax2_1 = T.axis.spatial((v_ax2, v_ax2 + 1), v_ax2)
                        v_ax3_1 = T.axis.spatial((v_ax3, v_ax3 + 1), v_ax3)
                        v_rv0, v_rv1 = T.axis.remap("RR", [rv0, rv1])
                        T.reads(x[v_ax0_1, v_ax1_1, v_ax2_1 * 16 // 12 + v_rv0, v_ax3_1 * 40 // 30 + v_rv1])
                        T.writes(adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1])
                        with T.init():
                            adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] = T.float32(0.0)
                        adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] = adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] + x[v_ax0_1, v_ax1_1, v_ax2_1 * 16 // 12 + v_rv0, v_ax3_1 * 40 // 30 + v_rv1]
        for ax0, ax1, ax2, ax3 in T.grid(1, 1024, 12, 30):
            with T.block("adaptive_pool_avg"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / (T.Cast("float32", T.Select((v_ax2 * 4 + 4) % 12 == 0, (v_ax2 * 16 + 16) // 12, (v_ax2 * 16 + 16) // 12 + 1) - v_ax2 * 16 // 12) * T.Cast("float32", T.Select((v_ax3 * 10 + 10) % 30 == 0, (v_ax3 * 40 + 40) // 30, (v_ax3 * 40 + 40) // 30 + 1) - v_ax3 * 40 // 30))
        # fmt: on

    def te_workload():
        x = te.placeholder([1, 1024, 16, 40], "float32", "x")
        y = topi.nn.adaptive_pool(x, [12, 30], pool_type="avg")
        f = te.create_prim_func([x, y])
        return [x, y]

    _check_workload(te_workload, tir_workload)


def test_nested_reduce_domain_dependency():
    @T.prim_func
    def tir_workload(
        x: T.Buffer((8, 8, 8, 8, 8), "float32"), compute: T.Buffer((8, 8, 8), "float32")
    ):
        T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})
        for i0, i1, i2 in T.grid(8, 8, 8):
            with T.block("compute_2"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(x[v_i0, v_i1, v_i2, 0:v_i1, 0 : v_i1 - 1])
                T.writes(compute[v_i0, v_i1, v_i2])
                for rv in range(v_i1):
                    with T.block("compute_1"):
                        v_i0_1 = T.axis.spatial((v_i0, v_i0 + 1), v_i0)
                        v_i1_1 = T.axis.spatial((v_i1, v_i1 + 1), v_i1)
                        v_i2_1 = T.axis.spatial((v_i2, v_i2 + 1), v_i2)
                        v_rv = T.axis.reduce(v_i1, rv)
                        T.reads(x[v_i0_1, v_i1_1, v_i2_1, v_rv, 0:v_rv])
                        T.writes(compute[v_i0_1, v_i1_1, v_i2_1])
                        with T.init():
                            compute[v_i0_1, v_i1_1, v_i2_1] = T.float32(0.0)
                        for rv_1 in range(v_rv):
                            with T.block("compute"):
                                v_i0_2 = T.axis.spatial((v_i0_1, v_i0_1 + 1), v_i0_1)
                                v_i1_2 = T.axis.spatial((v_i1_1, v_i1_1 + 1), v_i1_1)
                                v_i2_2 = T.axis.spatial((v_i2_1, v_i2_1 + 1), v_i2_1)
                                v_rv_1 = T.axis.reduce((v_rv, v_rv + 1), v_rv)
                                v_rv_2 = T.axis.reduce(v_rv, rv_1)
                                T.reads(x[v_i0_2, v_i1_2, v_i2_2, v_rv_1, v_rv_2])
                                T.writes(compute[v_i0_2, v_i1_2, v_i2_2])
                                with T.init():
                                    compute[v_i0_2, v_i1_2, v_i2_2] = T.float32(0.0)
                                compute[v_i0_2, v_i1_2, v_i2_2] = (
                                    compute[v_i0_2, v_i1_2, v_i2_2]
                                    + x[v_i0_2, v_i1_2, v_i2_2, v_rv_1, v_rv_2]
                                )

    def te_workload():
        x = te.placeholder([8, 8, 8, 8, 8], "float32", "x")

        def fcompute(*axes):
            r1 = te.reduce_axis(tvm.ir.Range.from_min_extent(0, axes[1]))
            r2 = te.reduce_axis(tvm.ir.Range.from_min_extent(0, r1))
            all_axes = [*axes, r1, r2]
            return te.sum(x(*all_axes), [r1, r2])

        y = te.compute([8, 8, 8], fcompute)
        f = te.create_prim_func([x, y])
        return [x, y]

    _check_workload(te_workload, tir_workload)


if __name__ == "__main__":
    tvm.testing.main()
