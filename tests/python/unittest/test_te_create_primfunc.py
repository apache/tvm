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
import tvm
from tvm.script import ty
from tvm import te, tir


def test_unique_name():
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16, 16), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((16, 16), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_prim_func([A, C])
    s = tir.Schedule(func, debug_mode=True)
    assert isinstance(s.get_sref(s.get_block("main")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("main_1")), tir.schedule.StmtSRef)


def _check_workload(te_workload, tir_workload):
    func = te.create_prim_func(te_workload())
    tvm.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tir.Schedule(func, debug_mode=True)
    assert s


def te_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [A, B, C]


@tvm.script.tir
def tir_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128, tir.reduce_axis(0, 128)]) as [i, j, k]:
        with tir.init():
            C[i, j] = 0.0
        C[i, j] += A[i, k] * B[j, k]


def test_matmul():
    _check_workload(te_matmul, tir_matmul)


def te_element_wise():
    A = te.placeholder((128, 128), name="A")
    B = te.compute((128, 128), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((128, 128), lambda x, y: B[x, y] + 1, name="C")
    return [A, C]


@tvm.script.tir
def tir_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))

    with tir.block([128, 128]) as [i, j]:
        B[i, j] = A[i, j] * 2.0
    with tir.block([128, 128]) as [i, j]:
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


@tvm.script.tir
def tir_conv2d(a: ty.handle, w: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16, 14, 14])
    W = tir.match_buffer(w, [16, 3, 3, 32])
    B = tir.match_buffer(b, [16, 32, 14, 14])
    Apad = tir.alloc_buffer([16, 16, 16, 16])

    with tir.block([16, 16, 16, 16], "Apad") as [nn, cc, yy, xx]:
        Apad[nn, cc, yy, xx] = tir.if_then_else(
            yy >= 1 and yy - 1 < 14 and xx >= 1 and xx - 1 < 14,
            A[nn, cc, yy - 1, xx - 1],
            0.0,
            dtype="float32",
        )
    with tir.block(
        [16, 32, 14, 14, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "B"
    ) as [nn, ff, yy, xx, rc, ry, rx]:
        with tir.init():
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


@tvm.script.tir
def tir_multi_output(a0: ty.handle, a1: ty.handle, b0: ty.handle, b1: ty.handle) -> None:
    m = tir.var("int32")
    n = tir.var("int32")
    A0 = tir.match_buffer(a0, (m, n))
    A1 = tir.match_buffer(a1, (m, n))
    B0 = tir.match_buffer(b0, (m, n))
    B1 = tir.match_buffer(b1, (m, n))

    for i0, i1 in tir.grid(m, n):
        with tir.block([m, n], "B.v0") as [i, j]:
            B0[i, j] = A0[i, j] + 2.0
        with tir.block([m, n], "B.v1") as [i, j]:
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


@tvm.script.tir
def tir_extern(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    # body
    with tir.block([], "C"):
        tir.reads([A[0:128, 0:128], B[0:128, 0:128]])
        tir.writes([C[0:128, 0:128]])
        tir.evaluate(
            tir.tvm_call_packed(
                "tvm.contrib.cblas.matmul",
                tir.tvm_stack_make_array(
                    A.data,
                    tir.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                tir.tvm_stack_make_array(
                    B.data,
                    tir.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
                    dtype="handle",
                ),
                tir.tvm_stack_make_array(
                    C.data,
                    tir.tvm_stack_make_shape(128, 128, dtype="handle"),
                    0,
                    2,
                    0.0,
                    0,
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


@tvm.script.tir
def tir_reordered_matmul(c: ty.handle, a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128, tir.reduce_axis(0, 128)]) as [i, j, k]:
        with tir.init():
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


if __name__ == "__main__":
    test_unique_name()
    test_matmul()
    test_element_wise()
    test_conv2d()
    test_multi_output()
    test_extern()
    test_arg_order()
    test_error_reporting()
