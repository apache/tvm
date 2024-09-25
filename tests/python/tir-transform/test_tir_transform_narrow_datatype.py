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
from tvm import relay, te
from tvm.driver.build_module import schedule_to_module
from tvm.script import tir as T
from tvm.tir import const
import tvm.testing


def lower_stmt(params, stmt, target_bits):
    func = tvm.tir.PrimFunc(params, stmt)
    func = tvm.tir.transform.NarrowDataType(target_bits)(tvm.IRModule.from_expr(func))["main"]
    stmt = func.body
    return stmt


def lower_sch(sch, args, target_bits, extra_passes=None):
    binds = {}
    arg_list = []
    for x in args:
        if isinstance(x, te.tensor.Tensor):
            buf = tvm.tir.decl_buffer(x.shape, dtype=x.dtype, name=x.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    sch = sch.normalize()

    mod = schedule_to_module(sch, args)
    mod = tvm.tir.transform.StorageFlatten(64)(mod)
    if extra_passes:
        for p in extra_passes:
            mod = p(mod)
    return tvm.tir.transform.NarrowDataType(target_bits)(mod)["main"].body


def test_basic():
    def check(m, n, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer([m * n], name="A")
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer([m * n], name="B")
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name="i") as i:
            with ib.for_range(0, n, name="j") as j:
                B[i * n + j] = A[i * n + j] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.loop_var.dtype == target_dtype
        assert stmt.body.loop_var.dtype == target_dtype

    # const shape
    # i32 -> i32
    check(2, 2, 32, "int32")
    # i64 -> i32
    check(const(2, dtype="int64"), const(2, dtype="int64"), 32, "int32")
    check(const(2**16, dtype="int64"), const(2**16, dtype="int64"), 32, "int64")
    # i32 -> i16
    check(2, 2, 16, "int16")
    check(2**10, 2**10, 16, "int32")

    # symbolic shape
    check(te.size_var(name="m", dtype="int32"), te.size_var(name="n", dtype="int32"), 32, "int32")
    check(te.size_var(name="m", dtype="int64"), te.size_var(name="n", dtype="int64"), 32, "int64")


def test_thread_axis():
    def check(m, n, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer([m * n], name="A")
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer([m * n], name="B")
        B = ib.buffer_ptr(Bb)
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", m)
        ib.scope_attr(tx, "thread_extent", n)
        B[bx * n + tx] = A[bx * n + tx] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.node.var.dtype == target_dtype
        assert stmt.body.node.var.dtype == target_dtype

    # i32 -> i32
    check(2, 32, target_bits=32, target_dtype="int32")
    # i64 -> i32
    check(const(2, dtype="int64"), const(32, dtype="int64"), target_bits=32, target_dtype="int32")
    check(
        const(2**30, dtype="int64"),
        const(32, dtype="int64"),
        target_bits=32,
        target_dtype="int64",
    )
    # i32 -> i16
    check(2, 32, target_bits=16, target_dtype="int16")
    check(2**14, 32, target_bits=16, target_dtype="int32")


def test_thread_axis_2():
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(T_reshape: T.Buffer((1, 12, 384, 384), "float32"), placeholder_1: T.Buffer((T.int64(1), T.int64(12), T.int64(384), 384), "bool"), T_where: T.Buffer((T.int64(1), T.int64(12), T.int64(384), 384), "float32")) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # body
            # with T.block("root")
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for i0_i1_i2_i3_fused_0 in T.serial(T.int64(7)):
                        with T.block("T_where"):
                            ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                            ax1 = T.axis.spatial(T.int64(12), ((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(1769472) // T.int64(147456))
                            ax2 = T.axis.spatial(T.int64(384), ((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(147456) // T.int64(384))
                            ax3 = T.axis.spatial(384, T.cast(((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(384), "int32"))
                            T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(1769472))
                            T.reads(placeholder_1[ax0, ax1, ax2, ax3], T_reshape[ax0, ax1, ax2, ax3])
                            T.writes(T_where[ax0, ax1, ax2, ax3])
                            T_where[ax0, ax1, ax2, ax3] = T.Select(T.cast(placeholder_1[ax0, ax1, ax2, ax3], "int32") != 0, T.float32(-1000000000), T_reshape[ax0, ax1, ax2, ax3])
    # fmt: on
    # TODO(@junrushao1994): make this test more "unit" after the new TVMScript printer/parser lands
    tvm.lower(Before)


def test_multilanes():
    def check(m, lanes, target_bits, target_dtype):
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer((m,), dtype="float32x{}".format(lanes), name="A")
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer((m,), dtype="float32x{}".format(lanes), name="B")
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name="i", dtype=m.dtype) as i:
            B[i] = A[i] + 1
        A[0] = B[1]
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.seq[0].loop_var.dtype == target_dtype

    # i32 -> i32
    check(const(2**10, dtype="int32"), 2, target_bits=32, target_dtype="int32")
    # i64 -> i32
    check(const(2**10, dtype="int64"), 2, target_bits=32, target_dtype="int32")
    check(const(2**32, dtype="int64"), 2, target_bits=32, target_dtype="int64")
    # i32 -> i16
    check(const(2**10, dtype="int32"), 2, target_bits=16, target_dtype="int16")
    check(const(2**16, dtype="int32"), 2, target_bits=16, target_dtype="int32")


def test_reduce():
    def check(m, target_bits, target_dtype):
        A = te.placeholder((m,), name="A", dtype="float32")
        k = te.reduce_axis((0, m), "k")
        B = te.compute((), lambda *idx: te.sum(A[k], axis=k), name="B")
        s = te.create_schedule(B.op)
        stmt = lower_sch(s, [A, B], target_bits)
        assert stmt[1].loop_var.dtype == target_dtype

    # i32 -> i32
    check(const(64, dtype="int32"), 32, "int32")
    # i64 -> i32
    check(const(64, dtype="int64"), 32, "int32")
    # i32 -> i16
    check(const(64, dtype="int32"), 16, "int16")
    check(const(2**16, dtype="int32"), 16, "int32")
    # symbolic
    check(te.var("n", dtype="int32"), 32, "int32")
    check(te.var("n", dtype="int64"), 32, "int64")


def test_slice():
    def check(m, n, target_bits, target_dtype):
        # The index may overflow in B, while not in A
        ib = tvm.tir.ir_builder.create()
        Ab = tvm.tir.decl_buffer([m * n], name="A")
        A = ib.buffer_ptr(Ab)
        Bb = tvm.tir.decl_buffer([m * n * 2], name="B")
        B = ib.buffer_ptr(Bb)
        with ib.for_range(0, m, name="i") as i:
            with ib.for_range(0, n, name="j") as j:
                A[i * n + j] = B[i * 2 * n + 2 * j] + 1
        stmt = ib.get()
        stmt = lower_stmt([Ab, Bb], stmt, target_bits)
        assert stmt.loop_var.dtype == target_dtype
        assert stmt.body.loop_var.dtype == target_dtype

    # The maximum index is (2**15 * 2**15 - 1) * 2 <= 2**31 - 1
    check(const(2**15, "int64"), const(2**15, "int64"), target_bits=32, target_dtype="int32")
    # The maximum index is (2**15 * 2**15 - 1 + 2**15) * 2 > 2**31 - 1
    check(
        const(2**15, "int64"), const((2**15 + 1), "int64"), target_bits=32, target_dtype="int64"
    )


def test_relay_basic():
    engine = relay.backend.te_compiler.get()

    def check(shapex, shapey, target_bits, target_dtype):
        x = relay.var("x", shape=shapex)
        y = relay.var("y", shape=shapey)
        z = relay.add(x, y)
        func = relay.Function([x, y], z)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        func = mod["main"]
        z = engine.lower(func, "llvm")
        stmt = lower_sch(z.schedule, tuple(z.inputs) + tuple(z.outputs), 32)
        # outer loop
        assert stmt.loop_var.dtype == target_dtype
        # inner loop
        if len(shapex) > 1 or len(shapey) > 1:
            assert stmt.body.loop_var.dtype == target_dtype

    check(
        (const(2**16, "int64"), const(2**15 + 1, "int64")),
        (1, const(2**15 + 1, "int64")),
        target_bits=32,
        target_dtype="int64",
    )
    check(
        (const(2**16, "int64"), const(2**15, "int64")),
        (1, const(2**15, "int64")),
        target_bits=32,
        target_dtype="int32",
    )
    check(
        (const(2**31, "int64"),), (const(2**31, "int64"),), target_bits=32, target_dtype="int32"
    )
    check(
        (const(2**31 + 1, "int64"),),
        (const(2**31 + 1, "int64"),),
        target_bits=32,
        target_dtype="int64",
    )


def test_relay_take():
    engine = relay.backend.te_compiler.get()

    def check(shape, index, target_bits, target_dtype):
        x = relay.var("x", shape=shape)
        y = relay.op.take(x, indices=index)
        func = relay.Function([x], y)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)
        func = mod["main"]
        z = engine.lower(func, "llvm")
        stmt = lower_sch(z.schedule, tuple(z.inputs) + tuple(z.outputs), 32)
        assert stmt.value.indices[0].dtype == target_dtype

    check(
        (const(2**16, "int64"), const(2**15 + 1, "int64")),
        relay.const(0, dtype="int64"),
        target_bits=32,
        target_dtype="int32",
    )
    check(
        (const(2**16, "int64"), const(2**15 + 1, "int64")),
        relay.const(2**31, dtype="int64"),
        target_bits=32,
        target_dtype="int64",
    )


def test_ramp_dtype_consistency():
    """
    for (i :int64, (int64)0, (int64)4) {
        A[ramp(i*(int64)2, (int64)1, 2)] = cast(int64, 2 ** 31 - 1) * i;
    }
    The infer result:
        base:   int64 -> int64 (since i is involved in another int64 expr)
        stride: int64 -> int32

    Thus ramp should still use int64 for both stride and base after rewrite.
    """
    n = tvm.tir.IntImm("int64", 4)
    m = tvm.tir.IntImm("int64", 2)
    A = te.compute((n, m), lambda i, j: tvm.tir.Cast("int64", 2**31 - 1) * i, name="A")
    s = te.create_schedule(A.op)
    s[A].vectorize(A.op.axis[1])
    lower_sch(s, [A], 32, extra_passes=[tvm.tir.transform.VectorizeLoop()])


def test_condition():
    @T.prim_func
    def before(A: T.Buffer((128,), "float32"), B: T.Buffer((130,), "float32")):
        for i, j in T.grid(T.int64(2), T.int64(65)):
            if i * T.int64(65) + j >= T.int64(0) and i * T.int64(65) + j < T.int64(128):
                A[i * T.int64(65) + j] = 0.0
        for i, j in T.grid(T.int64(2), T.int64(65)):
            B[i * T.int64(65) + j] = T.if_then_else(
                i * T.int64(65) + j >= T.int64(0) and i * T.int64(65) + j < T.int64(128),
                A[i * T.int64(65) + j],
                0.0,
                dtype="float32",
            )

    @T.prim_func
    def expected_after(A: T.Buffer(128, "float32"), B: T.Buffer(130, "float32")):
        for i, j in T.grid(2, 65):
            if i * 65 + j >= 0 and i * 65 + j < 128:
                A[i * 65 + j] = T.float32(0)
        for i, j in T.grid(2, 65):
            B[i * 65 + j] = T.if_then_else(
                i * 65 + j >= 0 and i * 65 + j < 128, A[i * 65 + j], T.float32(0), dtype="float32"
            )

    after = tvm.tir.transform.NarrowDataType(32)(
        tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    )["main"]
    tvm.ir.assert_structural_equal(after, expected_after.with_attr("global_symbol", "main"))


def test_block():
    @T.prim_func
    def before(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(8)):
                with T.block():
                    vi = T.axis.spatial(T.int64(128), i * T.int64(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    @T.prim_func
    def expected_after(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int32(16)):
            for j in T.serial(0, T.int32(8)):
                with T.block():
                    vi = T.axis.spatial(T.int32(128), i * T.int32(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    after = tvm.tir.transform.NarrowDataType(32)(
        tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    )["main"]
    tvm.ir.assert_structural_equal(after, expected_after.with_attr("global_symbol", "main"))


def test_avg_pool2d():
    @T.prim_func
    def before(PSUM: T.Buffer((313600,), "int32"), PAVG: T.Buffer((313600,), "int32")):
        for j in T.parallel(T.int64(0), T.int64(280)):
            for i in T.serial(T.int64(0), T.int64(35)):
                for vi in T.vectorized(T.int64(0), T.int64(32)):
                    PAVG[(((j * T.int64(1120)) + (i * T.int64(32))) + vi)] = T.cast(
                        T.Div(
                            T.cast(PSUM[(((j * T.int64(1120)) + (i * T.int64(32))) + vi)], "int64"),
                            T.max(
                                (
                                    (
                                        (
                                            T.min(
                                                T.int64(1),
                                                (T.int64(34) - T.floormod(j, T.int64(35))),
                                            )
                                            + T.int64(2)
                                        )
                                        - T.max(
                                            (T.int64(1) - T.floormod(j, T.int64(35))), T.int64(0)
                                        )
                                    )
                                    * (
                                        (T.min(T.int64(1), (T.int64(34) - i)) + T.int64(2))
                                        - T.max((T.int64(1) - i), T.int64(0))
                                    )
                                ),
                                T.int64(1),
                            ),
                        ),
                        "int32",
                    )

    @T.prim_func
    def expected_after(PSUM: T.Buffer((313600,), "int32"), PAVG: T.Buffer((313600,), "int32")):
        for j in T.parallel(T.int32(0), T.int32(280)):
            for i in T.serial(T.int32(0), T.int32(35)):
                for vi in T.vectorized(T.int32(0), T.int32(32)):
                    PAVG[(((j * T.int32(1120)) + (i * T.int32(32))) + vi)] = T.Div(
                        PSUM[(((j * T.int32(1120)) + (i * T.int32(32))) + vi)],
                        (
                            (
                                (
                                    T.min(T.int32(1), (T.int32(34) - T.floormod(j, T.int32(35))))
                                    + T.int32(2)
                                )
                                - T.max((T.int32(1) - T.floormod(j, T.int32(35))), T.int32(0))
                            )
                            * (
                                (T.min(T.int32(1), (T.int32(34) - i)) + T.int32(2))
                                - T.max((T.int32(1) - i), T.int32(0))
                            )
                        ),
                    )

    after = tvm.tir.transform.NarrowDataType(32)(
        tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    )
    after = tvm.tir.transform.Simplify()(after)
    tvm.ir.assert_structural_equal(after["main"], expected_after.with_attr("global_symbol", "main"))


def test_narrow_i64_valued_bufferload_index_to_i32():
    @T.prim_func
    def before(A: T.Buffer((16,), "int64")):
        for i in range(T.int64(15)):
            A[i + T.int64(1)] = A[i] + T.int64(1)

    @T.prim_func
    def expect(A: T.Buffer((16,), "int64")):
        for i in range(15):
            A[i + 1] = A[i] + T.int64(1)

    after = tvm.tir.transform.NarrowDataType(32)(
        tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    )["main"]
    tvm.ir.assert_structural_equal(after, expect.with_attr("global_symbol", "main"))


if __name__ == "__main__":
    tvm.testing.main()
