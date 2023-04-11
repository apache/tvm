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
from tvm import te
from tvm.script import tir as T
import numpy as np


@tvm.register_func("tvm.test_matmul")
def my_matmul(a, b, c):
    c.copyfrom(np.dot(a.numpy(), b.numpy()))


def check_packed_func(target="llvm"):
    ib = tvm.tir.ir_builder.create()

    m = n = k = 16

    #
    # Prepare buffer for a, b and c:
    #
    a = te.placeholder((m, k), name="a", dtype="float64")
    b = te.placeholder((k, n), name="b", dtype="float64")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute((m, n), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="c")

    a_buffer = tvm.tir.decl_buffer(
        a.shape, a.dtype, name="a_buffer", offset_factor=1, strides=[te.var("s1"), 1]
    )
    b_buffer = tvm.tir.decl_buffer(
        b.shape, b.dtype, name="b_buffer", offset_factor=1, strides=[te.var("s2"), 1]
    )
    c_buffer = tvm.tir.decl_buffer(
        c.shape, c.dtype, name="c_buffer", offset_factor=1, strides=[te.var("s3"), 1]
    )

    with ib.for_range(0, 10, "i", kind="parallel"):
        ib.emit(tvm.tir.call_packed("tvm.test_matmul", a_buffer, b_buffer, c_buffer))

    stmt = ib.get()

    # Construct a valid IRModule to be lowered:
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([a_buffer, b_buffer, c_buffer], stmt))

    target = tvm.target.Target(target)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod)
    mod = tvm.tir.transform.Apply(lambda f: f.with_attr("global_symbol", "main"))(mod)
    mod = tvm.tir.transform.MakePackedAPI()(mod)

    # Do the lowering:
    mod = tvm.tir.transform.LowerTVMBuiltin()(mod)

    # Get the PrimFunc from module:
    prim_func = mod.functions.items()[0][1]

    node = prim_func.body

    # Recursively visit PrimFunc until we meet the for-loop:
    while isinstance(node, (tvm.tir.AssertStmt, tvm.tir.LetStmt, tvm.tir.AttrStmt)):
        node = node.body

    # For-loop:
    assert isinstance(node, tvm.tir.stmt.For)

    #
    # let stack_tcode = tir.tvm_stack_alloca("arg_tcode", 4)
    #
    alloca_tcode = node.body
    assert isinstance(alloca_tcode, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "arg_tcode", 4
    )
    expected_var = alloca_tcode.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_tcode.body)

    tvm.ir.assert_structural_equal(alloca_tcode, expected_stmt, map_free_vars=True)

    #
    # let stack_value = tir.tvm_stack_alloca("arg_value", 4)
    #
    alloca_value = alloca_tcode.body
    assert isinstance(alloca_value, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "arg_value", 4
    )
    expected_var = alloca_value.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_value.body)

    tvm.ir.assert_structural_equal(alloca_value, expected_stmt, map_free_vars=True)

    #
    # let stack_array = tir.tvm_stack_alloca("array", 3)
    #
    alloca_array = alloca_value.body
    assert isinstance(alloca_array, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "array", 3
    )
    expected_var = alloca_array.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_array.body)

    tvm.ir.assert_structural_equal(alloca_array, expected_stmt, map_free_vars=True)

    #
    # let stack_shape = tir.tvm_stack_alloca("shape", 12)
    #
    alloca_shape = alloca_array.body
    assert isinstance(alloca_shape, tvm.tir.LetStmt)

    expected_value = tvm.tir.call_intrin(
        "handle", tvm.ir.Op.get("tir.tvm_stack_alloca"), "shape", 12
    )
    expected_var = alloca_shape.var
    expected_stmt = tvm.tir.LetStmt(expected_var, expected_value, alloca_shape.body)

    tvm.ir.assert_structural_equal(alloca_shape, expected_stmt, map_free_vars=True)


def test_lower_packed_func():
    check_packed_func("llvm")
    check_packed_func("stackvm")


@tvm.testing.requires_llvm
def test_call_packed_return_non_i32():
    # This call packed that return non i32 types
    expected_value = np.array([1.2, 1.4], dtype="float32")

    def packed_echo(value):
        return tvm.tir.call_intrin(
            value.dtype, tvm.ir.Op.get("tir.tvm_call_packed"), "testing.echo", value
        )

    def build_tir():
        Ab = tvm.tir.decl_buffer((2,), "float32")
        ib = tvm.tir.ir_builder.create()
        Aptr = ib.buffer_ptr(Ab)
        # return f32
        # Aptr[0] = testing.echo(expected_value[0])
        Aptr[0] = packed_echo(tvm.tir.const(expected_value[0], "float32"))
        # return handle
        # let Aptr_var = testing.echo(Aptr) in Aptr_var[1] = expected_value[1]
        Aptr_var = ib.let("Aptr_dup", packed_echo(Aptr.asobject().data))
        ib.emit(tvm.tir.BufferStore(Aptr, tvm.tir.const(expected_value[1], "float32"), [1]))

        stmt = ib.get()
        return tvm.IRModule.from_expr(
            tvm.tir.PrimFunc([Ab], stmt).with_attr("global_symbol", "packed_test")
        )

    mod = build_tir()
    f = tvm.build(mod, None, "llvm")
    a = tvm.nd.array(np.zeros(2, dtype="float32"))
    f(a)
    tvm.testing.assert_allclose(a.numpy(), expected_value)


def test_lower_overflow_int32():
    @T.prim_func
    def variance4(rxplaceholder: T.Buffer((T.int64(1), T.int64(32), T.int64(25690112)), "float32")):
        T.func_attr({"global_symbol": "variance4", "tir.noalias": True})
        rxplaceholder_red = T.allocate([32], "float32", "global")
        T_subtract = T.allocate([822083584], "float32", "global")
        rxplaceholder_red_1 = T.Buffer((T.int64(32),), data=rxplaceholder_red)
        rxplaceholder_1 = T.Buffer((T.int64(822083584),), data=rxplaceholder.data)
        T_subtract_1 = T.Buffer((T.int64(822083584),), data=T_subtract)
        for ax1, ax2 in T.grid(32, 25690112):
            cse_var_1: T.int32 = ax1 * 25690112 + ax2
            T_subtract_1[cse_var_1] = rxplaceholder_1[cse_var_1] - rxplaceholder_red_1[ax1]

    func = variance4
    tvm.build(func, target="llvm")  # should not crash


if __name__ == "__main__":
    test_call_packed_return_non_i32()
    test_lower_packed_func()
