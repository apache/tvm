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
"""Test the TIR codegen path of VM compiled mode.

Restrictions: all shape lowered, explicit allocation.
"""
import tvm
import tvm.testing
from tvm import relax
from tvm.ir import assert_structural_equal
from tvm.script import relax as R
from tvm.script import tir as T


def get_tir_mod(mod):
    builder = relax.ExecBuilder()
    return relax.vm_build._vmcodegen(builder, mod, exec_mode="compiled")


def test_add():
    @tvm.script.ir_module
    class Before:
        @R.function(pure=False)
        def foo(x: R.Tensor):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("test.vm.add", x, x, sinfo_args=(R.Tensor))
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__foo(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            T.func_attr({"global_symbol": "__vmtir__foo"})
            T.anylist_setitem_call_packed(
                r,
                T.int32(2),
                "test.vm.add",
                T.anylist_getitem(r, T.int32(0)),
                T.anylist_getitem(r, T.int32(0)),
            )
            T.anylist_setitem_call_packed(
                r, T.int32(1), "vm.builtin.copy", T.anylist_getitem(r, T.int32(2))
            )

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_tir_call():
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def shape_func(H: T.Buffer(T.int64(4), "int64")):
            T.func_attr({"global_symbol": "shape_func"})
            # generated compute function
            H[T.int64(0)] = H[T.int64(0)] + T.int64(1)

        @R.function(pure=False)
        def foo(x: R.Tensor([4], "int64")):
            R.func_attr({"global_symbol": "foo"})
            _ = Before.shape_func(x)
            return x

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def shape_func(H: T.Buffer(T.int64(4), "int64")):
            T.func_attr({"global_symbol": "shape_func"})
            # generated compute function
            H[T.int64(0)] = H[T.int64(0)] + T.int64(1)

        @T.prim_func
        def __vmtir__foo(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            T.func_attr({"global_symbol": "__vmtir__foo"})
            T.call_cpacked(
                "shape_func", T.anylist_getitem(r, T.int32(0)), T.reinterpret("handle", T.uint64(0))
            )
            T.anylist_setitem_call_packed(
                r, T.int32(1), "vm.builtin.copy", T.anylist_getitem(r, T.int32(0))
            )

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_if_cond():
    @tvm.script.ir_module
    class Before:
        @R.function(pure=False)
        def ife(cond: R.Tensor((), "bool"), x: R.Tensor) -> R.Tensor:
            R.func_attr({"global_symbol": "ife"})
            if cond:
                w = R.call_packed("test.vm.add", x, x, sinfo_args=(R.Tensor))
            else:
                w = R.call_packed("test.vm.mul", x, x, sinfo_args=(R.Tensor))
            return w

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__ife(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            T.func_attr({"global_symbol": "__vmtir__ife"})
            if T.Call(
                "bool",
                tvm.ir.Op.get("tir.tvm_call_packed"),
                ["vm.builtin.read_if_cond", T.anylist_getitem(r, T.int32(0))],
            ):
                T.anylist_setitem_call_packed(
                    r,
                    T.int32(4),
                    "test.vm.add",
                    T.anylist_getitem(r, T.int32(1)),
                    T.anylist_getitem(r, T.int32(1)),
                )
                T.anylist_setitem_call_packed(
                    r, T.int32(3), "vm.builtin.copy", T.anylist_getitem(r, T.int32(4))
                )
            else:
                T.anylist_setitem_call_packed(
                    r,
                    T.int32(5),
                    "test.vm.mul",
                    T.anylist_getitem(r, T.int32(1)),
                    T.anylist_getitem(r, T.int32(1)),
                )
                T.anylist_setitem_call_packed(
                    r, T.int32(3), "vm.builtin.copy", T.anylist_getitem(r, T.int32(5))
                )
            T.anylist_setitem_call_packed(
                r, T.int32(2), "vm.builtin.copy", T.anylist_getitem(r, T.int32(3))
            )

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_const():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            R.func_attr({"global_symbol": "main"})
            y = R.const([1, 2])
            z = (y, R.const([3, 4]), x)
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__main(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            # function attr dict
            T.func_attr({"global_symbol": "__vmtir__main"})
            # body
            T.anylist_setitem_call_packed(
                r,
                T.int32(2),
                "vm.builtin.make_tuple",
                T.anylist_getitem(c, T.int32(0)),
                T.anylist_getitem(c, T.int32(1)),
                T.anylist_getitem(r, T.int32(0)),
            )
            T.anylist_setitem_call_packed(
                r, T.int32(1), "vm.builtin.copy", T.anylist_getitem(r, T.int32(2))
            )

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_const_call():
    @tvm.script.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor):
            R.func_attr({"global_symbol": "main"})
            y = R.const([1, 2])
            z = R.call_packed("test.vm.add", x, y, sinfo_args=(R.Tensor))
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__main(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            # function attr dict
            T.func_attr({"global_symbol": "__vmtir__main"})
            # body
            T.anylist_setitem_call_packed(
                r,
                2,
                "test.vm.add",
                T.anylist_getitem(r, 0),
                T.anylist_getitem(c, 0),
            )
            T.anylist_setitem_call_packed(r, 1, "vm.builtin.copy", T.anylist_getitem(r, 2))

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm.testing.main()
