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
# pylint: disable=invalid-name, missing-docstring, too-many-statements
import tvm
from tvm import relay


def get_recursive_count_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    sb = relay.ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype="int32"))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype="int32"))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    mod[sum_up] = func
    iarg = relay.var("i", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    return mod, sum_up


def test_call_chain_inline_leaf():
    """Test when only leaf call is inlined.

    The call graph is like the following:
              main
              /  \
             g1   g2
             /
            g11(inline)
    """

    def get_mod():
        mod = tvm.IRModule({})
        x11 = relay.var("x11", shape=(3, 5))
        g11 = relay.GlobalVar("g11")
        fn11 = relay.Function([x11], x11)
        fn11 = fn11.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        mod[g11] = fn11

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1 + g11(x1))
        fn1 = relay.Function([x1, y1], sb.get())
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1 + x1)
        fn1 = relay.Function([x1, y1], sb.get())
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_call_chain_inline_multiple_levels():
    """Test when only leaf call is inlined.

    The call graph is like the following:
                  main
                 /    \
          g1(inline)   g2
               /
        g11(inline)

    """

    def get_mod():
        mod = tvm.IRModule({})
        x11 = relay.var("x11", shape=(3, 5))
        g11 = relay.GlobalVar("g11")
        fn11 = relay.Function([x11], x11)
        fn11 = fn11.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        mod[g11] = fn11

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1 + g11(x1))
        fn1 = relay.Function([x1, y1], sb.get())
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        mod = tvm.IRModule({})

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = p0 + p1 + p0
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_call_chain_inline_multiple_levels_extern_compiler():
    """Test when only leaf call is inlined.

    The call graph is like the following:
                  main
                 /    \
          g1(inline)   g2
               /
        g11(inline, external compiler)

    """

    def get_mod():
        mod = tvm.IRModule({})
        x11 = relay.var("x11", shape=(3, 5))
        g11 = relay.GlobalVar("g11")
        fn11 = relay.Function([x11], x11)
        fn11 = fn11.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn11 = fn11.with_attr("Compiler", "a")
        mod[g11] = fn11

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1 + g11(x1))
        fn1 = relay.Function([x1, y1], sb.get())
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        mod = tvm.IRModule({})
        x11 = relay.var("x11", shape=(3, 5))
        fn11 = relay.Function([x11], x11)
        fn11 = fn11.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn11 = fn11.with_attr("Compiler", "a")

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = p0 + p1 + fn11(p0)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_recursive_call_with_global():
    def get_mod():
        mod = tvm.IRModule({})

        x = relay.var("x", shape=[], dtype="int32")
        fn0 = relay.Function([x], x)
        fn0 = fn0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        gx = relay.GlobalVar("gx")
        mod[gx] = fn0

        sum_up = relay.GlobalVar("sum_up")
        i = relay.var("i", shape=[], dtype="int32")
        sb = relay.ScopeBuilder()
        with sb.if_scope(relay.equal(i, relay.const(0, dtype="int32"))):
            sb.ret(i)
        with sb.else_scope():
            one_less = relay.subtract(i, relay.const(1, dtype="int32"))
            global_call = gx(i)
            rec_call = relay.Call(sum_up, [one_less]) + global_call
            sb.ret(relay.add(rec_call, i))
        func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
        func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        mod[sum_up] = func
        iarg = relay.var("i", shape=[], dtype="int32")
        mod["main"] = relay.Function([iarg], sum_up(iarg))
        return mod

    def expected():
        mod = tvm.IRModule({})

        sum_up = relay.GlobalVar("sum_up")
        i = relay.var("i", shape=[], dtype="int32")
        sb = relay.ScopeBuilder()
        with sb.if_scope(relay.equal(i, relay.const(0, dtype="int32"))):
            sb.ret(i)
        with sb.else_scope():
            one_less = relay.subtract(i, relay.const(1, dtype="int32"))
            rec_call = relay.Call(sum_up, [one_less]) + i
            sb.ret(relay.add(rec_call, i))
        func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
        func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        mod[sum_up] = func
        iarg = relay.var("i", shape=[], dtype="int32")
        mod["main"] = relay.Function([iarg], sum_up(iarg))
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_recursive_called():
    mod, sum_up = get_recursive_count_loop()
    iarg = relay.var("i", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    ref_mod = mod
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, ref_mod, map_free_vars=True)


def test_recursive_not_called():
    def get_mod():
        mod, sum_up = get_recursive_count_loop()
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(2, 2))
        x1 = relay.var("x1", shape=(2, 2))
        fn1 = relay.Function([x1], x1)
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1
        mod["main"] = relay.Function([x, y], x + y + g1(x))
        return mod

    def expected():
        mod, sum_up = get_recursive_count_loop()
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(2, 2))
        mod["main"] = relay.Function([x, y], x + y + x)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    ref_mod = expected()
    assert tvm.ir.structural_equal(mod, ref_mod, map_free_vars=True)


def test_recursive_not_called_extern_compiler():
    def get_mod():
        mod, sum_up = get_recursive_count_loop()
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(2, 2))
        x1 = relay.var("x1", shape=(2, 2))
        fn1 = relay.Function([x1], x1)
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1
        mod["main"] = relay.Function([x, y], x + y + g1(x))
        return mod

    def expected():
        mod, sum_up = get_recursive_count_loop()
        x = relay.var("x", shape=(2, 2))
        y = relay.var("y", shape=(2, 2))
        x1 = relay.var("x1", shape=(2, 2))
        fn1 = relay.Function([x1], x1)
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")
        mod["main"] = relay.Function([x, y], x + y + fn1(x))
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    ref_mod = expected()
    assert tvm.ir.structural_equal(mod, ref_mod, map_free_vars=True)


def test_globalvar_as_call_arg():
    def get_mod():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1)
        fn1 = relay.Function([x1, y1], sb.get())
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = p0 + p1
        call_fn2 = p2 - p3
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_globalvar_as_call_arg_extern_compiler():
    def get_mod():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1)
        fn1 = relay.Function([x1, y1], sb.get())
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = fn2.with_attr("Compiler", "b")
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1)
        fn1 = relay.Function([x1, y1], sb.get())
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = fn2.with_attr("Compiler", "b")

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = relay.Call(fn1, [p0, p1])
        call_fn2 = relay.Call(fn2, [p2, p3])
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_inline_globalvar_without_args():
    def get_mod():
        mod = tvm.IRModule({})
        fn1 = relay.Function([], relay.const(1))
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = relay.Function([], relay.const(2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        g2 = relay.GlobalVar("g2")
        mod[g1] = fn1
        mod = relay.transform.InferType()(mod)
        mod[g2] = fn2
        p = relay.var("p", "bool")
        mod["main"] = relay.Function([p], relay.Call(relay.If(p, g1, g2), []))
        return relay.transform.InferType()(mod)

    def expected():
        mod = tvm.IRModule({})
        fn1 = relay.Function([], relay.const(1))
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = relay.Function([], relay.const(2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        p = relay.var("p", "bool")
        mod["main"] = relay.Function([p], relay.Call(relay.If(p, fn1, fn2), []))
        return relay.transform.InferType()(mod)

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_inline_globalvar_without_args_extern_compiler():
    def get_mod():
        mod = tvm.IRModule({})
        fn1 = relay.Function([], relay.const(1))
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")
        fn2 = relay.Function([], relay.const(2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = fn2.with_attr("Compiler", "b")
        g1 = relay.GlobalVar("g1")
        g2 = relay.GlobalVar("g2")
        mod[g1] = fn1
        mod[g2] = fn2
        p = relay.var("p", "bool")
        mod["main"] = relay.Function([p], relay.Call(relay.If(p, g1, g2), []))
        return mod

    def expected():
        mod = tvm.IRModule({})
        fn1 = relay.Function([], relay.const(1))
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn1 = fn1.with_attr("Compiler", "a")
        fn2 = relay.Function([], relay.const(2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn2 = fn2.with_attr("Compiler", "b")
        p = relay.var("p", "bool")
        mod["main"] = relay.Function([p], relay.Call(relay.If(p, fn1, fn2), []))
        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_globalvar_called_by_multiple_functions():
    """Test when only leaf call is inlined.

    The call graph is like the following:
                  main    g0
                 /    \   /
                g1    g2(inline)
    """

    def get_mod():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1)
        fn1 = relay.Function([x1, y1], sb.get())
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        sb1 = relay.ScopeBuilder()
        sb1.ret(x2 - y2)
        fn2 = relay.Function([x2, y2], sb1.get())
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        z0 = relay.var("z0", shape=(3, 5))
        fn0 = relay.Function([x0, y0, z0], g2(x0, y0) + z0)
        g0 = relay.GlobalVar("g0")
        mod[g0] = fn0

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn1 = g1(p0, p1)
        call_fn2 = g2(p2, p3)
        mod["main"] = relay.Function([p0, p1, p2, p3], call_fn1 * call_fn2)
        return mod

    def expected():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        sb = relay.ScopeBuilder()
        sb.ret(x1 + y1)
        fn1 = relay.Function([x1, y1], sb.get())
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        p0 = relay.var("p0", shape=(3, 5))
        p1 = relay.var("p1", shape=(3, 5))
        p2 = relay.var("p2", shape=(3, 5))
        p3 = relay.var("p3", shape=(3, 5))

        call_fn2 = p2 - p3
        mod["main"] = relay.Function([p0, p1, p2, p3], g1(p0, p1) * call_fn2)

        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        z0 = relay.var("z0", shape=(3, 5))

        fn0 = relay.Function([x0, y0, z0], x0 - y0 + z0)
        g0 = relay.GlobalVar("g0")
        mod[g0] = fn0

        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_entry_with_inline():
    """Test entry function with inline

    The call graph is like the following:
                g1(inline)    g2(inline)
    """

    def get_mod():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + y1)
        fn1 = fn1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - y2)
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, get_mod(), map_free_vars=True)


def test_callee_not_inline():
    """Test entry function with inline

    The call graph is like the following:
                    main
                      |
                 g2(inline)
                      |
                     g1
    """

    def get_mod():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + y1)
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, get_mod(), map_free_vars=True)


def test_callee_not_inline_leaf_inline():
    """Test entry function with inline

    The call graph is like the following:
                    main
                      |
                 g2(inline)
                      |
                     g1
                      |
                 g0(inline)
    """

    def get_mod():
        mod = tvm.IRModule({})
        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        fn0 = relay.Function([x0, y0], x0 * y0)
        fn0 = fn0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g0 = relay.GlobalVar("g0")
        mod[g0] = fn0

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + g0(x1, y1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2
        return mod

    def expected():
        mod = tvm.IRModule({})
        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + x1 * y1)
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


def test_callee_not_inline_leaf_inline_extern_compiler():
    """Test entry function with inline

    The call graph is like the following:
                    main
                      |
                 g2(inline)
                      |
                     g1
                      |
                 g0(inline, external compiler)
    """

    def get_mod():
        mod = tvm.IRModule({})
        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        fn0 = relay.Function([x0, y0], x0 * y0)
        fn0 = fn0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn0 = fn0.with_attr("Compiler", "aa")
        g0 = relay.GlobalVar("g0")
        mod[g0] = fn0

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + g0(x1, y1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2
        return mod

    def expected():
        mod = tvm.IRModule({})
        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        fn0 = relay.Function([x0, y0], x0 * y0)
        fn0 = fn0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn0 = fn0.with_attr("Compiler", "aa")

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + fn0(x1, y1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2

        return mod

    mod = get_mod()
    mod = relay.transform.Inline()(mod)
    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
