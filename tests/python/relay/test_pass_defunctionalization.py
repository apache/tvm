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
import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay import transform, ExprVisitor, TypeVisitor
from tvm.relay.testing import Prelude

# determine if type t is a FuncType or has a nested FuncType
def has_func_type(t):
    class FuncTypeVisitor(TypeVisitor):
        def __init__(self):
            super().__init__()
            self.has_func = False

        def visit_func_type(self, ftt):
            self.has_func = True

    ftvisitor = FuncTypeVisitor()
    ftvisitor.visit(t)
    return ftvisitor.has_func


# determine whether a program has any higher order functions
# a higher order function is defined as one that:
# - has function type arguments
# - returns a function
def assert_no_higher_order_functions(expr, mod):
    class CheckFirstOrderVisitor(ExprVisitor):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod
            self.hof = []
            self.visited_gv = set()

        def visit_call(self, call):
            is_higher_order = False
            # check return type
            if has_func_type(call.checked_type):
                is_higher_order = True
            # check argument types
            for a in call.args:
                if has_func_type(a.checked_type):
                    is_higher_order = True
            # if it is higher order, save it for debugging later
            if is_higher_order:
                self.hof.append(call)
            super().visit_call(call)

        def visit_global_var(self, gv):
            # visit global vars to visit entire program
            if gv not in self.visited_gv:
                self.visited_gv.add(gv)
                self.visit(self.mod[gv])

    mod = transform.InferType()(mod)
    check_fo_visitor = CheckFirstOrderVisitor(mod)
    check_fo_visitor.visit(expr)

    nl = "\n--------\n"
    errmsg = f"""found {len(check_fo_visitor.hof)} higher order functions:
  {nl.join(expr.astext() for expr in check_fo_visitor.hof)}"""

    assert len(check_fo_visitor.hof) == 0, errmsg


# assert that a program is defunctionalized and returns
# defunctionalized module
# assumes program starts from mod['main']
def defunctionalized(mod):
    mod = transform.InferType()(mod)
    mod["main"] = transform.Defunctionalization(mod["main"], mod)
    mod = transform.InferType()(mod)
    assert_no_higher_order_functions(mod["main"], mod)

    return mod


# adt list to python list
def to_list(mod, l):
    list = mod.get_global_type_var("List")
    list_adt = mod[list]
    cons = list_adt.constructors[0]
    nil = list_adt.constructors[1]

    assert isinstance(l, ConstructorValue)
    val = l
    ret = []
    while True:
        if val.tag == cons.tag:
            ret.append(val.fields[0].numpy())
            val = val.fields[1]
        else:
            assert val.tag == nil.tag
            break
    return ret


# list to adt list
def to_adt_list(mod, arr):
    expr = mod["main"]
    l = mod.get_global_type_var("List")
    list_adt = mod[l]
    cons = list_adt.constructors[0]
    nil = list_adt.constructors[1]

    li = nil()
    for a in arr:
        li = cons(relay.const(a), li)
    adt = relay.create_executor(mod=mod).evaluate(li)
    mod["main"] = expr
    return adt


def test_simple():
    code = """
#[version = "0.0.5"]
def @simple[A, B](%f: fn(A) -> B, %xs: A) -> B {
  %f(%xs)
}
def @main(%l: Tensor[(5, 5), float32]) -> Tensor[(5, 5), float32] {
  %0 = fn[A](%x: A) -> A {
    %x
  };
  @simple(%0, %l)
}
"""
    mod = tvm.relay.fromtext(code)
    defunc_mod = defunctionalized(mod)

    input = np.random.rand(5, 5).astype("float32")

    out = relay.create_executor("debug", mod=mod).evaluate()(input)

    defunc_out = relay.create_executor("debug", mod=defunc_mod).evaluate()(input)

    np.testing.assert_equal(out.numpy(), defunc_out.numpy())


def test_global_recursion():
    code = """
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}
def @id[A](%x: A) -> A {
  %x
}
def @map[A, B](%f: fn(A) -> B, %xs: List[A]) -> List[B] {
  match (%xs) {
    Cons(%x, %rest) => Cons(%f(%x), @map(%f, %rest)),
    Nil => Nil,
  }
}
def @main(%l: List[float32]) -> List[float32] {
  @map(@id, %l)
}
"""
    mod = tvm.relay.fromtext(code)
    defunc_mod = defunctionalized(mod)

    input = np.random.rand(10).astype("float32")

    out = relay.create_executor("debug", mod=mod).evaluate(mod["main"])(to_adt_list(mod, input))

    defunc_out = relay.create_executor("debug", mod=defunc_mod).evaluate()(
        to_adt_list(defunc_mod, input)
    )

    np.testing.assert_array_equal(to_list(mod, out), to_list(defunc_mod, defunc_out))


def test_recursive_datatype():
    # CPS will create recursive datatype
    code = """
#[version = "0.0.5"]
type List[A] {
  Cons(A, List[A]),
  Nil,
}
def @sum(%f: fn(int32) -> int32, %k: List[int32]) -> int32 {
  match (%k) {
    Cons(%x, %rest) => %0 = fn(%n) {
      %x + %f(%n)
    };
    @sum(%0, %rest),
    Nil => %f(0),
  }
}
def @id[A](%x: A) -> A {
  %x
}
def @main(%l: List[int32]) -> int32 {
  @sum(@id, %l)
}
"""
    mod = tvm.relay.fromtext(code)
    defunc_mod = defunctionalized(mod)

    input = np.random.randint(1, 100, 10)

    out = relay.create_executor("debug", mod=mod).evaluate(mod["main"])(to_adt_list(mod, input))

    defunc_out = relay.create_executor("debug", mod=defunc_mod).evaluate()(
        to_adt_list(defunc_mod, input)
    )

    tvm.testing.assert_allclose(out.numpy(), defunc_out.numpy())


if __name__ == "__main__":
    tvm.testing.main()
