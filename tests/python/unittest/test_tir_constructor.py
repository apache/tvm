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

def test_expr_constructor():
    x = tvm.tir.Var("xx", "float32")
    assert isinstance(x, tvm.tir.Var)
    assert x.name == "xx"

    x = tvm.tir.Reduce(None, [1],
                       [tvm.tir.IterVar((0, 1), "x", 2)],
                       None, 0)
    assert isinstance(x, tvm.tir.Reduce)
    assert x.combiner == None
    assert x.value_index == 0

    x = tvm.tir.FloatImm("float32", 1.0)
    assert isinstance(x, tvm.tir.FloatImm)
    assert x.value == 1.0
    assert x.dtype == "float32"

    x = tvm.tir.IntImm("int64", 2)
    assert isinstance(x, tvm.tir.IntImm)
    assert x.value == 2
    assert x.dtype == "int64"

    x = tvm.tir.StringImm("xyza")
    assert isinstance(x, tvm.tir.StringImm)
    assert x.value == "xyza"

    x = tvm.tir.Cast("float32", tvm.tir.IntImm("uint32", 1))
    assert isinstance(x, tvm.tir.Cast)
    assert x.dtype == "float32"
    assert x.value.value == 1

    a = tvm.tir.const(1.0, dtype="float32")
    b = te.var("x", dtype="float32")

    for cls in [tvm.tir.Add,
                tvm.tir.Sub,
                tvm.tir.Mul,
                tvm.tir.Div,
                tvm.tir.Mod,
                tvm.tir.Min,
                tvm.tir.Max,
                tvm.tir.LT,
                tvm.tir.LE,
                tvm.tir.GT,
                tvm.tir.GE]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)


    a = tvm.runtime.convert(te.var("x") > 1)
    b = tvm.runtime.convert(te.var("x") == 1)

    for cls in [tvm.tir.And,
                tvm.tir.Or]:
        x = cls(a, b)
        assert isinstance(x, cls)
        assert x.a == a
        assert x.b.same_as(b)

    x = tvm.tir.Not(a)
    assert isinstance(x, tvm.tir.Not)
    assert x.a == a

    x = tvm.tir.Select(a, a, b)
    assert isinstance(x, tvm.tir.Select)
    assert x.true_value == a
    assert x.false_value == b
    assert x.condition == a

    buffer_var = te.var("x", dtype="handle")
    x = tvm.tir.Load("float32", buffer_var, 1, a)
    assert isinstance(x, tvm.tir.Load)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.index.value == 1
    assert x.predicate == a

    x = tvm.tir.Ramp(1, 2, 10)
    assert isinstance(x, tvm.tir.Ramp)
    assert x.base.value == 1
    assert x.stride.value == 2
    assert x.lanes == 10

    x = tvm.tir.Broadcast(a, 10)
    assert isinstance(x, tvm.tir.Broadcast)
    assert x.value == a
    assert x.lanes == 10

    x = tvm.tir.Shuffle([a], [0])
    assert isinstance(x, tvm.tir.Shuffle)
    assert x.vectors[0] == a
    assert x.indices[0].value == 0

    x = tvm.tir.Call("float32", "xyz", [a], tvm.tir.Call.Extern, None, 0)
    assert isinstance(x, tvm.tir.Call)
    assert x.dtype == "float32"
    assert x.name == "xyz"
    assert x.args[0] == a
    assert x.call_type == tvm.tir.Call.Extern
    assert x.func == None
    assert x.value_index == 0

    v = te.var("aa")
    x = tvm.tir.Let(v, 1, v)
    assert x.var == v
    assert x.value.value == 1
    assert x.body == v


def test_stmt_constructor():
    v = te.var("aa")
    buffer_var = te.var("buf", dtype="handle")
    nop = tvm.tir.Evaluate(1)
    x = tvm.tir.LetStmt(v, 1, tvm.tir.Evaluate(1))
    assert isinstance(x, tvm.tir.LetStmt)
    assert x.var == v
    assert x.value.value == 1
    assert isinstance(x.body, tvm.tir.Evaluate)

    x = tvm.tir.AttrStmt(v == 1, "xx", 1, tvm.tir.Evaluate(1))
    assert isinstance(x, tvm.tir.AttrStmt)
    assert x.value.value == 1

    x = tvm.tir.AssertStmt(tvm.tir.const(1, "uint1"),
                            tvm.runtime.convert("hellow"),
                            nop)
    assert isinstance(x, tvm.tir.AssertStmt)
    assert x.body == nop

    x = tvm.tir.For(te.var("x"), 0, 10, 0, 0, nop)
    assert isinstance(x, tvm.tir.For)
    assert x.min.value == 0
    assert x.extent.value == 10
    assert x.body == nop

    x = tvm.tir.Store(buffer_var, 1, 10, tvm.tir.const(1, "uint1"))
    assert isinstance(x, tvm.tir.Store)
    assert x.buffer_var == buffer_var
    assert x.index.value == 10
    assert x.value.value == 1

    tensor = te.placeholder((), dtype="float32")
    x = tvm.tir.Provide(tensor.op, 0, 10, [])
    assert isinstance(x, tvm.tir.Provide)
    assert x.value_index == 0
    assert x.value.value == 10

    x = tvm.tir.Allocate(buffer_var, "float32", [10],
                          tvm.tir.const(1, "uint1"), nop)
    assert isinstance(x, tvm.tir.Allocate)
    assert x.dtype == "float32"
    assert x.buffer_var == buffer_var
    assert x.body == nop

    x = tvm.tir.AttrStmt(buffer_var, "xyz", 1, nop)
    assert isinstance(x, tvm.tir.AttrStmt)
    assert x.node == buffer_var
    assert x.attr_key == "xyz"
    assert x.body == nop

    x = tvm.tir.Free(buffer_var)
    assert isinstance(x, tvm.tir.Free)
    assert x.buffer_var == buffer_var

    x = tvm.tir.Realize(None, 0, "float", [], tvm.tir.const(1, "uint1"), nop)
    assert isinstance(x, tvm.tir.Realize)
    assert x.body == nop

    x = tvm.tir.IfThenElse(tvm.tir.const(1, "uint1"),
                            tvm.tir.Evaluate(11),
                            nop)
    assert isinstance(x, tvm.tir.IfThenElse)
    assert x.then_case.value.value == 11
    assert x.else_case == nop

    b = tvm.tir.decl_buffer((1, 2))
    x = tvm.tir.Prefetch(b, [])
    assert isinstance(x, tvm.tir.Prefetch)


if __name__ == "__main__":
    test_expr_constructor()
    test_stmt_constructor()
