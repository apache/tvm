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
from tvm import relay
from tvm.relay import TypeFunctor, TypeMutator, TypeVisitor
from tvm.relay.ty import (
    TypeVar,
    IncompleteType,
    TensorType,
    FuncType,
    TupleType,
    TypeRelation,
    RefType,
    GlobalTypeVar,
    TypeCall,
)
from tvm.relay.adt import TypeData


def check_visit(typ):
    try:
        ef = TypeFunctor()
        ef.visit(typ)
        assert False
    except NotImplementedError:
        pass

    ev = TypeVisitor()
    ev.visit(typ)

    tvm.ir.assert_structural_equal(TypeMutator().visit(typ), typ, map_free_vars=True)


def test_type_var():
    tv = TypeVar("a")
    check_visit(tv)


def test_incomplete_type():
    it = IncompleteType()
    check_visit(it)


def test_tensor_type():
    tt = TensorType([])
    check_visit(tt)


def test_func_type():
    tv = TypeVar("tv")
    tt = relay.TensorType(tvm.runtime.convert([1, 2, 3]), "float32")
    ft = FuncType([tt], tt, type_params=[tv])
    check_visit(ft)


def test_tuple_type():
    tt = TupleType([TupleType([])])
    check_visit(tt)


def test_type_relation():
    func = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Broadcast")
    attrs = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3, 4))
    tp = TypeVar("tp")
    tf = FuncType([], TupleType([]), [], [])
    tt = TensorType([1, 2, 3], "float32")
    tr = TypeRelation(func, [tp, tf, tt], 2, attrs)

    check_visit(tr)


def test_ref_type():
    rt = RefType(TupleType([]))
    check_visit(rt)


def test_global_type_var():
    gtv = GlobalTypeVar("gtv")
    check_visit(gtv)


def test_type_call():
    tc = TypeCall(GlobalTypeVar("tf"), [TupleType([])])
    check_visit(tc)


def test_type_data():
    td = TypeData(GlobalTypeVar("td"), [TypeVar("tv")], [])
    check_visit(td)


if __name__ == "__main__":
    test_type_var()
    test_incomplete_type()
    test_tensor_type()
    test_func_type()
    test_tuple_type()
    test_type_relation()
    test_ref_type()
    test_global_type_var()
    test_type_call()
    test_type_data()
