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
from tvm import relay
from tvm.relay.analysis import check_kind
from nose.tools import raises


def test_typevar_kind():
    # returns the same kind
    tp1 = relay.TypeVar('tp1', relay.Kind.Type)
    tp2 = relay.TypeVar('tp2', relay.Kind.Shape)
    tp3 = relay.TypeVar('tp3', relay.Kind.Constraint)

    assert check_kind(tp1) == relay.Kind.Type
    assert check_kind(tp2) == relay.Kind.Shape
    assert check_kind(tp3) == relay.Kind.Constraint


def test_tuple_kind():
    # only contain type kinds
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    tf = relay.FuncType(tvm.convert([]), tt, tvm.convert([]), tvm.convert([]))
    fields = tvm.convert([tp, tf, tt])

    tup_ty = relay.TupleType(fields)
    assert check_kind(tup_ty) == relay.Kind.Type


def test_func_kind():
    # only contain type kinds
    tp1 = relay.TypeVar('tp1', relay.Kind.Type)
    tp2 = relay.TypeVar('tp2', relay.Kind.Type)

    shape = tvm.convert([1, 2, 3])
    dtype = 'float32'
    tensor_type = relay.TensorType(shape, dtype)

    tr = relay.TypeRelation(None, tvm.convert([tensor_type, tp1]) , 1, None)

    type_params = tvm.convert([tp1, tp2])
    type_constraints = tvm.convert([tr])
    arg_types = tvm.convert([tp1, tensor_type])
    ret_type = relay.TupleType(tvm.convert([tp2, tensor_type]))

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert check_kind(tf) == relay.Kind.Type


def test_ref_kind():
    # only contain type kinds
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    ft = relay.FuncType(tvm.convert([]), tt, tvm.convert([]), tvm.convert([]))

    rt1 = relay.RefType(tt)
    assert check_kind(rt1) == relay.Kind.Type
    rt2 = relay.RefType(ft)
    assert check_kind(rt2) == relay.Kind.Type
    rt3 = relay.RefType(relay.TupleType([rt1, rt2]))
    assert check_kind(rt3) == relay.Kind.Type


def test_relation_kind():
    # only have type kinds for arguments
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    tf = relay.FuncType(tvm.convert([]), tt, tvm.convert([]), tvm.convert([]))
    args = tvm.convert([tf, tt, tp])

    tr = relay.TypeRelation(None, args, 2, None)
    assert check_kind(tr) == relay.Kind.Constraint


def test_global_typevar_kind():
    v1 = relay.GlobalTypeVar('gtv1', relay.Kind.AdtHandle)
    v2 = relay.GlobalTypeVar('gtv2', relay.Kind.Type)

    assert check_kind(v1) == relay.Kind.AdtHandle
    assert check_kind(v2) == relay.Kind.Type


def test_typecall_kind():
    gtv = relay.GlobalTypeVar('gtv')

    mod = relay.Module()
    data = relay.TypeData(gtv, [], [])
    mod[gtv] = data
    empty_call = relay.TypeCall(gtv, [])
    assert check_kind(empty_call, mod) == relay.Kind.Type

    new_mod = relay.Module()
    tv = relay.TypeVar('tv')
    new_data = relay.TypeData(gtv, [tv], [])
    new_mod[gtv] = new_data
    call = relay.TypeCall(gtv, [relay.TupleType([])])
    assert check_kind(call, new_mod) == relay.Kind.Type


@raises(tvm._ffi.base.TVMError)
def test_invalid_tuple_kind():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tp2 = relay.TypeVar('tp2', relay.Kind.BaseType)
    tp3 = relay.TypeVar('tp3', relay.Kind.ShapeVar)
    fields = tvm.convert([tp1, tp2, tp3])

    tup_ty = relay.TupleType(fields)
    check_kind(tup_ty)


@raises(tvm._ffi.base.TVMError)
def test_invalid_func_kind():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tp2 = relay.TypeVar('tp2', relay.Kind.BaseType)
    tp3 = relay.TypeVar('tp3', relay.Kind.ShapeVar)

    type_params = tvm.convert([tp1, tp2, tp3])
    type_constraints = tvm.convert([])
    arg_types = tvm.convert([tp1, tp2])
    ret_type = tp3

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    check_kind(tf)


@raises(tvm._ffi.base.TVMError)
def test_invalid_ref_kind():
    tp = relay.TypeVar('tp', relay.Kind.Shape)
    rt = relay.RefType(tp)
    check_kind(rt)


@raises(tvm._ffi.base.TVMError)
def test_invalid_relation_kind():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tp2 = relay.TypeVar('tp2', relay.Kind.BaseType)
    tp3 = relay.TypeVar('tp3', relay.Kind.ShapeVar)
    args = tvm.convert([tp1, tp2, tp3])

    func = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    tr = relay.TypeRelation(func, args, 2, None)
    check_kind(tr)


@raises(tvm._ffi.base.TVMError)
def test_typecall_invalid_callee():
    # global type var must be an ADT handle
    gtv = relay.GlobalTypeVar('v1', relay.Kind.Type)
    check_kind(relay.TypeCall(gtv, []))


@raises(tvm._ffi.base.TVMError)
def test_typecall_invalid_args():
    # args must all be type kind
    mod = relay.Module()
    gtv = relay.GlobalTypeVar('v1')
    data = relay.TypeData(gtv, [], [])
    mod[gtv] = data

    check_kind(relay.TypeCall(gtv, [data]))


@raises(tvm._ffi.base.TVMError)
def test_typecall_invalid_num_args():
    mod = relay.Module()
    gtv = relay.GlobalTypeVar('v1')
    tv = relay.TypeVar('tv')
    data = relay.TypeData(gtv, [tv], [])
    mod[gtv] = data
    check_kind(relay.TypeCall(gtv, []))


@raises(tvm._ffi.base.TVMError)
def test_func_with_invalid_ret_type():
    tp1 = relay.TypeVar('tp1', relay.Kind.Type)
    tp2 = relay.TypeVar('tp2', relay.Kind.Shape)
    tf = relay.FuncType(tvm.convert([tp1]), tp2, tvm.convert([tp1, tp2]), tvm.convert([]))

    check_kind(tf)


@raises(tvm._ffi.base.TVMError)
def test_func_with_invalid_arg_types():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tp2 = relay.TypeVar('tp2', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([tp1]), tp2, tvm.convert([tp1, tp2]), tvm.convert([]))

    check_kind(tf)


@raises(tvm._ffi.base.TVMError)
def test_func_with_invalid_tuple():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)

    ret_type = relay.TupleType(tvm.convert([tp1, tp1, tp1]))

    tf = relay.FuncType(tvm.convert([]), ret_type, tvm.convert([tp1]), tvm.convert([]))
    check_kind(tf)


@raises(tvm._ffi.base.TVMError)
def test_func_with_invalid_relation():
    tp1 = relay.TypeVar('tp1', relay.Kind.Type)
    tp2 = relay.TypeVar('tp2', relay.Kind.Shape)
    tp3 = relay.TypeVar('tp3', relay.Kind.ShapeVar)

    func = tvm.get_env_func("tvm.relay.type_relation.Identity")
    tr = relay.TypeRelation(func, tvm.convert([tp2, tp3]), 1, None)

    tf = relay.FuncType(tvm.convert([tp1]), tp1, tvm.convert([tp1, tp2, tp3]), tvm.convert([tr]))
    check_kind(tf)


@raises(tvm._ffi.base.TVMError)
def test_tuple_with_invalid_func():
    tensor_type = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')

    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tf = relay.FuncType(tvm.convert([]), tp1, tvm.convert([tp1]), tvm.convert([]))

    tup_ty = relay.TupleType(tvm.convert([tensor_type, tf]))
    check_kind(tup_ty)


if __name__ == "__main__":
    test_tuple_kind()
    test_func_kind()
    test_ref_kind()
    test_relation_kind()
    test_global_typevar_kind()
    test_typecall_kind()
    test_invalid_tuple_kind()
    test_invalid_func_kind()
    test_invalid_ref_kind()
    test_invalid_relation_kind()
    test_typecall_invalid_callee()
    test_typecall_invalid_args()
    test_typecall_invalid_num_args()
    test_func_with_invalid_ret_type()
    test_func_with_invalid_arg_types()
    test_func_with_invalid_tuple()
    test_func_with_invalid_relation()
    test_tuple_with_invalid_func()
