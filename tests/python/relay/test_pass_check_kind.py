import tvm
from tvm import relay
from tvm.relay.ir_pass import check_kind
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


def test_relation_kind():
    # only have type kinds for arguments
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    tf = relay.FuncType(tvm.convert([]), tt, tvm.convert([]), tvm.convert([]))
    args = tvm.convert([tf, tt, tp])

    tr = relay.TypeRelation(None, args, 2, None)
    assert check_kind(tr) == relay.Kind.Constraint


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
def test_invalid_relation_kind():
    tp1 = relay.TypeVar('tp1', relay.Kind.Shape)
    tp2 = relay.TypeVar('tp2', relay.Kind.BaseType)
    tp3 = relay.TypeVar('tp3', relay.Kind.ShapeVar)
    args = tvm.convert([tp1, tp2, tp3])

    func = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    tr = relay.TypeRelation(func, args, 2, None)
    check_kind(tr)


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
    test_relation_kind()
    test_invalid_tuple_kind()
    test_invalid_func_kind()
    test_invalid_relation_kind()
    test_func_with_invalid_ret_type()
    test_func_with_invalid_arg_types()
    test_func_with_invalid_tuple()
    test_func_with_invalid_relation()
    test_tuple_with_invalid_func()
