import tvm
from tvm import relay
from tvm.relay.ir_pass import check_kind

# tuples contain only type kinds
def test_tuple_kinds():
    pass

def test_func_kind():
    tp1 = relay.TypeParam('tp1', relay.Kind.Type)
    tp2 = relay.TypeParam('tp2', relay.Kind.Type)

    shape = tvm.convert([1, 2, 3])
    dtype = 'float32'
    tensor_type = relay.TensorType(shape, dtype)

    type_params = tvm.convert([tp1, tp2])
    type_constraints = tvm.convert([])
    arg_types = tvm.convert([tp1, tensor_type])
    ret_type = tp2

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert check_kind(tf)

# type relations only have type kinds of args
def test_relation_kinds():
    pass

# negative tests for both cases
def test_invalid_tuple_kinds():
    pass

def test_invalid_func_kind():
    tp1 = relay.TypeParam('tp1', relay.Kind.Shape)
    tp2 = relay.TypeParam('tp2', relay.Kind.BaseType)
    tp3 = relay.TypeParam('tp3', relay.Kind.ShapeVar)

    type_params = tvm.convert([tp1, tp2, tp3])
    type_constraints = tvm.convert([])
    arg_types = tvm.convert([tp1, tp2])
    ret_type = tp3

    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert not check_kind(tf)

def test_invalid_relation_kinds():
    pass
