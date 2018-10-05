import tvm
from tvm import relay

def test_tensor_type_alpha_eq():
    t1 = relay.TensorType((3, 4), "float32")
    t2 = relay.TensorType((3, 4), "float32")
    t3 = relay.TensorType((3, 4, 5), "float32")
    assert t1 == t2
    assert t1 != t3

    t1 = relay.TensorType((), "float32")
    t2 = relay.TensorType((), "float32")
    assert t1 == t2


def test_incomplete_type_alpha_eq():
    t1 = relay.IncompleteType(relay.Kind.Shape)
    t2 = relay.IncompleteType(relay.Kind.Type)
    t3 = relay.IncompleteType(relay.Kind.Type)

    # only equal when there is pointer equality
    assert t2 == t2
    assert t1 == t1
    assert t1 != t2
    assert t2 != t3


def test_type_param_alpha_eq():
    t1 = relay.TypeParam("v1", relay.Kind.Type)
    t2 = relay.TypeParam("v2", relay.Kind.Shape)
    t3 = relay.TypeParam("v3", relay.Kind.Type)

    # only pointer equality and eq_map allow equal params
    assert t1 == t1
    assert t2 == t2
    assert t1 != t2 # different kind
    assert t1 != t3 # not in eq_map

    # function types are the only way to put type params
    # in eq map
    ft1 = relay.FuncType(tvm.convert([]), t1, tvm.convert([t1]), tvm.convert([]))
    ft2 = relay.FuncType(tvm.convert([]), t3, tvm.convert([t3]), tvm.convert([]))
    # actually an invalid type because t2 is wrong kind
    ft3 = relay.FuncType(tvm.convert([]), t2, tvm.convert([t2]), tvm.convert([]))

    assert ft1 == ft2
    assert ft1 != ft3 # kinds still do not match


def test_func_type_alpha_eq():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")

    tp1 = relay.TypeParam("v1", relay.Kind.Type)
    tp2 = relay.TypeParam("v2", relay.Kind.Type)
    tp3 = relay.TypeParam("v3", relay.Kind.Shape)
    tp4 = relay.TypeParam("v3", relay.Kind.Shape)

    tr1 = relay.TypeRelation(None, tvm.convert([tp1, tp3]), 1, None)
    tr2 = relay.TypeRelation(None, tvm.convert([tp2, tp4]), 1, None)

    ft1 = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    ft2 = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp2, tp4]),
                         tvm.convert([tr2]))
    assert ft1 == ft2

    ft3 = relay.FuncType(tvm.convert([t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft1 != ft3

    ft4 = relay.FuncType(tvm.convert([t2, t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft1 != ft4

    ft5 = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([]))
    assert ft1 != ft5

    ft6 = relay.FuncType(tvm.convert([t1, t2]), tp2,
                         tvm.convert([tp1, tp2, tp3]),
                         tvm.convert([tr1]))
    assert ft1 != ft6

    ft7 = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp2, tp3, tp4]),
                         tvm.convert([tr1, tr2]))
    assert ft1 != ft7


def test_tuple_type_alpha_eq():
    t1 = relay.TensorType((1, 2, 3), "float32")
    t2 = relay.TensorType((1, 2, 3, 4), "float32")
    tp1 = relay.TypeParam("v1", relay.Kind.Type)
    tp2 = relay.TypeParam("v2", relay.Kind.Type)

    tup1 = relay.TupleType(tvm.convert([t1, t2, tp1]))
    tup2 = relay.TupleType(tvm.convert([t1, t2, tp1]))
    tup3 = relay.TupleType(tvm.convert([t2, t1, tp1]))
    tup4 = relay.TupleType(tvm.convert([t1, t2, tp2]))

    # as long as types are alpha-equal and in same order,
    # tuples should be alpha-equal
    assert tup1 == tup2
    assert tup1 != tup3
    assert tup1 != tup4


def test_type_relation_alpha_eq():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")
    t3 = relay.TensorType((1, 2, 3, 4), "float32")

    tr1 = relay.TypeRelation(None, tvm.convert([t1, t2]), 1, None)
    tr2 = relay.TypeRelation(None, tvm.convert([t1, t2]), 1, None)
    tr3 = relay.TypeRelation(None, tvm.convert([t2, t1]), 1, None)
    tr4 = relay.TypeRelation(None, tvm.convert([t2, t3]), 1, None)
    tr5 = relay.TypeRelation(None, tvm.convert([t1, t3, t2]), 2, None)
    tr6 = relay.TypeRelation(None, tvm.convert([t1, t3, t2]), 1, None)

    # number of args, input count, and order should be the same
    assert tr1 == tr2
    assert tr1 != tr3
    assert tr1 != tr4
    assert tr1 != tr5
    assert tr1 != tr6
    assert tr5 != tr6


if __name__ == "__main__":
    test_tensor_type_alpha_eq()
    test_incomplete_type_alpha_eq()
    test_type_param_alpha_eq()
    test_func_type_alpha_eq()
    test_tuple_type_alpha_eq()
    test_type_relation_alpha_eq()
