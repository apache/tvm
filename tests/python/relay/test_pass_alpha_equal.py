import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
from tvm.relay.ir_builder import convert

def test_tensor_type_alpha_equal():
    t1 = relay.TensorType((3, 4), "float32")
    t2 = relay.TensorType((3, 4), "float32")
    t3 = relay.TensorType((3, 4, 5), "float32")
    assert t1 == t2
    assert t1 != t3

    t1 = relay.TensorType((), "float32")
    t2 = relay.TensorType((), "float32")
    assert t1 == t2

def test_constant_alpha_equal():
    x = convert(1)
    y = convert(2)
    assert alpha_equal(x, x)
    assert not alpha_equal(x, y)
    assert alpha_equal(x, convert(1))

def test_incomplete_type_alpha_equal():
    t1 = relay.IncompleteType(relay.Kind.Shape)
    t2 = relay.IncompleteType(relay.Kind.Type)
    t3 = relay.IncompleteType(relay.Kind.Type)

    # only equal when there is pointer equality
    assert t2 == t2
    assert t1 == t1
    assert t1 != t2
    assert t2 != t3


def test_type_param_alpha_equal():
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


def test_func_type_alpha_equal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")

    tp1 = relay.TypeParam("v1", relay.Kind.Type)
    tp2 = relay.TypeParam("v2", relay.Kind.Type)
    tp3 = relay.TypeParam("v3", relay.Kind.Shape)
    tp4 = relay.TypeParam("v3", relay.Kind.Shape)

    broadcast = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    identity = tvm.get_env_func("tvm.relay.type_relation.Identity")

    tr1 = relay.TypeRelation(broadcast, tvm.convert([tp1, tp3]), 1, None)
    tr2 = relay.TypeRelation(broadcast, tvm.convert([tp2, tp4]), 1, None)
    tr3 = relay.TypeRelation(identity, tvm.convert([tp1, tp3]), 1, None)

    ft = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    translate_vars = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp2, tp4]),
                         tvm.convert([tr2]))
    assert ft == translate_vars

    different_args = relay.FuncType(tvm.convert([t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft != different_args

    different_order = relay.FuncType(tvm.convert([t2, t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft != different_order

    no_rel = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([]))
    assert ft != no_rel

    more_vars = relay.FuncType(tvm.convert([t1, t2]), tp2,
                         tvm.convert([tp1, tp2, tp3]),
                         tvm.convert([tr1]))
    assert ft != more_vars

    all_the_vars = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp2, tp3, tp4]),
                         tvm.convert([tr1, tr2]))
    assert ft != all_the_vars

    different_rel = relay.FuncType(tvm.convert([t1, t2]), tp1,
                                   tvm.convert([tp1, tp3]),
                                   tvm.convert([tr3]))
    assert ft != different_rel

    more_rels = relay.FuncType(tvm.convert([t1, t2]), tp1,
                                   tvm.convert([tp1, tp3]),
                                   tvm.convert([tr1, tr3]))
    assert ft != more_rels


def test_tuple_type_alpha_equal():
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


def test_type_relation_alpha_equal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")
    t3 = relay.TensorType((1, 2, 3, 4), "float32")

    # functions are compared only by pointer equality so
    # we need to be sure to use the same pointers
    broadcast = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    identity = tvm.get_env_func("tvm.relay.type_relation.Identity")

    # attrs are also compared only by pointer equality
    attr1 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr2 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))

    tr = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr1)
    same = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr1)
    diff_func = relay.TypeRelation(identity, tvm.convert([t1, t2]), 1, attr1)
    diff_order = relay.TypeRelation(broadcast, tvm.convert([t2, t1]), 1, attr1)
    diff_args = relay.TypeRelation(broadcast, tvm.convert([t2, t3]), 1, attr1)
    diff_attr = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr2)

    bigger = relay.TypeRelation(identity, tvm.convert([t1, t3, t2]), 2, attr1)
    diff_num_inputs = relay.TypeRelation(identity, tvm.convert([t1, t3, t2]), 1, attr2)

    # func, number of args, input count, and order should be the same
    assert tr == same
    assert tr != diff_func
    assert tr != diff_order
    assert tr != diff_args
    assert tr != diff_attr
    assert tr != bigger

    assert bigger != diff_num_inputs


if __name__ == "__main__":
    test_tensor_type_alpha_equal()
    test_incomplete_type_alpha_equal()
    test_type_param_alpha_equal()
    test_func_type_alpha_equal()
    test_tuple_type_alpha_equal()
    test_type_relation_alpha_equal()
