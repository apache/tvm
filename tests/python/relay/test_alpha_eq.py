"""Test alpha-equivalence of expressions and types."""
# pylint: disable=invalid-name, missing-docstring
# pylint: disable=wildcard-import, unused-wildcard-import
from relay.make import *
from relay.ir import alpha_eq, ShapeOp, Kind
from relay.typing import TYPE_DEFAULTS
from relay import ir

INT_TYPE_WIDTH = TYPE_DEFAULTS["INT_WIDTH"]
INT_TYPE_LANES = TYPE_DEFAULTS["INT_LANES"]

def int_type(width=32) -> ir.Type:
    return TensorType(IntType(width), ShapeSeq([]))

def float_type(width=32) -> ir.Type:
    return TensorType(FloatType(width), ShapeSeq([]))

def bool_type() -> ir.Type:
    return TensorType(BoolType(), ShapeSeq([]))

def nest_quantifiers(ids, body) -> ir.Type:
    ret = body
    for tid in reversed(ids):
        ret = TypeQuantifier(tid, ret)
    return ret

def test_local_id_not_eq() -> None:
    assert not alpha_eq(LocalId("x"), LocalId("y"))

def test_local_id_eq() -> None:
    x = LocalId("x")
    assert alpha_eq(x, x)

def test_global_id_not_eq() -> None:
    left = GlobalId("xyz")
    right = GlobalId("xyz")
    assert not alpha_eq(left, right)

def test_global_id_eq() -> None:
    ident = GlobalId("xyz")
    assert alpha_eq(ident, ident)

def test_operator_id_not_eq() -> None:
    left = OperatorId("xyz")
    right = OperatorId("xyz")
    # equality on operator id is pointer equality
    assert not alpha_eq(left, right)

def test_operator_id_eq() -> None:
    x = OperatorId("xyz")
    assert alpha_eq(x, x)

def test_float_literal_eq() -> None:
    x = FloatLit(1.0)
    y = FloatLit(1.0)
    assert alpha_eq(x, y)

def test_float_literal_not_eq() -> None:
    x = FloatLit(1.0)
    y = FloatLit(2.0)
    assert not alpha_eq(x, y)

def test_int_literal_eq() -> None:
    x = IntLit(1)
    y = IntLit(1)
    assert alpha_eq(x, y)

def test_int_literal_not_eq() -> None:
    x = IntLit(1)
    y = IntLit(2)
    assert not alpha_eq(x, y)

def test_bool_literal_eq() -> None:
    x = BoolLit(True)
    y = BoolLit(True)
    assert alpha_eq(x, y)

def test_bool_literal_not_eq() -> None:
    x = BoolLit(True)
    y = BoolLit(False)
    assert not alpha_eq(x, y)

def test_tensor_literal_eq() -> None:
    x = TensorLit([IntLit(1), IntLit(2)])
    y = TensorLit([IntLit(1), IntLit(2)])
    assert alpha_eq(x, y)

def test_tensor_literal_not_eq() -> None:
    x = TensorLit([IntLit(1), IntLit(2)])
    y = TensorLit([IntLit(1), IntLit(3)])
    z = TensorLit([IntLit(1)])
    assert not alpha_eq(x, y)
    assert not alpha_eq(x, z)

def test_product_literal_eq() -> None:
    x = Tuple([IntLit(1), IntLit(2)])
    y = Tuple([IntLit(1), IntLit(2)])
    assert alpha_eq(x, y)

def test_product_literal_not_eq() -> None:
    x = Tuple([IntLit(1), IntLit(2)])
    y = Tuple([IntLit(2), IntLit(2)])
    z = Tuple([IntLit(1), IntLit(2), IntLit(3)])
    assert not alpha_eq(x, y)
    assert not alpha_eq(x, z)

def test_projection_eq() -> None:
    prod = Tuple([IntLit(3), FloatLit(3.5)])

    assert alpha_eq(Projection(prod, 0), Projection(prod, 0))
    assert alpha_eq(Projection(prod, 1), Projection(prod, 1))

def test_projection_not_eq() -> None:
    prod1 = Tuple([IntLit(3), IntLit(4)])
    prod2 = Tuple([IntLit(3)])
    prod3 = Tuple([IntLit(3), IntLit(4), FloatLit(3.5)])

    assert not alpha_eq(Projection(prod1, 0), Projection(prod1, 1))
    assert not alpha_eq(Projection(prod1, 0), Projection(prod2, 0))
    assert not alpha_eq(Projection(prod1, 0), Projection(prod3, 0))
    assert not alpha_eq(Projection(prod1, 1), Projection(prod3, 1))

def test_cast_not_eq() -> None:
    left = Cast(IntType(1), IntLit(2))
    right = Cast(IntType(1), IntLit(1))
    assert not alpha_eq(left, right)

    # same literal, different type
    left = Cast(IntType(1), IntLit(2))
    right = Cast(IntType(2), IntLit(2))
    assert not alpha_eq(left, right)

def test_cast_eq() -> None:
    left = Cast(IntType(1), IntLit(2))
    right = Cast(IntType(1), IntLit(2))
    assert alpha_eq(left, right)

def test_param_not_eq() -> None:
    left = Param(LocalId("foo"), int_type())
    right = Param(LocalId("foo"), bool_type())
    assert not alpha_eq(left, right)

def test_param_eq() -> None:
    left = Param(LocalId("foo"), int_type())
    right = Param(LocalId("bar"), int_type())
    assert alpha_eq(left, right)

def test_function_not_eq() -> None:
    params1 = [Param(LocalId("x"), int_type())]
    fn1 = Function([], params1, int_type(), LocalId("x"))
    params2 = [Param(LocalId("y"), bool_type())]
    fn2 = Function([], params2, int_type(), LocalId("y"))
    assert not alpha_eq(fn1, fn2)

    params3 = [Param(LocalId("x"), int_type()), Param(LocalId("y"), int_type())]
    fn3 = Function([], params3, int_type(), LocalId("z"))
    assert not alpha_eq(fn1, fn3)

def test_function_eq() -> None:
    x = LocalId("x")
    y = LocalId("y")
    params1 = [Param(x, int_type())]
    fn1 = Function([], params1, int_type(), x)
    params2 = [Param(y, int_type())]
    fn2 = Function([], params2, int_type(), y)
    assert alpha_eq(fn1, fn2)

def test_call_not_eq() -> None:
    x = LocalId("x")
    y = LocalId("y")
    params1 = [Param(x, int_type())]
    fn1 = Function([], params1, int_type(), x)
    args1 = [IntLit(1)]
    call1 = Call(fn1, args1)

    args2 = [IntLit(2)]
    call2 = Call(fn1, args2)
    assert not alpha_eq(call1, call2)

    params2 = [Param(y, int_type())]
    fn2 = Function([], params2, float_type(), FloatLit(0.0))
    call3 = Call(fn2, args1)
    assert not alpha_eq(call1, call3)
    assert not alpha_eq(call2, call3)

def test_call_eq() -> None:
    x = LocalId("x")
    y = LocalId("y")
    params1 = [Param(x, int_type())]
    fn1 = Function([], params1, int_type(), x)
    args = [IntLit(1)]
    call1 = Call(fn1, args)

    params2 = [Param(y, int_type())]
    fn2 = Function([], params2, int_type(), y)
    call2 = Call(fn2, args)
    assert alpha_eq(call1, call2)

def test_debug_not_eq() -> None:
    left = Debug(IntLit(1))
    right = Debug(IntLit(2))
    assert not alpha_eq(left, right)

def test_debug_eq() -> None:
    left = Debug(IntLit(1))
    right = Debug(IntLit(1))
    assert alpha_eq(left, right)

def test_let_not_eq() -> None:
    x = LocalId("x")
    y = LocalId("y")
    let1 = Let(x, int_type(), IntLit(10), IntLit(11))
    let2 = Let(y, int_type(), IntLit(10), IntLit(12))
    assert not alpha_eq(let1, let2)

    let3 = Let(x, int_type(), IntLit(10), x)
    let4 = Let(y, int_type(), IntLit(12), y)
    assert not alpha_eq(let3, let4)

def test_let_eq() -> None:
    x = LocalId("x")
    y = LocalId("y")
    let1 = Let(x, int_type(), IntLit(10), x)
    let2 = Let(y, int_type(), IntLit(10), y)
    assert alpha_eq(let1, let2)

def test_ref_eq() -> None:
    r1 = Ref(IntLit(5))
    r2 = Ref(IntLit(5))
    assert alpha_eq(r1, r2)

def test_ref_not_eq() -> None:
    r1 = Ref(IntLit(5))
    r2 = Ref(FloatLit(3.5))
    r3 = Ref(r1)
    assert not alpha_eq(r1, r2)
    assert not alpha_eq(r1, r3)
    assert not alpha_eq(r2, r3)

def test_val_ref_eq() -> None:
    vr1 = ReadRef(Ref(IntLit(35)))
    vr2 = ReadRef(Ref(Tuple([IntLit(12), FloatLit(2.5)])))
    assert alpha_eq(vr1, vr1)
    assert alpha_eq(vr2, vr2)

def test_val_ref_not_eq() -> None:
    vr1 = ReadRef(Ref(IntLit(5)))
    vr2 = ReadRef(Ref(vr1))
    vr3 = ReadRef(Ref(FloatLit(5.0)))
    assert not alpha_eq(vr1, vr2)
    assert not alpha_eq(vr1, vr3)
    assert not alpha_eq(vr2, vr3)

def test_set_ref_eq() -> None:
    sr1 = WriteRef(Ref(FloatLit(5.0)), FloatLit(6.0))
    sr2 = WriteRef(Ref(Tuple([IntLit(3), BoolLit(False)])),
                 Tuple([IntLit(5), BoolLit(True)]))
    assert alpha_eq(sr1, sr1)
    assert alpha_eq(sr2, sr2)

def test_set_ref_not_eq() -> None:
    r1 = Ref(FloatLit(5.0))
    r2 = Ref(IntLit(5))
    r3 = Ref(IntLit(6))

    assert not alpha_eq(WriteRef(r1, FloatLit(6.0)),
                        WriteRef(r2, IntLit(6)))
    assert not alpha_eq(WriteRef(r2, IntLit(6)), WriteRef(r2, IntLit(7)))
    assert not alpha_eq(WriteRef(r2, IntLit(7)), WriteRef(r3, IntLit(7)))

# Type alpha-equality tests

def test_base_type_eq() -> None:
    assert alpha_eq(IntType(32), IntType(32))
    assert alpha_eq(BoolType(), BoolType())
    assert alpha_eq(FloatType(32), FloatType(32))

def test_tensor_type_eq() -> None:
    tt1 = TensorType(
        IntType(32), ShapeSeq([ShapeSingleton(1), ShapeSingleton(2), ShapeSingleton(3)]))
    tt2 = TensorType(
        FloatType(32), ShapeSeq([ShapeSingleton(3), ShapeSingleton(3)]))
    assert alpha_eq(tt1, tt1)
    assert alpha_eq(tt2, tt2)

def test_tensor_type_not_eq() -> None:
    tt1 = TensorType(
        IntType(32), ShapeSeq([ShapeSingleton(1), ShapeSingleton(2), ShapeSingleton(3)]))
    tt2 = TensorType(
        FloatType(32), ShapeSeq([ShapeSingleton(1), ShapeSingleton(2), ShapeSingleton(3)]))
    tt3 = TensorType(
        IntType(32), ShapeSeq([ShapeSingleton(3), ShapeSingleton(3)]))
    assert not alpha_eq(tt1, tt2)
    assert not alpha_eq(tt1, tt3)

def test_ref_type_eq() -> None:
    rt1 = RefType(int_type())
    rt2 = RefType(float_type())
    assert alpha_eq(rt1, rt1)
    assert alpha_eq(rt2, rt2)

def test_ref_type_not_eq() -> None:
    rt1 = RefType(int_type())
    rt2 = RefType(float_type())
    assert not alpha_eq(rt1, rt2)

def test_product_type_eq() -> None:
    pt1 = TupleType([int_type(), RefType(float_type())])
    pt2 = TupleType([float_type(), float_type(), int_type()])
    assert alpha_eq(pt1, pt1)
    assert alpha_eq(pt2, pt2)

def test_product_type_not_eq() -> None:
    pt1 = TupleType([int_type(), int_type()])
    pt2 = TupleType([int_type(), int_type(), float_type()])
    pt3 = TupleType([bool_type(), float_type()])
    assert not alpha_eq(pt1, pt2)
    assert not alpha_eq(pt1, pt3)

def test_type_id_eq() -> None:
    id1 = TypeParam("id1", Kind.Shape)
    id2 = TypeParam("id2", Kind.BaseType)
    id3 = TypeParam("id2", Kind.Type)

    assert alpha_eq(id1, id1)
    assert alpha_eq(id2, id2)
    assert alpha_eq(id3, id3)

def test_type_id_not_eq() -> None:
    # name is just a hint, we use pointer equality as the rule
    # (unless there is a quantifier to give context)
    id1 = TypeParam("id1", Kind.Shape)
    id2 = TypeParam("id1", Kind.Shape)
    id3 = TypeParam("id3", Kind.BaseType)

    assert not alpha_eq(id1, id2)
    assert not alpha_eq(id1, id3)

def test_arrow_type_eq() -> None:
    ar1 = TypeArrow([int_type()], bool_type())
    ar2 = TypeArrow([int_type(), int_type()], TupleType([]))
    assert alpha_eq(ar1, ar1)
    assert alpha_eq(ar2, ar2)

def test_arrow_type_not_eq() -> None:
    t1 = int_type()
    t2 = bool_type()
    t3 = [int_type(), bool_type()]

    assert not alpha_eq(TypeArrow([t1], t2), TypeArrow([t1], t1))
    assert not alpha_eq(TypeArrow(t3, t1), TypeArrow([t2], t1))
    assert not alpha_eq(TypeArrow([t1], TypeArrow([t1], t1)),
                        TypeArrow([t1], t1))

def test_type_quantifier_eq() -> None:
    id1 = TypeParam("id1", Kind.Shape)
    id2 = TypeParam("id2", Kind.Shape)
    tq1 = TypeQuantifier(id1, TensorType(IntType(32), id1))
    tq2 = TypeQuantifier(id2, TensorType(IntType(32), id2))

    assert alpha_eq(tq1, tq1)
    assert alpha_eq(tq1, tq2)

def test_nested_type_quantifier_eq() -> None:
    id1 = TypeParam("id1", Kind.BaseType)
    id2 = TypeParam("id2", Kind.Shape)
    id3 = TypeParam("id3", Kind.BaseType)
    id4 = TypeParam("id4", Kind.Shape)
    tq1 = TypeQuantifier(id1, TypeQuantifier(id2, TensorType(id1, id2)))
    tq2 = TypeQuantifier(id3, TypeQuantifier(id4, TensorType(id3, id4)))

    assert alpha_eq(tq1, tq1)
    assert alpha_eq(tq1, tq2)

def test_type_quantifier_not_eq() -> None:
    id1 = TypeParam("id1", Kind.Shape)
    id2 = TypeParam("id2", Kind.BaseType)
    id3 = TypeParam("id3", Kind.Shape)

    tq1 = TypeQuantifier(id1, TensorType(IntType(32), id1))
    tq2 = TypeQuantifier(id2, TensorType(id2, ShapeSeq([ShapeSingleton(3)])))
    tq3 = TypeQuantifier(id1, TensorType(IntType(32), id3))
    tq4 = TypeQuantifier(id1, TensorType(FloatType(32), id1))

    assert not alpha_eq(tq1, tq2)
    assert not alpha_eq(tq1, tq3)
    assert not alpha_eq(tq1, tq4)
    assert not alpha_eq(tq2, tq3)
    assert not alpha_eq(tq2, tq4)

def test_shape_singleton_eq() -> None:
    single1 = ShapeSingleton(10)
    single2 = ShapeSingleton(10)

    assert alpha_eq(single1, single1)
    assert alpha_eq(single1, single2)

def test_shape_singelton_not_eq() -> None:
    single1 = ShapeSingleton(10)
    single2 = ShapeSingleton(11)

    assert not alpha_eq(single1, single2)

def test_shape_attr_eq() -> None:
    attr1 = ShapeAttr("x")
    attr2 = ShapeAttr("x")

    assert alpha_eq(attr1, attr1)
    assert alpha_eq(attr1, attr2)

def test_shape_attr_not_eq() -> None:
    id1 = "x"
    id2 = "y"
    attr1 = ShapeAttr(id1)
    attr2 = ShapeAttr(id2)

    assert not alpha_eq(attr1, attr2)

def test_shape_seq_eq() -> None:
    empty = ShapeSeq([])
    seq1 = ShapeSeq([ShapeSingleton(5)])
    seq2 = ShapeSeq([ShapeSingleton(5)])

    assert alpha_eq(empty, empty)
    assert alpha_eq(seq1, seq2)

def test_shape_seq_not_eq() -> None:
    empty = ShapeSeq([])
    seq = ShapeSeq([ShapeSingleton(5)])
    single = ShapeSingleton(5)

    assert not alpha_eq(empty, seq)
    assert not alpha_eq(seq, single)

def test_shape_projection_eq() -> None:
    proj1 = ShapeProjection(ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)]), 0)
    proj2 = ShapeProjection(ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)]), 0)

    assert alpha_eq(proj1, proj2)

def test_shape_projection_not_eq() -> None:
    proj1 = ShapeProjection(ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)]), 0)
    proj2 = ShapeProjection(ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)]), 1)
    proj3 = ShapeProjection(ShapeSeq([ShapeSingleton(2), ShapeSingleton(1)]), 0)
    proj4 = ShapeProjection(ShapeSeq([ShapeSingleton(2), ShapeSingleton(1)]), 1)

    assert not alpha_eq(proj1, proj2)
    assert not alpha_eq(proj1, proj3)
    assert not alpha_eq(proj1, proj4)
    assert not alpha_eq(proj2, proj3)
    assert not alpha_eq(proj2, proj4)
    assert not alpha_eq(proj3, proj4)

def test_shape_binary_op_eq() -> None:
    empty = ShapeSeq([])
    single = ShapeSingleton(5)
    seq = ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)])

    op1 = ShapeBinaryOp(ShapeOp.SHPLUS, empty, empty)
    op2 = ShapeBinaryOp(ShapeOp.SHSUB, single, single)
    op3 = ShapeBinaryOp(ShapeOp.SHMUL, seq, seq)
    op4 = ShapeBinaryOp(ShapeOp.SHDIV, seq, seq)

    assert alpha_eq(op1, op1)
    assert alpha_eq(op2, op2)
    assert alpha_eq(op3, op3)
    assert alpha_eq(op4, op4)

def test_shape_binary_op_not_eq() -> None:
    empty = ShapeSeq([])
    single = ShapeSingleton(5)
    seq = ShapeSeq([ShapeSingleton(1), ShapeSingleton(2)])

    assert not alpha_eq(ShapeBinaryOp(ShapeOp.SHPLUS, empty, empty), empty)
    assert not alpha_eq(ShapeBinaryOp(ShapeOp.SHMUL, seq, ShapeSingleton(1)), seq)
    assert not alpha_eq(
        ShapeBinaryOp(ShapeOp.SHPLUS, single, single),
        ShapeBinaryOp(ShapeOp.SHPLUS,
                      ShapeSeq([single]),
                      ShapeSeq([single])))
    assert not alpha_eq(
        ShapeBinaryOp(ShapeOp.SHPLUS, empty, empty),
        ShapeBinaryOp(ShapeOp.SHSUB, empty, empty))
    assert not alpha_eq(
        ShapeBinaryOp(ShapeOp.SHMUL, empty, empty),
        ShapeBinaryOp(ShapeOp.SHDIV, empty, empty))

def test_shape_nested_in_quantifier() -> None:
    b1 = TypeParam("b", Kind.BaseType)
    x1 = TypeParam("x", Kind.Shape)
    y1 = TypeParam("y", Kind.Shape)

    b2 = TypeParam("b", Kind.BaseType)
    x2 = TypeParam("x", Kind.Shape)
    y2 = TypeParam("y", Kind.Shape)

    b3 = TypeParam("b", Kind.BaseType)
    x3 = TypeParam("x", Kind.Shape)
    y3 = TypeParam("y", Kind.Shape)

    tq1 = nest_quantifiers(
        [b1, x1, y1],
        TypeArrow(
            [TensorType(b1, x1), TensorType(b1, y2)],
            TensorType(
                b1,
                ShapeBinaryOp(ShapeOp.SHPLUS,
                              ShapeSeq([x1, ShapeProjection(y1, 1),
                                        ShapeSingleton(5), ShapeAttr("att")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    tq2 = nest_quantifiers(
        [b2, x2, y2],
        TypeArrow(
            [TensorType(b2, x2), TensorType(b2, y2)],
            TensorType(
                b2,
                ShapeBinaryOp(ShapeOp.SHPLUS,
                              ShapeSeq([x2, ShapeProjection(y2, 1),
                                        ShapeSingleton(5), ShapeAttr("att")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    # different attr, var order, position, and constant
    tq3 = nest_quantifiers(
        [b3, x3, y3],
        TypeArrow(
            [TensorType(b3, x3), TensorType(b3, y3)],
            TensorType(
                b3,
                ShapeBinaryOp(ShapeOp.SHPLUS,
                              ShapeSeq([x3, ShapeProjection(y3, 1),
                                        ShapeSingleton(4), ShapeAttr("att")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    tq4 = nest_quantifiers(
        [b3, x3, y3],
        TypeArrow(
            [TensorType(b3, x3), TensorType(b3, y3)],
            TensorType(
                b3,
                ShapeBinaryOp(ShapeOp.SHPLUS,
                              ShapeSeq([x3, ShapeProjection(y3, 2),
                                        ShapeSingleton(5), ShapeAttr("att2")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    tq5 = nest_quantifiers(
        [b3, x3, y3],
        TypeArrow(
            [TensorType(b3, x3), TensorType(b3, y3)],
            TensorType(
                b3,
                ShapeBinaryOp(ShapeOp.SHMUL,
                              ShapeSeq([x3, ShapeProjection(y3, 1),
                                        ShapeSingleton(5), ShapeAttr("att")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    tq6 = nest_quantifiers(
        [b3, y3, x3],
        TypeArrow(
            [TensorType(b3, x3), TensorType(b3, y3)],
            TensorType(
                b3,
                ShapeBinaryOp(ShapeOp.SHPLUS,
                              ShapeSeq([x3, ShapeProjection(y3, 1),
                                        ShapeSingleton(5), ShapeAttr("att")]),
                              ShapeSeq([ShapeSingleton(1) for i in range(6)])))))

    assert alpha_eq(tq1, tq2)
    assert not alpha_eq(tq1, tq3)
    assert not alpha_eq(tq2, tq3)
    assert not alpha_eq(tq1, tq4)
    assert not alpha_eq(tq2, tq4)
    assert not alpha_eq(tq1, tq5)
    assert not alpha_eq(tq2, tq5)
    assert not alpha_eq(tq1, tq6)
    assert not alpha_eq(tq2, tq6)
