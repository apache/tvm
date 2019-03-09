import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
import numpy as np

# TODO(@jmp): Re-enable later when hypothesis is added as a dependency.
# from hypothesis import given, reject, settings
# from hypothesis.strategies import text, lists, integers, composite, recursive, deferred

def gnf_print(expr):
    return expr.astext(gnf=True)

def anf_print(expr):
    return expr.astext(gnf=False)

# TODO(@jmp): Re-enable later when hypothesis is added as a dependency.
# exprs = deferred(lambda: constants()
#                     #    | projections(exprs)
#                        | tuples(exprs))

# @composite
# def constants(draw):
#     # python_tensor = draw(recursive(integers(), lists))
#     # python_tensor = draw(lists(integers(min_value=-1000, max_value=1000)))
#     python_tensor = draw(integers(min_value=-1000, max_value=1000))
#     # TODO: generate higher dimensional and 0D tensors. must be box shaped
#     return relay.Constant(tvm.nd.array(np.array(python_tensor).astype("int32")))

# @composite
# def tuples(draw, field_type):
#     return relay.Tuple(draw(lists(field_type, max_size=5)))

# @composite
# def projections(draw, field_type):
#     return relay.TupleGetItem(draw(field_type), draw(integers(min_value=-1000, max_value=1000)))

# # TODO(@jmp): figure out a way to not have to derandomize all the time
# @settings(deadline=500, derandomize=True)
# @given(exprs)
# def test_roundtrip_pp(e):
#     alpha_equal(relay.fromtext(anf_print(e)), e)

def print_parse(e, gnf = True):
    return alpha_equal(relay.fromtext(e.astext(gnf=gnf)), e)

def parse_print(s):
    s = "v0.0.1\n"+s
    return relay.fromtext(s).astext() == s

def roundtrip(e, s, gnf = True):
    return print_parse(e, gnf) and parse_print(s)

def test_gnf_simple():
    assert roundtrip(relay.const(1), "1")

def test_tuple():
    assert roundtrip(relay.Tuple([]), "()")
    assert roundtrip(relay.Tuple([relay.const(1)]), "(1,)")
    assert roundtrip(relay.Tuple([relay.const(1), relay.const(1)]), "(1, 1)")
    one = relay.const(1)
    assert print_parse(relay.Tuple([one, one]))
    tup = relay.Tuple([relay.const(1), relay.const(1)])
    assert roundtrip(relay.Tuple([tup, tup]), "%0 = (1, 1);\n(%0, %0)")
    assert print_parse(relay.Tuple([tup, tup]), gnf=False)
    assert relay.Tuple([tup, tup]).astext(gnf=False) == "v0.0.1\n((1, 1), (1, 1))"

def test_tensor_type():
    assert relay.TensorType([5, 5]).astext() == "v0.0.1\nTensor[(5, 5), float32]"

def test_tuple_type():
    assert relay.TupleType([]).astext() == "v0.0.1\n()"
    assert relay.TupleType([relay.scalar_type("int32")]).astext() == "v0.0.1\n(int32,)"
    assert relay.TupleType([relay.scalar_type("int32"),relay.scalar_type("int32")]).astext() == "v0.0.1\n(int32, int32)"

def test_func_type():
    assert relay.FuncType([relay.scalar_type("int32"), relay.scalar_type("int32")], relay.scalar_type("int32")).astext() == "v0.0.1\nfn (int32, int32) -> int32"

def test_let():
    x = relay.var("x")
    y = relay.var("y")
    assert roundtrip(relay.Let(x, relay.const(1), relay.const(5)), "let %x = 1;\n5")
    assert roundtrip(relay.Let(x, relay.const(1), x), "let %x = 1;\n%x")
    assert roundtrip(relay.Let(x, relay.Tuple([relay.const(1), relay.const(1)]), x), "let %x = (1, 1);\n%x")
    assert roundtrip(relay.Let(x, relay.Let(y, relay.const(2), y), x), "let %x = {\n  let %y = 2;\n  %y\n};\n%x")

def test_func():
    x = relay.var("x")
    assert roundtrip(relay.Function([x], x), "fn (%x) {\n  %x\n}")
    assert roundtrip(relay.Function([x], relay.Tuple([x, x])), "fn (%x) {\n  (%x, %x)\n}")

if __name__ == "__main__":
    one = relay.const(1)
    tup = relay.Tuple([relay.const(1), relay.const(1)])
    print(gnf_print(relay.TupleGetItem(relay.Tuple([one, one]), 0)))
    print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0)).astext())
    print(gnf_print(relay.If(relay.const(True), relay.const(1), relay.const(1))))
    print(gnf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    print(anf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    SEMVER = "v0.0.1"
    print(relay.TupleGetItem(relay.Tuple([one, one]), 0).astext())
    print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; %x").astext())
    print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5)).astext())
    print(gnf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; %x")))
    print(gnf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; 3")))
    print(gnf_print(relay.fromtext(SEMVER+"fn(%x) { (%x, %x) }")))
    print(gnf_print(relay.If(one, relay.TupleGetItem(relay.Tuple([one, one]), 0), one)))
    print(relay.If(relay.const(True), tup, tup).astext())
    print(gnf_print(relay.If(relay.GlobalVar("foo"), relay.TupleGetItem(relay.Tuple([one, one]), 0), one)))
    print(anf_print(relay.fromtext(SEMVER+"(fn(%x, %y) { %x })(1, 2)")))
    print(gnf_print(relay.fromtext(SEMVER+"(fn(%x, %y) { %x })(1, 2)")))
    print(relay.fromtext(SEMVER+"(fn(%x, %y) { %x })(1, 2)").astext())
    print(relay.fromtext(SEMVER+"fn(%x, %y) { %x + %y }").astext())
    print(anf_print(relay.fromtext(SEMVER+"fn(%x, %y) { %x + %y }")))
    print(gnf_print(relay.fromtext(SEMVER+"fn(%x, %y) { %x + %y }")))
    print(relay.Call(relay.fromtext(SEMVER+"fn(%x) { %x }"), [relay.const(1)], attrs=tvm.make.node("DictAttrs", n="foo")).astext())
    # print(anf_print(relay.Call(relay.fromtext(SEMVER+"fn(%x) { %x }"), [relay.const(1)], attrs=tvm.make.node("DictAttrs", n="foo"))))
    # print(relay.fromtext(SEMVER+"add(n=5)").astext())
    # print(anf_print(relay.fromtext(SEMVER+"fn (n=5) { () }")))
    x = relay.var("x", shape=(3, 2))
    y = relay.var("y")
    one = relay.const(10e10, dtype="float32")
    z = relay.add(x, one)
    z = relay.add(z, z)
    f = relay.Function([x, y], z)
    print(z.astext())
    print(f.astext())
    print(gnf_print(z))
    print(gnf_print(f))
    print(anf_print(z))
    print(anf_print(f))
    x = relay.var("x", "float32")
    y = relay.var("y", "float32")
    z = relay.add(x, y)
    z = relay.add(z, z)
    f = relay.Function([x, y], z)
    env = relay.Module()
    env["myf"] = f
    print(env.astext())
    print(gnf_print(env))
    print(anf_print(env))
    print(gnf_print(relay.fromtext(SEMVER+"let %x = { let %y = 2; ((%y + %y, %y * %y), 1) }; %x")))
    print(anf_print(relay.fromtext(SEMVER+"let %x = { let %y = 2; ((%y + %y, %y * %y), 1) }; %x")))
    print(relay.const([1,2,3]).astext())
    print(gnf_print(relay.const([1,2,3])))
    print(anf_print(relay.const([1,2,3])))
