import tvm
from tvm import relay
from tvm.relay.ir_pass import alpha_equal
from tvm.relay._expr import anf_print, gnf_print
import numpy as np

from hypothesis import given, reject, settings
from hypothesis.strategies import text, lists, integers, composite, recursive, deferred

exprs = deferred(lambda: constants()
                    #    | projections(exprs)
                       | tuples(exprs))

@composite
def constants(draw):
    # python_tensor = draw(recursive(integers(), lists))
    # python_tensor = draw(lists(integers(min_value=-1000, max_value=1000)))
    python_tensor = draw(integers(min_value=-1000, max_value=1000))
    # TODO: generate higher dimensional and 0D tensors. must be box shaped
    return relay.Constant(tvm.nd.array(np.array(python_tensor).astype("int32")))

@composite
def tuples(draw, field_type):
    return relay.Tuple(draw(lists(field_type, max_size=5)))

@composite
def projections(draw, field_type):
    return relay.TupleGetItem(draw(field_type), draw(integers(min_value=-1000, max_value=1000)))

# TODO: figure out a way to not have to derandomize all the time
@settings(deadline=500, derandomize=True)
@given(exprs)
def test_roundtrip_pp(e):
    alpha_equal(relay.fromtext(anf_print(e)), e)

def test_gnf():
    assert gnf_print(relay.const(1)) == "v0.0.1\n1\n"
    assert gnf_print(relay.Tuple([relay.const(1), relay.const(1)])) == "v0.0.1\n(1, 1)\n"
    one = relay.const(1)
    assert gnf_print(relay.Tuple([one, one])) == "v0.0.1\n(1, 1)\n"
        
    # assert gnf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))) == "v0.0.1\n%0 = True\nif (%0) {\n  %1 = 1\n  %2 = (%1, %1)\n  %2.0\n} else {\n  %1 = 1\n  %2 = 1\n  %3 = (%1, %1, %2)\n  %3.0\n}\n"

def test_tensor_type():
    assert gnf_print(relay.TensorType([5, 5])) == "v0.0.1\nTensor[(5, 5), float32]\n"

def test_tuple_type():
    assert gnf_print(relay.TupleType([])) == "v0.0.1\n()\n"
    assert gnf_print(relay.TupleType([relay.scalar_type("int32")])) == "v0.0.1\n(int32,)\n"
    assert gnf_print(relay.TupleType([relay.scalar_type("int32"),relay.scalar_type("int32")])) == "v0.0.1\n(int32, int32)\n"

def test_func_type():
    assert gnf_print(relay.FuncType([relay.scalar_type("int32"), relay.scalar_type("int32")], relay.scalar_type("int32"))) == "v0.0.1\nfn (int32, int32) -> int32\n"

if __name__ == "__main__":
    # for _ in range(10):
    #     print(anf_print(exprs.example()))
    one = relay.const(1)
    tup = relay.Tuple([relay.const(1), relay.const(1)])
    print(gnf_print(relay.TupleGetItem(relay.Tuple([one, one]), 0)))
    print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0)).astext())
    print(gnf_print(relay.If(relay.const(True), relay.const(1), relay.const(1))))
    print(gnf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    print(anf_print(relay.If(relay.const(True), relay.TupleGetItem(relay.Tuple([one, one]), 0), relay.TupleGetItem(relay.Tuple([one, one, relay.const(1)]), 0))))
    SEMVER = "v0.0.1"
    print(gnf_print(relay.fromtext(SEMVER+"let %x = 1; 5")))
    print(relay.fromtext(SEMVER+"let %x = 1; %x").astext())
    print(relay.fromtext(SEMVER+"let %x = (1, 1); %x").astext())
    print(relay.TupleGetItem(relay.Tuple([one, one]), 0).astext())
    print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; %x").astext())
    print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5)).astext())
    print(gnf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.Let(relay.var("x"), relay.Tuple([tup, tup]), relay.const(5))))
    print(anf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; %x")))
    print(gnf_print(relay.fromtext(SEMVER+"let %x = 1; let %x = 2; 3")))
    print(anf_print(relay.fromtext(SEMVER+"fn(%x) { %x }")))
    print(gnf_print(relay.fromtext(SEMVER+"fn(%x) { %x }")))
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
    print(gnf_print(relay.fromtext(SEMVER+"let %x = { let %y = 2; %y }; %x")))
    print(gnf_print(relay.fromtext(SEMVER+"let %x = { let %y = 2; ((%y + %y, %y * %y), 1) }; %x")))
    print(anf_print(relay.fromtext(SEMVER+"let %x = { let %y = 2; ((%y + %y, %y * %y), 1) }; %x")))
    print(relay.const([1,2,3]).astext())