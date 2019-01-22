from tvm import relay
from tvm.relay.ir_pass import infer_type

def test_dup_type():
    a = relay.TypeVar("a")
    av = relay.Var("av", a)
    make_id = relay.Function([av], relay.Tuple([av, av]), None, [a])
    t = relay.scalar_type("float32")
    b = relay.Var("b", t)
    assert relay.ir_pass.infer_type(make_id(b)).checked_type == relay.TupleType([t, t])


def test_id_type():
    mod = relay.Module()
    id_type = relay.TypeVar("id")
    a = relay.TypeVar("a")
    make_id = relay.Var("make_id", relay.FuncType([a], id_type(a), [a]))
    t = relay.scalar_type("float32")
    b = relay.Var("b", t)
    assert relay.ir_pass.infer_type(relay.Call(make_id, [b]), mod).checked_type == id_type(t)


if __name__ == "__main__":
    test_dup_type()
    test_id_type()
