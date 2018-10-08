import tvm
from tvm import relay
from tvm.relay.parser import parse_expr, ParseError
from nose.tools import nottest, raises

def get_scalar(x):
    return x.data.asnumpy().item()

def test_int_literal():
    assert isinstance(parse_expr("1"), relay.Constant)
    assert isinstance(parse_expr("1").data, tvm.ndarray.NDArray)
    
    assert get_scalar(parse_expr("1")) == 1
    assert get_scalar(parse_expr("10")) == 10
    assert get_scalar(parse_expr("0")) == 0
    assert get_scalar(parse_expr("-100")) == -100
    assert get_scalar(parse_expr("-05")) == -5

def test_float_literal():
    assert get_scalar(parse_expr("1.0")) == 1.0
    assert get_scalar(parse_expr("1.56667")) == 1.56667
    assert get_scalar(parse_expr("0.0")) == 0.0
    assert get_scalar(parse_expr("-10.0")) == -10.0

def test_negative():
    assert isinstance(parse_expr("let %x = 1; -%x").body, relay.Call)
    assert get_scalar(parse_expr("--10")) == 10
    assert get_scalar(parse_expr("---10")) == -10

def test_bin_op():
    assert isinstance(parse_expr("1 * 1"), relay.Call)
    assert isinstance(parse_expr("1 / 1"), relay.Call)
    assert isinstance(parse_expr("1 + 1"), relay.Call)
    assert isinstance(parse_expr("1 - 1"), relay.Call)
    assert isinstance(parse_expr("1 < 1"), relay.Call)
    assert isinstance(parse_expr("1 > 1"), relay.Call)
    assert isinstance(parse_expr("1 <= 1"), relay.Call)
    assert isinstance(parse_expr("1 >= 1"), relay.Call)
    assert isinstance(parse_expr("1 == 1"), relay.Call)
    assert isinstance(parse_expr("1 != 1"), relay.Call)

@nottest
def test_vars():
    # temp vars won't work b/c they start with a digit
    # # temp var
    # temp_var = parse_expr("%1")
    # assert isinstance(temp_var, relay.Var)
    # assert temp_var.name == "1"

    # var
    # var = parse_expr("let %foo = 0; %foo")
    var = parse_expr("%foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name == "foo"

    # global var
    global_var = parse_expr("@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name == "foo"

    # operator id
    op = parse_expr("foo")
    assert isinstance(op, relay.Op)
    assert op.name == "foo"

def test_let():
    let = parse_expr("let %x = 1; ()")
    assert isinstance(let, relay.Let)
    assert isinstance(let.var, relay.Var)
    assert isinstance(let.value, relay.Constant)
    assert get_scalar(let.value) == 1
    assert isinstance(let.body, relay.Tuple)

def test_seq():
    assert isinstance(parse_expr("(); ()"), relay.Let)
    assert parse_expr("(); ()").var.name_hint == "_"

    assert isinstance(parse_expr("{ let %x = 1; () }; ()"), relay.Let)
    assert parse_expr("{ let %x = 1; () }; ()").var.name_hint == "_"

    assert isinstance(parse_expr("{ (); () }; ()"), relay.Let)
    assert parse_expr("{ (); () }; ()").var.name_hint == "_"

@raises(ParseError)
def test_let_global_var():
    parse_expr("let @x = 1; ()")

@raises(ParseError)
def test_let_op():
    parse_expr("let x = 1; ()")

def test_tuple():
    assert isinstance(parse_expr("()"), relay.Tuple)
    assert len(parse_expr("()").fields) == 0

    assert isinstance(parse_expr("(0,)"), relay.Tuple)
    assert len(parse_expr("(0,)").fields) == 1

    assert isinstance(parse_expr("(0, 1)"), relay.Tuple)
    assert len(parse_expr("(0, 1)").fields) == 2

    assert isinstance(parse_expr("(0, 1, 2)"), relay.Tuple)
    assert len(parse_expr("(0, 1, 2)").fields) == 3

def test_func():
    id_func = parse_expr("fn (%x) => { %x }")
    assert isinstance(id_func, relay.Function)
    assert id_func.params[0].var.name_hint == "x"
    assert isinstance(id_func.params[0].type, relay.IncompleteType)
    assert id_func.params[0].var == id_func.body

    assert isinstance(parse_expr("fn (%x, %y) => { %x + %y }"), relay.Function)
