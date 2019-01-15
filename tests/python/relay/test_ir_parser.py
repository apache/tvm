import tvm
from tvm import relay
from tvm.relay.parser import enabled
from tvm.relay.ir_pass import alpha_equal
from nose import SkipTest
from nose.tools import nottest, raises
from numpy import isclose
from typing import Union
from functools import wraps
if enabled():
    from tvm.relay._parser import ParseError
    raises_parse_error = raises(ParseError)
else:
    raises_parse_error = lambda x: x

BINARY_OPS = {
    "*": relay.multiply,
    "/": relay.divide,
    "+": relay.add,
    "-": relay.subtract,
    "<": relay.less,
    ">": relay.greater,
    "<=": relay.less_equal,
    ">=": relay.greater_equal,
    "==": relay.equal,
    "!=": relay.not_equal,
}

TYPES = {
    "int8",
    "int16",
    "int32",
    "int64",

    "uint8",
    "uint16",
    "uint32",
    "uint64",

    "float16",
    "float32",
    "float64",

    "bool",

    "int8x4",
    "uint1x4",
    "float16x4",
}

def get_scalar(x):
    # type: (relay.Constant) -> (Union[float, int, bool])
    return x.data.asnumpy().item()

int32 = relay.scalar_type("int32")

_ = relay.Var("_")
X = relay.Var("x")
Y = relay.Var("y")
X_ANNO = relay.Var("x", int32)
Y_ANNO = relay.Var("y", int32)

UNIT = relay.Tuple([])

# decorator to determine if parser is enabled
def if_parser_enabled(func):
    # https://stackoverflow.com/q/7727678
    @wraps(func)
    def wrapper():
        if not enabled():
            raise SkipTest("ANTLR is not installed!")
        func()
    return wrapper

@if_parser_enabled
def test_comments():
    assert alpha_equal(
        relay.fromtext("""
            // This is a line comment!
            ()
        """),
        UNIT
    )

    assert alpha_equal(
        relay.fromtext("""
            /* This is a block comment!
               This is still a block comment!
            */
            ()
        """),
        UNIT
    )

@if_parser_enabled
def test_int_literal():
    assert isinstance(relay.fromtext("1"), relay.Constant)
    assert isinstance(relay.fromtext("1").data, tvm.ndarray.NDArray)
    
    assert get_scalar(relay.fromtext("1")) == 1
    assert get_scalar(relay.fromtext("10")) == 10
    assert get_scalar(relay.fromtext("0")) == 0
    assert get_scalar(relay.fromtext("-100")) == -100
    assert get_scalar(relay.fromtext("-05")) == -5

@if_parser_enabled
def test_float_literal():
    assert get_scalar(relay.fromtext("1.0")) == 1.0
    assert isclose(get_scalar(relay.fromtext("1.56667")), 1.56667)
    assert get_scalar(relay.fromtext("0.0")) == 0.0
    assert get_scalar(relay.fromtext("-10.0")) == -10.0

    # scientific notation
    assert isclose(get_scalar(relay.fromtext("1e-1")), 1e-1)
    assert get_scalar(relay.fromtext("1e+1")) == 1e+1
    assert isclose(get_scalar(relay.fromtext("1E-1")), 1E-1)
    assert get_scalar(relay.fromtext("1E+1")) == 1E+1
    assert isclose(get_scalar(relay.fromtext("1.0e-1")), 1.0e-1)
    assert get_scalar(relay.fromtext("1.0e+1")) == 1.0e+1
    assert isclose(get_scalar(relay.fromtext("1.0E-1")), 1.0E-1)
    assert get_scalar(relay.fromtext("1.0E+1")) == 1.0E+1

@if_parser_enabled
def test_bool_literal():
    assert get_scalar(relay.fromtext("True")) == True
    assert get_scalar(relay.fromtext("False")) == False

@if_parser_enabled
def test_negative():
    assert isinstance(relay.fromtext("let %x = 1; -%x").body, relay.Call)
    assert get_scalar(relay.fromtext("--10")) == 10
    assert get_scalar(relay.fromtext("---10")) == -10

@if_parser_enabled
def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert alpha_equal(
            relay.fromtext("1 {} 1".format(bin_op)),
            BINARY_OPS.get(bin_op)(relay.const(1), relay.const(1))
        )

@if_parser_enabled
def test_parens():
    assert alpha_equal(relay.fromtext("1 * 1 + 1"), relay.fromtext("(1 * 1) + 1"))
    assert not alpha_equal(relay.fromtext("1 * 1 + 1"), relay.fromtext("1 * (1 + 1)"))

@if_parser_enabled
def test_op_assoc():
    assert alpha_equal(relay.fromtext("1 * 1 + 1 < 1 == 1"), relay.fromtext("(((1 * 1) + 1) < 1) == 1"))
    assert alpha_equal(relay.fromtext("1 == 1 < 1 + 1 * 1"), relay.fromtext("1 == (1 < (1 + (1 * 1)))"))

@nottest
@if_parser_enabled
def test_vars():
    # temp vars won't work b/c they start with a digit
    # # temp var
    # temp_var = relay.fromtext("%1")
    # assert isinstance(temp_var, relay.Var)
    # assert temp_var.name == "1"

    # var
    var = relay.fromtext("let %foo = (); %foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name_hint == "foo"

    # global var
    global_var = relay.fromtext("@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name_hint == "foo"

    # operator id
    op = relay.fromtext("foo")
    assert isinstance(op, relay.Op)
    assert op.name == "foo"

@if_parser_enabled
def test_let():
    assert alpha_equal(
        relay.fromtext("let %x = 1; ()"),
        relay.Let(
            X,
            relay.const(1),
            UNIT
        )
    )

@if_parser_enabled
def test_seq():
    assert alpha_equal(
        relay.fromtext("(); ()"),
        relay.Let(
            _,
            UNIT,
            UNIT)
    )

    assert alpha_equal(
        relay.fromtext("let %_ = { 1 }; ()"),
        relay.Let(
            X,
            relay.const(1),
            UNIT
        )
    )

@raises_parse_error
@if_parser_enabled
def test_let_global_var():
    relay.fromtext("let @x = 1; ()")

@raises_parse_error
@if_parser_enabled
def test_let_op():
    relay.fromtext("let x = 1; ()")

@if_parser_enabled
def test_tuple():
    assert alpha_equal(relay.fromtext("()"), relay.Tuple([]))

    assert alpha_equal(relay.fromtext("(0,)"), relay.Tuple([relay.const(0)]))

    assert alpha_equal(relay.fromtext("(0, 1)"), relay.Tuple([relay.const(0), relay.const(1)]))

    assert alpha_equal(relay.fromtext("(0, 1, 2)"), relay.Tuple([relay.const(0), relay.const(1), relay.const(2)]))

@if_parser_enabled
def test_func():
    # 0 args
    assert alpha_equal(
        relay.fromtext("fn () { 0 }"),
        relay.Function(
            [],
            relay.const(0),
            None,
            []
        )
    )

    # 1 arg
    assert alpha_equal(
        relay.fromtext("fn (%x) { %x }"),
        relay.Function(
            [X],
            X,
            None,
            []
        )
    )

    # 2 args
    assert alpha_equal(
        relay.fromtext("fn (%x, %y) { %x + %y }"),
        relay.Function(
            [X, Y],
            relay.add(X, Y),
            None,
            []
        )
    )

    # annotations
    assert alpha_equal(
        relay.fromtext("fn (%x: int32) -> int32 { %x }"),
        relay.Function(
            [X_ANNO],
            X_ANNO,
            int32,
            []
        )
    )

# TODO(@jmp): Crashes if %x isn't annnotated.
# @nottest
@if_parser_enabled
def test_defn():
    id_defn = relay.fromtext(
        """
        def @id(%x: int32) -> int32 {
            %x
        }
        """)
    assert isinstance(id_defn, relay.Module)

@if_parser_enabled
def test_recursive_call():
    id_defn = relay.fromtext(
        """
        def @id(%x: int32) -> int32 {
            @id(%x)
        }
        """)
    assert isinstance(id_defn, relay.Module)

@if_parser_enabled
def test_ifelse():
    assert alpha_equal(
        relay.fromtext(
        """
        if (True) {
            0
        } else {
            1
        }
        """
        ),
        relay.If(
            relay.const(True),
            relay.const(0),
            relay.const(1)
        )
    )

@raises_parse_error
@if_parser_enabled
def test_ifelse_scope():
    relay.fromtext(
        """
        if (True) {
            let %x = ();
            ()
        } else {
            %x
        }
        """
    )

@if_parser_enabled
def test_call():
    # select right function to call: simple ident case
    id_func = relay.Var("id")
    assert alpha_equal(
        relay.fromtext(
        """
        let %id = fn (%x) { %x };
        10 * %id(10)
        """
        ),
        relay.Let(
            id_func,
            relay.Function([X], X, None, []),
            relay.multiply(relay.const(10), relay.Call(id_func, [relay.const(10)]))
        )
    )

    # 0 args
    constant = relay.Var("constant")
    assert alpha_equal(
        relay.fromtext(
        """
        let %constant = fn () { 0 };
        %constant()
        """
        ),
        relay.Let(
            constant,
            relay.Function([], relay.const(0), None, []),
            relay.Call(constant, [], None, None)
        )
    )

    # 1 arg
    id_var = relay.Var("id")
    assert alpha_equal(
        relay.fromtext(
            """
            let %id = fn (%x) { %x };
            %id(1)
            """
        ),
        relay.Let(
            id_var,
            relay.Function([X], X, None, []),
            relay.Call(id_var, [relay.const(1)], None, None)
        )
    )

    # 2 args
    multiply = relay.Var("multiply")
    assert alpha_equal(
        relay.fromtext(
        """
        let %multiply = fn (%x, %y) { %x * %y };
        %multiply(0, 0)
        """
        ),
        relay.Let(
            multiply,
            relay.Function(
                [X, Y],
                relay.multiply(X, Y),
                None,
                []
            ),
            relay.Call(multiply, [relay.const(0), relay.const(0)], None, None)
        )
    )

    # anonymous function
    assert alpha_equal(
        relay.fromtext(
        """
        (fn (%x) { %x })(0)
        """
        ),
        relay.Call(
            relay.Function(
                [X],
                X,
                None,
                []
            ),
            [relay.const(0)],
            None,
            None
        )
    )

    # curried function
    curried_mult = relay.Var("curried_mult")
    alpha_equal(
        relay.fromtext(
            """
            let %curried_mult =
                fn (%x) {
                fn (%y) {
                    %x * %y
                }
                };
            %curried_mult(0);
            %curried_mult(0)(0)
            """
        ),
        relay.Let(
            curried_mult,
            relay.Function(
                [X],
                relay.Function(
                    [Y],
                    relay.multiply(X, Y),
                    None,
                    []
                ),
                None,
                []
            ),
            relay.Let(
                _,
                relay.Call(curried_mult, [relay.const(0)], None, None),
                relay.Call(relay.Call(curried_mult, [relay.const(0)], None, None), [relay.const(0)], None, None)
            )
        )
    )

    # op
    alpha_equal(
        relay.fromtext("abs(1)"),
        relay.Call(relay.op.get("abs"), [relay.const(1)], None, None)
    )

# Types

@if_parser_enabled
def test_incomplete_type():
    assert alpha_equal(
        relay.fromtext("let %_ : _ = (); ()"),
        relay.Let(
            _,
            UNIT,
            UNIT
        )
    )

@if_parser_enabled
def test_builtin_types():
    for builtin_type in TYPES:
        relay.fromtext("let %_ : {} = (); ()".format(builtin_type))

@nottest
@if_parser_enabled
def test_call_type():
    assert False

@if_parser_enabled
def test_tensor_type():
    assert alpha_equal(
        relay.fromtext("let %_ : Tensor[(), float32] = (); ()"),
        relay.Let(
            relay.Var("_", relay.TensorType((), "float32")),
            UNIT,
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext("let %_ : Tensor[(1,), float32] = (); ()"),
        relay.Let(
            relay.Var("_", relay.TensorType((1,), "float32")),
            UNIT,
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext("let %_ : Tensor[(1, 1), float32] = (); ()"),
        relay.Let(
            relay.Var("_", relay.TensorType((1, 1), "float32")),
            UNIT,
            UNIT
        )
    )

@if_parser_enabled
def test_function_type():
    assert alpha_equal(
        relay.fromtext(
            """
            let %_: fn () -> int32 = fn () -> int32 { 0 }; ()
            """
        ),
        relay.Let(
            relay.Var("_", relay.FuncType([], int32, [], [])),
            relay.Function([], relay.const(0), int32, []),
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext(
            """
            let %_: fn (int32) -> int32 = fn (%x: int32) -> int32 { 0 }; ()
            """
        ),
        relay.Let(
            relay.Var("_", relay.FuncType([int32], int32, [], [])),
            relay.Function([relay.Var("x", int32)], relay.const(0), int32, []),
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext(
            """
            let %_: fn (int32, int32) -> int32 = fn (%x: int32, %y: int32) -> int32 { 0 }; ()
            """
        ),
        relay.Let(
            relay.Var("_", relay.FuncType([int32, int32], int32, [], [])),
            relay.Function([relay.Var("x", int32), relay.Var("y", int32)], relay.const(0), int32, []),
            UNIT
        )
    )

@if_parser_enabled
def test_tuple_type():
    assert alpha_equal(
        relay.fromtext(
        """
        let %_: () = (); ()
        """),
        relay.Let(
            relay.Var("_", relay.TupleType([])),
            UNIT,
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext(
        """
        let %_: (int32,) = (0,); ()
        """),
        relay.Let(
            relay.Var("_", relay.TupleType([int32])),
            relay.Tuple([relay.const(0)]),
            UNIT
        )
    )

    assert alpha_equal(
        relay.fromtext(
        """
        let %_: (int32, int32) = (0, 1); ()
        """),
        relay.Let(
            relay.Var("_", relay.TupleType([int32, int32])),
            relay.Tuple([relay.const(0), relay.const(1)]),
            UNIT
        )
    )
