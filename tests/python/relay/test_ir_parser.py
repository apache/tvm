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
    raises_parse_error = raises(tvm._ffi.base.TVMError)
else:
    raises_parse_error = lambda x: x

SEMVER = "v0.0.1"

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

def parses_as(code, expr):
    # type: (str, relay.Expr) -> bool
    return alpha_equal(relay.fromtext(SEMVER + "\n" + code), expr)

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
    assert parses_as(
        """
        // This is a line comment!
        ()
        """,
        UNIT
    )

    assert parses_as(
        """
        /* This is a block comment!
            This is still a block comment!
        */
        ()
        """,
        UNIT
    )

@if_parser_enabled
def test_int_literal():
    assert isinstance(relay.fromtext(SEMVER+"1"), relay.Constant)
    assert isinstance(relay.fromtext(SEMVER+"1").data, tvm.ndarray.NDArray)
    
    assert get_scalar(relay.fromtext(SEMVER+"1")) == 1
    assert get_scalar(relay.fromtext(SEMVER+"10")) == 10
    assert get_scalar(relay.fromtext(SEMVER+"0")) == 0
    assert get_scalar(relay.fromtext(SEMVER+"-100")) == -100
    assert get_scalar(relay.fromtext(SEMVER+"-05")) == -5

@if_parser_enabled
def test_float_literal():
    assert get_scalar(relay.fromtext(SEMVER+"1.0")) == 1.0
    assert isclose(get_scalar(relay.fromtext(SEMVER+"1.56667")), 1.56667)
    assert get_scalar(relay.fromtext(SEMVER+"0.0")) == 0.0
    assert get_scalar(relay.fromtext(SEMVER+"-10.0")) == -10.0

    # scientific notation
    assert isclose(get_scalar(relay.fromtext(SEMVER+"1e-1")), 1e-1)
    assert get_scalar(relay.fromtext(SEMVER+"1e+1")) == 1e+1
    assert isclose(get_scalar(relay.fromtext(SEMVER+"1E-1")), 1E-1)
    assert get_scalar(relay.fromtext(SEMVER+"1E+1")) == 1E+1
    assert isclose(get_scalar(relay.fromtext(SEMVER+"1.0e-1")), 1.0e-1)
    assert get_scalar(relay.fromtext(SEMVER+"1.0e+1")) == 1.0e+1
    assert isclose(get_scalar(relay.fromtext(SEMVER+"1.0E-1")), 1.0E-1)
    assert get_scalar(relay.fromtext(SEMVER+"1.0E+1")) == 1.0E+1

@if_parser_enabled
def test_bool_literal():
    assert get_scalar(relay.fromtext(SEMVER+"True")) == True
    assert get_scalar(relay.fromtext(SEMVER+"False")) == False

@if_parser_enabled
def test_negative():
    assert isinstance(relay.fromtext(SEMVER+"let %x = 1; -%x").body, relay.Call)
    assert get_scalar(relay.fromtext(SEMVER+"--10")) == 10
    assert get_scalar(relay.fromtext(SEMVER+"---10")) == -10

@if_parser_enabled
def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert parses_as(
            "1 {} 1".format(bin_op),
            BINARY_OPS.get(bin_op)(relay.const(1), relay.const(1))
        )

@if_parser_enabled
def test_parens():
    assert alpha_equal(relay.fromtext(SEMVER+"1 * 1 + 1"), relay.fromtext(SEMVER+"(1 * 1) + 1"))
    assert not alpha_equal(relay.fromtext(SEMVER+"1 * 1 + 1"), relay.fromtext(SEMVER+"1 * (1 + 1)"))

@if_parser_enabled
def test_op_assoc():
    assert alpha_equal(relay.fromtext(SEMVER+"1 * 1 + 1 < 1 == 1"), relay.fromtext(SEMVER+"(((1 * 1) + 1) < 1) == 1"))
    assert alpha_equal(relay.fromtext(SEMVER+"1 == 1 < 1 + 1 * 1"), relay.fromtext(SEMVER+"1 == (1 < (1 + (1 * 1)))"))

@nottest
@if_parser_enabled
def test_vars():
    # temp vars won't work b/c they start with a digit
    # # temp var
    # temp_var = relay.fromtext("%1")
    # assert isinstance(temp_var, relay.Var)
    # assert temp_var.name == "1"

    # var
    var = relay.fromtext(SEMVER+"let %foo = (); %foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name_hint == "foo"

    # global var
    global_var = relay.fromtext(SEMVER+"@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name_hint == "foo"

    # operator id
    op = relay.fromtext(SEMVER+"foo")
    assert isinstance(op, relay.Op)
    assert op.name == "foo"

@if_parser_enabled
def test_let():
    assert parses_as(
        "let %x = 1; ()",
        relay.Let(
            X,
            relay.const(1),
            UNIT
        )
    )

    assert parses_as(
        """
        let %x = 1;
        let %y = 2;
        ()
        """,
        relay.Let(
            X,
            relay.const(1),
            relay.Let(
                Y,
                relay.const(2),
                UNIT
            )
        )
    )

@if_parser_enabled
def test_seq():
    assert parses_as(
        "(); ()",
        relay.Let(
            _,
            UNIT,
            UNIT)
    )

    assert parses_as(
        "let %_ = { 1 }; ()",
        relay.Let(
            X,
            relay.const(1),
            UNIT
        )
    )

@if_parser_enabled
def test_graph():
    assert parses_as(
        "%0 = (); %1 = 1; (%0, %0, %1)",
        relay.Tuple([UNIT, UNIT, relay.const(1)])
    )

    assert not parses_as(
        "%0 = (); %1 = 1; (%0, %0, %1)",
        relay.Tuple([relay.Tuple([]), relay.Tuple([]), relay.const(1)])
    )

@raises_parse_error
@if_parser_enabled
def test_graph_wrong_order():
    relay.fromtext(SEMVER+"%1 = (); %1")

@raises_parse_error
@if_parser_enabled
def test_let_global_var():
    relay.fromtext(SEMVER+"let @x = 1; ()")

@raises_parse_error
@if_parser_enabled
def test_let_op():
    relay.fromtext(SEMVER+"let x = 1; ()")

@if_parser_enabled
def test_tuple():
    assert parses_as("()", relay.Tuple([]))

    assert parses_as("(0,)", relay.Tuple([relay.const(0)]))

    assert parses_as("(0, 1)", relay.Tuple([relay.const(0), relay.const(1)]))

    assert parses_as("(0, 1, 2)", relay.Tuple([relay.const(0), relay.const(1), relay.const(2)]))

@if_parser_enabled
def test_func():
    # 0 args
    assert parses_as(
        "fn () { 0 }",
        relay.Function(
            [],
            relay.const(0),
            None,
            []
        )
    )

    # 1 arg
    assert parses_as(
        "fn (%x) { %x }",
        relay.Function(
            [X],
            X,
            None,
            []
        )
    )

    # 2 args
    assert parses_as(
        "fn (%x, %y) { %x + %y }",
        relay.Function(
            [X, Y],
            relay.add(X, Y),
            None,
            []
        )
    )

    # annotations
    assert parses_as(
        "fn (%x: int32) -> int32 { %x }",
        relay.Function(
            [X_ANNO],
            X_ANNO,
            int32,
            []
        )
    )

    # attributes
    assert parses_as(
        "fn (n=5) { () }",
        relay.Function([], UNIT, None, None, tvm.make.node("DictAttrs", n=relay.const(5)))
    )

# TODO(@jmp): Crashes if %x isn't annnotated.
@if_parser_enabled
def test_defn():
    id_defn = relay.fromtext(
        SEMVER+
        """
        def @id(%x: int32) -> int32 {
            %x
        }
        """)
    assert isinstance(id_defn, relay.Module)

@if_parser_enabled
def test_recursive_call():
    id_defn = relay.fromtext(
        SEMVER+
        """
        def @id(%x: int32) -> int32 {
            @id(%x)
        }
        """)
    assert isinstance(id_defn, relay.Module)

@if_parser_enabled
def test_ifelse():
    assert parses_as(
        """
        if (True) {
            0
        } else {
            1
        }
        """,
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
        SEMVER+
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
    assert parses_as(
        """
        let %id = fn (%x) { %x };
        10 * %id(10)
        """,
        relay.Let(
            id_func,
            relay.Function([X], X, None, []),
            relay.multiply(relay.const(10), relay.Call(id_func, [relay.const(10)]))
        )
    )

    # 0 args
    constant = relay.Var("constant")
    assert parses_as(
        """
        let %constant = fn () { 0 };
        %constant()
        """,
        relay.Let(
            constant,
            relay.Function([], relay.const(0), None, []),
            relay.Call(constant, [], None, None)
        )
    )

    # 1 arg
    id_var = relay.Var("id")
    assert parses_as(
        """
        let %id = fn (%x) { %x };
        %id(1)
        """,
        relay.Let(
            id_var,
            relay.Function([X], X, None, []),
            relay.Call(id_var, [relay.const(1)], None, None)
        )
    )

    # 2 args
    multiply = relay.Var("multiply")
    assert parses_as(
        """
        let %multiply = fn (%x, %y) { %x * %y };
        %multiply(0, 0)
        """,
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
    assert parses_as(
        """
        (fn (%x) { %x })(0)
        """,
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

    # TODO(@jmp): re-enable after sequence parsing improvements
    # curried function
    # curried_mult = relay.Var("curried_mult")
    # assert parses_as(
    #     """
    #     let %curried_mult =
    #         fn (%x) {
    #         fn (%y) {
    #             %x * %y
    #         }
    #         };
    #     %curried_mult(0);
    #     %curried_mult(0)(0)
    #     """,
    #     relay.Let(
    #         curried_mult,
    #         relay.Function(
    #             [X],
    #             relay.Function(
    #                 [Y],
    #                 relay.multiply(X, Y),
    #                 None,
    #                 []
    #             ),
    #             None,
    #             []
    #         ),
    #         relay.Let(
    #             _,
    #             relay.Call(curried_mult, [relay.const(0)], None, None),
    #             relay.Call(relay.Call(curried_mult, [relay.const(0)], None, None), [relay.const(0)], None, None)
    #         )
    #     )
    # )

    # op
    assert parses_as(
        "abs(1)",
        relay.Call(relay.op.get("abs"), [relay.const(1)], None, None)
    )

# Types

@if_parser_enabled
def test_incomplete_type():
    assert parses_as(
        "let %_ : _ = (); ()",
        relay.Let(
            _,
            UNIT,
            UNIT
        )
    )

@if_parser_enabled
def test_builtin_types():
    for builtin_type in TYPES:
        relay.fromtext(SEMVER+"let %_ : {} = (); ()".format(builtin_type))

@nottest
@if_parser_enabled
def test_call_type():
    assert False

@if_parser_enabled
def test_tensor_type():
    assert parses_as(
        "let %_ : Tensor[(), float32] = (); ()",
        relay.Let(
            relay.Var("_", relay.TensorType((), "float32")),
            UNIT,
            UNIT
        )
    )

    assert parses_as(
        "let %_ : Tensor[(1,), float32] = (); ()",
        relay.Let(
            relay.Var("_", relay.TensorType((1,), "float32")),
            UNIT,
            UNIT
        )
    )

    assert parses_as(
        "let %_ : Tensor[(1, 1), float32] = (); ()",
        relay.Let(
            relay.Var("_", relay.TensorType((1, 1), "float32")),
            UNIT,
            UNIT
        )
    )

@if_parser_enabled
def test_function_type():
    assert parses_as(
        """
        let %_: fn () -> int32 = fn () -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([], int32, [], [])),
            relay.Function([], relay.const(0), int32, []),
            UNIT
        )
    )

    assert parses_as(
        """
        let %_: fn (int32) -> int32 = fn (%x: int32) -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([int32], int32, [], [])),
            relay.Function([relay.Var("x", int32)], relay.const(0), int32, []),
            UNIT
        )
    )

    assert parses_as(
        """
        let %_: fn (int32, int32) -> int32 = fn (%x: int32, %y: int32) -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([int32, int32], int32, [], [])),
            relay.Function([relay.Var("x", int32), relay.Var("y", int32)], relay.const(0), int32, []),
            UNIT
        )
    )

@if_parser_enabled
def test_tuple_type():
    assert parses_as(
        """
        let %_: () = (); ()
        """,
        relay.Let(
            relay.Var("_", relay.TupleType([])),
            UNIT,
            UNIT
        )
    )

    assert parses_as(
        """
        let %_: (int32,) = (0,); ()
        """,
        relay.Let(
            relay.Var("_", relay.TupleType([int32])),
            relay.Tuple([relay.const(0)]),
            UNIT
        )
    )

    assert parses_as(
        """
        let %_: (int32, int32) = (0, 1); ()
        """,
        relay.Let(
            relay.Var("_", relay.TupleType([int32, int32])),
            relay.Tuple([relay.const(0), relay.const(1)]),
            UNIT
        )
    )
