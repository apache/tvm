import tvm
from tvm import relay
from tvm.relay.parser import parse_expr, parse_prog, ParseError, Program
from tvm.relay.ir_pass import alpha_equal
from tvm.relay.expr import pretty_print
from nose.tools import nottest, raises
from typing import Union

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
    "Int8",
    "Int16",
    "Int32",
    "Int64",

    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",

    "Float16",
    "Float32",
    "Float64",

    "Bool",
}

CALL_TYPES = {
    "Int": 2,
    "UInt": 2,
    "Float": 2,
    "Bool": 1,
}

def get_scalar(x):
    # type: (relay.Constant) -> (Union[float, int, bool])
    return x.data.asnumpy().item()

def to_constant(x):
    # type: (Union[float, int, bool]) -> relay.Constant
    return relay.Constant(tvm.nd.array(x))

def to_tensor_type(x):
    # type: (str) -> relay.TensorType
    return relay.TensorType([], x)

_ = relay.Var("_")
X = relay.Var("x")
Y = relay.Var("y")

int64 = to_tensor_type("int64")

UNIT = relay.Tuple([])
TYPE_HOLE = relay.IncompleteType()

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

    # scientific notation
    assert get_scalar(parse_expr("1e-1")) == 1e-1
    assert get_scalar(parse_expr("1e+1")) == 1e+1
    assert get_scalar(parse_expr("1E-1")) == 1E-1
    assert get_scalar(parse_expr("1E+1")) == 1E+1
    assert get_scalar(parse_expr("1.0e-1")) == 1.0e-1
    assert get_scalar(parse_expr("1.0e+1")) == 1.0e+1
    assert get_scalar(parse_expr("1.0E-1")) == 1.0E-1
    assert get_scalar(parse_expr("1.0E+1")) == 1.0E+1

def test_bool_literal():
    assert get_scalar(parse_expr("true")) == True
    assert get_scalar(parse_expr("false")) == False

def test_negative():
    assert isinstance(parse_expr("let %x = 1; -%x").body, relay.Call)
    assert get_scalar(parse_expr("--10")) == 10
    assert get_scalar(parse_expr("---10")) == -10

def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert alpha_equal(
            parse_expr("1 {} 1".format(bin_op)),
            BINARY_OPS.get(bin_op)(to_constant(1), to_constant(1))
        )

def test_parens():
    assert alpha_equal(parse_expr("1 * 1 + 1"), parse_expr("(1 * 1) + 1"))
    assert not alpha_equal(parse_expr("1 * 1 + 1"), parse_expr("1 * (1 + 1)"))

def test_op_assoc():
    assert alpha_equal(parse_expr("1 * 1 + 1 < 1 == 1"), parse_expr("(((1 * 1) + 1) < 1) == 1"))
    assert alpha_equal(parse_expr("1 == 1 < 1 + 1 * 1"), parse_expr("1 == (1 < (1 + (1 * 1)))"))

@nottest
def test_vars():
    # temp vars won't work b/c they start with a digit
    # # temp var
    # temp_var = parse_expr("%1")
    # assert isinstance(temp_var, relay.Var)
    # assert temp_var.name == "1"

    # var
    var = parse_expr("let %foo = (); %foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name_hint == "foo"

    # global var
    global_var = parse_expr("@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name_hint == "foo"

    # operator id
    op = parse_expr("foo")
    assert isinstance(op, relay.Op)
    assert op.name == "foo"

def test_let():
    assert alpha_equal(
        parse_expr("let %x = 1; ()"),

        relay.Let(
            X,
            to_constant(1),
            UNIT,
            TYPE_HOLE
        )
    )

def test_seq():
    assert alpha_equal(
        parse_expr("(); ()"),

        relay.Let(
            _,
            UNIT,
            UNIT,
            TYPE_HOLE)
    )

    assert alpha_equal(
        parse_expr("{ (); () }; ()"),
        # Can't use _ constant, because the _'s are different.
        relay.Let(
            relay.Var("_"),
            relay.Let(relay.Var("_"), UNIT, UNIT, TYPE_HOLE),
            UNIT,
            TYPE_HOLE)
    )

@raises(ParseError)
def test_seq_scope():
    parse_expr("{ let %x = 1; %x }; %x")

@raises(ParseError)
def test_let_global_var():
    parse_expr("let @x = 1; ()")

@raises(ParseError)
def test_let_op():
    parse_expr("let x = 1; ()")

def test_tuple():
    assert alpha_equal(parse_expr("()"), relay.Tuple([]))

    assert alpha_equal(parse_expr("(0,)"), relay.Tuple([to_constant(0)]))

    assert alpha_equal(parse_expr("(0, 1)"), relay.Tuple([to_constant(0), to_constant(1)]))

    assert alpha_equal(parse_expr("(0, 1, 2)"), relay.Tuple([to_constant(0), to_constant(1), to_constant(2)]))

def test_func():
    # 0 args
    assert alpha_equal(
        parse_expr("fn () -> { 0 }"),
        relay.Function(
            [],
            TYPE_HOLE,
            to_constant(0),
            []
        )
    )

    # 1 arg
    assert alpha_equal(
        parse_expr("fn (%x) -> { %x }"),
        relay.Function(
            [relay.Param(X, TYPE_HOLE)],
            TYPE_HOLE,
            X,
            []
        )
    )

    # 2 args
    assert alpha_equal(
        parse_expr("fn (%x, %y) -> { %x + %y }"),
        relay.Function(
            [relay.Param(X, TYPE_HOLE),
             relay.Param(Y, TYPE_HOLE)],
            TYPE_HOLE,
            relay.add(X, Y),
            []
        )
    )

    # annotations
    assert alpha_equal(
        parse_expr("fn (%x: Int64) -> Int64 { %x }"),
        relay.Function(
            [relay.Param(X, int64)],
            int64,
            X,
            []
        )
    )

# TODO(@jmp): Figure out why this is crashing.
@nottest
def test_defn():
    id_defn = parse_prog("def @id(%x) -> { %x }")
    assert isinstance(id_defn, Program)

def test_ifelse():
    assert alpha_equal(
        parse_expr(
        """
        if (true) {
            0
        } else {
            1
        }
        """
        ),
        relay.If(
            to_constant(True),
            to_constant(0),
            to_constant(1)
        )
    )

@raises(ParseError)
def test_ifelse_scope():
    parse_expr(
        """
        if (true) {
            let %x = ();
            ()
        } else {
            %x
        }
        """
    )

def test_call():
    # 0 args
    constant = relay.Var("constant")
    assert alpha_equal(
        parse_expr(
        """
        let %constant = fn () -> { 0 };
        %constant()
        """
        ),
        relay.Let(
            constant,
            relay.Function([], TYPE_HOLE, to_constant(0), []),
            relay.Call(constant, [], None, None),
            TYPE_HOLE
        )
    )

    # 1 arg
    id_var = relay.Var("id")
    assert alpha_equal(
        parse_expr(
            """
            let %id = fn (%x) -> { %x };
            %id(1)
            """
        ),
        relay.Let(
            id_var,
            relay.Function([relay.Param(X, TYPE_HOLE)], TYPE_HOLE, X, []),
            relay.Call(id_var, [to_constant(1)], None, None),
            TYPE_HOLE
        )
    )

    # 2 args
    multiply = relay.Var("multiply")
    assert alpha_equal(
        parse_expr(
        """
        let %multiply = fn (%x, %y) -> { %x * %y };
        %multiply(0, 0)
        """
        ),
        relay.Let(
            multiply,
            relay.Function(
                [relay.Param(X, TYPE_HOLE), relay.Param(Y, TYPE_HOLE)],
                TYPE_HOLE,
                relay.multiply(X, Y),
                []
            ),
            relay.Call(multiply, [to_constant(0), to_constant(0)], None, None),
            TYPE_HOLE
        )
    )

    # anonymous function
    assert alpha_equal(
        parse_expr(
        """
        (fn (%x) -> { %x })(0)
        """
        ),
        relay.Call(
            relay.Function(
                [relay.Param(X, TYPE_HOLE)],
                TYPE_HOLE,
                X,
                []
            ),
            [to_constant(0)],
            None,
            None
        )
    )

    # curried function
    curried_mult = relay.Var("curried_mult")
    alpha_equal(
        parse_expr(
            """
            let %curried_mult =
                fn (%x) -> {
                fn (%y) -> {
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
                [relay.Param(X, TYPE_HOLE)],
                TYPE_HOLE,
                relay.Function(
                    [relay.Param(Y, TYPE_HOLE)],
                    TYPE_HOLE,
                    relay.multiply(X, Y),
                    []
                ),
                []
            ),
            relay.Let(
                _,
                relay.Call(curried_mult, [to_constant(0)], None, None),
                relay.Call(relay.Call(curried_mult, [to_constant(0)], None, None), [to_constant(0)], None, None),
                TYPE_HOLE
            ),
            TYPE_HOLE
        )
    )

# Types

def test_builtin_types():
    for builtin_type in TYPES:
        parse_expr("let %_ : {} = (); ()".format(builtin_type))

def test_call_type():
    # tests e.g.
    # let %_ : Int[0] = (); ()
    # let %_ : Int[0, 1] = (); ()
    for call_type, arity in CALL_TYPES.items():
        args = []
        for i in range(arity):
            args.append(i)
            parse_expr("let %_ : {}{} = (); ()".format(call_type, args))

def test_function_type():
    assert alpha_equal(
        parse_expr(
            """
            let %_: () -> Int64 = fn () -> Int64 { 0 }; ()
            """
        ),
        relay.Let(
            _,
            relay.Function([], int64, to_constant(0), []),
            UNIT,
            relay.FuncType([], int64, [], [])
        )
    )

    assert alpha_equal(
        parse_expr(
            """
            let %_: (Int64) -> Int64 = fn (%x: Int64) -> Int64 { 0 }; ()
            """
        ),
        relay.Let(
            _,
            relay.Function([relay.Param(X, int64)], int64, to_constant(0), []),
            UNIT,
            relay.FuncType([int64], int64, [], [])
        )
    )

    assert alpha_equal(
        parse_expr(
            """
            let %_: (Int64, Int64) -> Int64 = fn (%x: Int64, %y: Int64) -> Int64 { 0 }; ()
            """
        ),
        relay.Let(
            _,
            relay.Function([relay.Param(X, int64), relay.Param(Y, int64)], int64, to_constant(0), []),
            UNIT,
            relay.FuncType([int64, int64], int64, [], [])
        )
    )

def test_tuple_type():
    assert alpha_equal(
        parse_expr(
        """
        let %_: () = (); ()
        """),
        relay.Let(
            _,
            UNIT,
            UNIT,
            relay.TupleType([])
        )
    )

    assert alpha_equal(
        parse_expr(
        """
        let %x: (Int64,) = (0,); ()
        """),
        relay.Let(
            X,
            relay.Tuple([to_constant(0)]),
            UNIT,
            relay.TupleType([int64])
        )
    )

    assert alpha_equal(
        parse_expr(
        """
        let %x: (Int64, Int64) = (0, 1); ()
        """),
        relay.Let(
            X,
            relay.Tuple([to_constant(0), to_constant(1)]),
            UNIT,
            relay.TupleType([int64, int64])
        )
    )
