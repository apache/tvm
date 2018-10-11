import tvm
from tvm import relay
from tvm.relay.parser import parse_expr, parse_prog, ParseError, Program
from tvm.relay.ir_pass import alpha_equal
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
    assert alpha_equal(
        parse_expr("let %x = 1; ()"),

        relay.Let(
            relay.Var("x"),
            to_constant(1),
            UNIT,
            TYPE_HOLE
        )
    )

def test_seq():
    assert alpha_equal(
        parse_expr("(); ()"),

        relay.Let(
            relay.Var("_"),
            UNIT,
            UNIT,
            TYPE_HOLE)
    )

    assert alpha_equal(
        parse_expr("{ (); () }; ()"),

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
    # TODO(@jmp): get function alpha eqs to work

    # assert alpha_equal(
    #     parse_expr("fn (%x) -> { %x }"),
    #     relay.Function(
    #         [relay.Param(relay.Var("x"), TYPE_HOLE)],
    #         TYPE_HOLE,
    #         relay.Var("x"),
    #         []
    #     )
    # )

    # assert alpha_equal(
    #     parse_expr("fn (%x, %y) -> { %x + %y }"),
    #     relay.Function(
    #         [relay.Param(relay.Var("x"), TYPE_HOLE),
    #          relay.Param(relay.Var("y"), TYPE_HOLE)],
    #         TYPE_HOLE,
    #         relay.add(relay.Var("x"), relay.Var("y")),
    #         []
    #     )
    # )

    id_func = parse_expr("fn (%x) -> { %x }")
    assert isinstance(id_func, relay.Function)
    assert id_func.params[0].var.name_hint == "x"
    assert isinstance(id_func.params[0].type, relay.IncompleteType)
    assert id_func.params[0].var == id_func.body

    assert isinstance(parse_expr("fn (%x, %y) -> { %x + %y }"), relay.Function)

    # annotations

    id_func_annotated = parse_expr("fn (%x: Int64) -> Int64 { %x }")
    assert id_func_annotated.params[0].type == int64
    assert id_func_annotated.ret_type == int64

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
    parse_expr(
        """
        let %constant = fn () -> { 0 };
        %constant()
        """
    )

    # assert alpha_equal(
    #     parse_expr(
    #     """
    #     let %constant = fn () -> { 0 };
    #     %constant()
    #     """
    #     ),
    #     relay.Let(
    #         relay.Var("constant"),
    #         relay.Function([], TYPE_HOLE, to_constant(0), []),
    #         relay.Call(relay.Var("constant"), [], None, None),
    #         TYPE_HOLE
    #     )
    # )

    # 1 arg
    parse_expr(
        """
        let %id = fn (%x) -> { %x };
        %id(1)
        """
    )

    # 2 args
    parse_expr(
        """
        let %multiply = fn (%x, %y) -> { %x * %y };
        %multiply(0, 0)
        """
    )

    # anonymous function
    parse_expr(
        """
        (fn (%x) -> { %x })(0)
        """
    )

    # curried function
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
    )

# Types

def test_builtin_types():
    for builtin_type in TYPES:
        parse_expr("let %_ : {} = (); ()".format(builtin_type))

def test_call_type():
    # tests e.g.
    # let %_ : Int(0) = (); ()
    # let %_ : Int(0, 1) = (); ()
    for call_type, arity in CALL_TYPES.items():
        for i in range(1, arity + 1):
            # custom tuple printing to avoid hanging comma for one-tuples
            tup = "(" + ",".join([str(num) for num in range(i)]) + ")"
            print("let %_ : {}{} = (); ()".format(call_type, tup))
            parse_expr("let %_ : {}{} = (); ()".format(call_type, tup))

def test_function_type():
    assert alpha_equal(
        parse_expr(
            """
            let %_: () -> Int64 = fn () -> Int64 { 0 }; ()
            """
        ),
        relay.Let(
            relay.Var("_"),
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
            relay.Var("_"),
            relay.Function([relay.Param(relay.Var("x"), int64)], int64, to_constant(0), []),
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
            relay.Var("_"),
            relay.Function([relay.Param(relay.Var("x"), int64), relay.Param(relay.Var("y"), int64)], int64, to_constant(0), []),
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
            relay.Var("_"),
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
            relay.Var("x"),
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
            relay.Var("x"),
            relay.Tuple([to_constant(0), to_constant(1)]),
            UNIT,
            relay.TupleType([int64, int64])
        )
    )
