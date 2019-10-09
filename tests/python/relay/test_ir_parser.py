# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm
from tvm import relay
from tvm.relay.analysis import graph_equal, assert_graph_equal
from tvm.relay.analysis import alpha_equal, assert_alpha_equal
import pytest
from numpy import isclose
from typing import Union
from functools import wraps
raises_parse_error = pytest.mark.xfail(raises=tvm._ffi.base.TVMError)

SEMVER = "v0.0.4"

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

LIST_DEFN = """
type List[A] {
    Cons(A, List[A]),
    Nil,
}
"""

def roundtrip(expr):
    x = relay.fromtext(str(expr))
    assert_graph_equal(x, expr)


def parse_text(code):
    expr = relay.fromtext(SEMVER + "\n" + code)
    roundtrip(expr)
    return expr


def parses_as(code, expr):
    # type: (str, relay.Expr) -> bool
    parsed = parse_text(code)
    result = graph_equal(parsed, expr)
    return result

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

    assert parses_as(
        """
        /* This is a block comment!
           /*Block comment is recursive!*/
        */
        ()
        """,
        UNIT
    )


def test_int_literal():
    assert isinstance(parse_text("1"), relay.Constant)
    assert isinstance(parse_text("1").data, tvm.ndarray.NDArray)

    assert get_scalar(parse_text("1")) == 1
    assert get_scalar(parse_text("10")) == 10
    assert get_scalar(parse_text("0")) == 0
    assert get_scalar(parse_text("-100")) == -100
    assert get_scalar(parse_text("-05")) == -5


def test_float_literal():
    assert get_scalar(parse_text("1.0f")) == 1.0
    assert isclose(get_scalar(parse_text("1.56667f")), 1.56667)
    assert get_scalar(parse_text("0.0f")) == 0.0
    assert get_scalar(parse_text("-10.0f")) == -10.0

    # scientific notation
    assert isclose(get_scalar(parse_text("1e-1f")), 1e-1)
    assert get_scalar(parse_text("1e+1f")) == 1e+1
    assert isclose(get_scalar(parse_text("1E-1f")), 1E-1)
    assert get_scalar(parse_text("1E+1f")) == 1E+1
    assert isclose(get_scalar(parse_text("1.0e-1f")), 1.0e-1)
    assert get_scalar(parse_text("1.0e+1f")) == 1.0e+1
    assert isclose(get_scalar(parse_text("1.0E-1f")), 1.0E-1)
    assert get_scalar(parse_text("1.0E+1f")) == 1.0E+1


def test_bool_literal():
    assert get_scalar(parse_text("True")) == True
    assert get_scalar(parse_text("False")) == False


def test_negative():
    assert isinstance(parse_text("let %x = 1; -%x").body, relay.Call)
    assert get_scalar(parse_text("--10")) == 10
    assert get_scalar(parse_text("---10")) == -10


def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert parses_as(
            "1 {} 1".format(bin_op),
            BINARY_OPS.get(bin_op)(relay.const(1), relay.const(1))
        )


def test_parens():
    assert graph_equal(parse_text("1 * 1 + 1"), parse_text("(1 * 1) + 1"))
    assert not graph_equal(parse_text("1 * 1 + 1"), parse_text("1 * (1 + 1)"))


def test_op_assoc():
    assert graph_equal(parse_text("1 * 1 + 1 < 1 == 1"), parse_text("(((1 * 1) + 1) < 1) == 1"))
    assert graph_equal(parse_text("1 == 1 < 1 + 1 * 1"), parse_text("1 == (1 < (1 + (1 * 1)))"))


@pytest.mark.skip
def test_vars():
    # temp vars won't work b/c they start with a digit
    # # temp var
    # temp_var = parse_text("%1")
    # assert isinstance(temp_var, relay.Var)
    # assert temp_var.name == "1"

    # var
    var = parse_text("let %foo = (); %foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name_hint == "foo"

    # global var
    global_var = parse_text("@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name_hint == "foo"

    # operator id
    op = parse_text("foo")
    assert isinstance(op, relay.Op)
    assert op.name == "foo"


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


def test_seq():
    assert parses_as(
        "();; ()",
        relay.Let(
            _,
            UNIT,
            UNIT)
    )

    assert parses_as(
        "let %_ = 1; ()",
        relay.Let(
            X,
            relay.const(1),
            UNIT
        )
    )


def test_graph():
    code = "%0 = (); %1 = 1; (%0, %0, %1)"
    assert parses_as(
        code,
        relay.Tuple([UNIT, UNIT, relay.const(1)])
    )
    assert not parses_as(
        code,
        relay.Tuple([relay.Tuple([]), relay.Tuple([]), relay.const(1)])
    )


@raises_parse_error
def test_graph_wrong_order():
    parse_text("%1 = (); %1")


@raises_parse_error
def test_let_global_var():
    parse_text("let @x = 1; ()")


@raises_parse_error
def test_let_op():
    parse_text("let x = 1; ()")


def test_tuple():
    assert parses_as("()", relay.Tuple([]))

    assert parses_as("(0,)", relay.Tuple([relay.const(0)]))

    assert parses_as("(0, 1)", relay.Tuple([relay.const(0), relay.const(1)]))

    assert parses_as("(0, 1, 2)", relay.Tuple([relay.const(0), relay.const(1), relay.const(2)]))


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
def test_defn():
    id_defn = parse_text(
        """
        def @id(%x: int32) -> int32 {
            %x
        }
        """)
    assert isinstance(id_defn, relay.Module)


def test_recursive_call():
    id_defn = parse_text(
        """
        def @id(%x: int32) -> int32 {
            @id(%x)
        }
        """)
    assert isinstance(id_defn, relay.Module)


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
def test_ifelse_scope():
    parse_text(
        """
        if (True) {
            let %x = ();
            ()
        } else {
            %x
        }
        """
    )


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


def test_incomplete_type():
    assert parses_as(
        "let %_ : _ = (); ()",
        relay.Let(
            _,
            UNIT,
            UNIT
        )
    )


def test_builtin_types():
    for builtin_type in TYPES:
        parse_text("let %_ : {} = (); ()".format(builtin_type))


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
        "let %_ : Tensor[(1), float32] = (); ()",
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


def test_adt_defn():
    mod = relay.Module()

    glob_typ_var = relay.GlobalTypeVar("Ayy")
    prog = relay.TypeData(
            glob_typ_var,
            [],
            [relay.Constructor("Nil", [], glob_typ_var)])
    mod[glob_typ_var] = prog
    assert parses_as(
        """
        type Ayy { Nil }
        """,
        mod
    )


def test_empty_adt_defn():
    mod = relay.Module()

    glob_typ_var = relay.GlobalTypeVar("Ayy")
    prog = relay.TypeData(glob_typ_var, [], [])
    mod[glob_typ_var] = prog
    assert parses_as(
        """
        type Ayy { }
        """,
        mod
    )


def test_multiple_cons_defn():
    mod = relay.Module()

    list_var = relay.GlobalTypeVar("List")
    typ_var = relay.TypeVar("A")
    prog = relay.TypeData(
            list_var,
            [typ_var],
            [
                relay.Constructor("Cons", [typ_var, list_var(typ_var)], list_var),
                relay.Constructor("Nil", [], list_var),
            ])
    mod[list_var] = prog
    assert parses_as(LIST_DEFN, mod)


def test_multiple_type_param_defn():
    glob_typ_var = relay.GlobalTypeVar("Either")
    typ_var_a = relay.TypeVar("A")
    typ_var_b = relay.TypeVar("B")
    prog = relay.TypeData(
            glob_typ_var,
            [typ_var_a, typ_var_b],
            [
                relay.Constructor("Left", [typ_var_a], glob_typ_var),
                relay.Constructor("Right", [typ_var_b], glob_typ_var),
            ])
    mod = relay.Module()
    mod[glob_typ_var] = prog
    assert parses_as(
        """
        type Either[A, B] {
          Left(A),
          Right(B),
        }
        """,
        mod
    )


def test_match():
    # pair each match keyword with whether it specifies a complete match or not
    match_keywords = [("match", True), ("match?", False)]
    for (match_keyword, is_complete) in match_keywords:
        mod = relay.Module()

        list_var = relay.GlobalTypeVar("List")
        typ_var = relay.TypeVar("A")
        cons_constructor = relay.Constructor(
            "Cons", [typ_var, list_var(typ_var)], list_var)
        nil_constructor = relay.Constructor("Nil", [], list_var)
        list_def = relay.TypeData(
            list_var,
            [typ_var],
            [cons_constructor, nil_constructor])
        mod[list_var] = list_def

        length_var = relay.GlobalVar("length")
        typ_var = relay.TypeVar("A")
        input_type = list_var(typ_var)
        input_var = relay.Var("xs", input_type)
        rest_var = relay.Var("rest")
        cons_case = relay.Let(
            _,
            UNIT,
            relay.add(relay.const(1), relay.Call(length_var, [rest_var])))
        body = relay.Match(input_var,
            [relay.Clause(
                relay.PatternConstructor(
                    cons_constructor,
                    [relay.PatternWildcard(), relay.PatternVar(rest_var)]),
                cons_case),
            relay.Clause(
                relay.PatternConstructor(nil_constructor, []),
                relay.const(0))],
            complete=is_complete
        )
        length_func = relay.Function(
            [input_var],
            body,
            int32,
            [typ_var]
        )
        mod[length_var] = length_func

        assert parses_as(
            """
            %s

            def @length[A](%%xs: List[A]) -> int32 {
              %s (%%xs) {
                Cons(_, %%rest) => {
                  ();;
                  1 + @length(%%rest)
                },
                Nil => 0,
              }
            }
            """ % (LIST_DEFN, match_keyword),
            mod
        )


def test_adt_cons_expr():
    mod = relay.Module()

    list_var = relay.GlobalTypeVar("List")
    typ_var = relay.TypeVar("A")
    cons_constructor = relay.Constructor(
        "Cons", [typ_var, list_var(typ_var)], list_var)
    nil_constructor = relay.Constructor("Nil", [], list_var)
    list_def = relay.TypeData(
        list_var,
        [typ_var],
        [cons_constructor, nil_constructor])
    mod[list_var] = list_def

    make_singleton_var = relay.GlobalVar("make_singleton")
    input_var = relay.Var("x", int32)
    make_singleton_func = relay.Function(
        [input_var],
        cons_constructor(input_var, nil_constructor()),
        list_var(int32)
    )
    mod[make_singleton_var] = make_singleton_func

    assert parses_as(
        """
        %s

        def @make_singleton(%%x: int32) -> List[int32] {
          Cons(%%x, Nil)
        }
        """ % LIST_DEFN,
        mod
    )


@raises_parse_error
def test_duplicate_adt_defn():
    parse_text(
        """
        %s

        type List[A] {
          Cons(A, List[A]),
          Nil,
        }
        """ % LIST_DEFN
    )


@raises_parse_error
def test_duplicate_adt_cons():
    parse_text(
        """
        type Ayy { Lmao }
        type Haha { Lmao }
        """
    )


@raises_parse_error
def test_duplicate_adt_cons_defn():
    parse_text(
        """
        type Ayy { Lmao }
        type Lmao { Ayy }
        """
    )


@raises_parse_error
def test_duplicate_global_var():
    parse_text(
        """
        def @id[A](%x: A) -> A { x }
        def @id[A](%x: A) -> A { x }
        """
    )


def test_extern_adt_defn():
    # TODO(weberlo): update this test once extern is implemented
    mod = relay.Module()

    extern_var = relay.GlobalTypeVar("T")
    typ_var = relay.TypeVar("A")
    extern_def = relay.TypeData(extern_var, [typ_var], [])
    mod[extern_var] = extern_def

    assert parses_as(
        """
        extern type T[A]
        """,
        mod
    )


if __name__ == "__main__":
    test_comments()
    test_int_literal()
    test_float_literal()
    test_bool_literal()
    test_negative()
    test_bin_op()
    test_parens()
    test_op_assoc()
    test_let()
    test_seq()
    test_graph()
    test_tuple()
    test_func()
    test_defn()
    test_recursive_call()
    test_ifelse()
    test_call()
    test_incomplete_type()
    test_builtin_types()
    test_tensor_type()
    test_function_type()
    test_tuple_type()
    test_adt_defn()
    test_empty_adt_defn()
    test_multiple_cons_defn()
    test_multiple_type_param_defn()
    test_match()
    test_adt_cons_expr()
    test_duplicate_adt_defn()
    test_duplicate_adt_cons()
    test_duplicate_adt_cons_defn()
    test_duplicate_global_var()
    test_extern_adt_defn()
