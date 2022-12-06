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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relay
import tvm.relay.testing
from numpy import isclose
from typing import Union

SEMVER = '#[version = "0.0.5"]\n'

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


def assert_graph_equal(lhs, rhs):
    tvm.ir.assert_structural_equal(lhs, rhs, map_free_vars=True)


def graph_equal(lhs, rhs):
    return tvm.ir.structural_equal(lhs, rhs, map_free_vars=True)


def roundtrip_expr(expr):
    text = tvm.relay.Expr.astext(expr, show_meta_data=False)
    x = tvm.parser.parse_expr(text)
    assert_graph_equal(x, expr)


# Testing Utilities for expressions.
def roundtrip(expr):
    x = tvm.parser.fromtext(expr.astext())
    assert_graph_equal(x, expr)


def parse_text(code):
    expr = tvm.parser.parse_expr(code)
    roundtrip_expr(expr)
    return expr


def parses_as(code, expr):
    # type: (str, relay.Expr) -> bool
    parsed = parse_text(code)
    result = graph_equal(parsed, expr)
    return result


# Testing Utilities for full modules.
def parse_module(code):
    mod = tvm.parser.parse(SEMVER + code)
    roundtrip(mod)
    return mod


def assert_parses_as(code, expr):
    parsed = parse_text(code)
    assert_graph_equal(parsed, expr)


def assert_parse_module_as(code, mod):
    mod = tvm.relay.transform.InferType()(mod)
    parsed = parse_module(code)
    assert_graph_equal(parsed, mod)


def get_scalar(x):
    # type: (relay.Constant) -> (Union[float, int, bool])
    return x.data.numpy().item()


int32 = relay.scalar_type("int32")

_ = relay.Var("_")
X = relay.Var("x")
Y = relay.Var("y")
X_ANNO = relay.Var("x", int32)
Y_ANNO = relay.Var("y", int32)

UNIT = relay.Tuple([])


def test_comments():
    assert_parses_as(
        """
        // This is a line comment!
        ()
        """,
        UNIT,
    )

    assert_parses_as(
        """
        /* This is a block comment!
            This is still a block comment!
        */
        ()
        """,
        UNIT,
    )

    assert_parses_as(
        """
        /* This is a block comment!
           /*Block comment is recursive!*/
        */
        ()
        """,
        UNIT,
    )


def test_int_literal():
    assert isinstance(parse_text("1"), relay.Constant)
    assert isinstance(parse_text("1").data, tvm.nd.NDArray)

    assert get_scalar(parse_text("1")) == 1
    assert get_scalar(parse_text("10")) == 10
    assert get_scalar(parse_text("0")) == 0
    assert get_scalar(parse_text("-100")) == -100
    assert get_scalar(parse_text("-05")) == -5
    assert get_scalar(parse_text("9223372036854775807")) == 9223372036854775807

    assert get_scalar(parse_text("-42i")) == -42
    assert get_scalar(parse_text("-42i16")) == -42
    assert get_scalar(parse_text("-42i32")) == -42
    assert get_scalar(parse_text("-42i64")) == -42

    assert_parses_as("-42i16", relay.const(-42, "int16"))
    assert_parses_as("-42i32", relay.const(-42, "int32"))
    assert_parses_as("-42i", relay.const(-42, "int32"))
    assert_parses_as("-42", relay.const(-42, "int32"))
    assert_parses_as("-42i64", relay.const(-42, "int64"))
    assert_parses_as("2147483647", relay.const(2147483647, "int32"))
    assert_parses_as("2147483648", relay.const(2147483648, "int64"))

    with pytest.raises(tvm.error.DiagnosticError):
        # Unrepresentable
        parse_text("2147483648i32")
    with pytest.raises(tvm.error.DiagnosticError):
        # Unrepresentable
        parse_text("32768i16")


def test_float_literal():
    assert get_scalar(parse_text("1.0f")) == 1.0
    assert isclose(get_scalar(parse_text("1.56667f")), 1.56667)
    assert get_scalar(parse_text("0.0f")) == 0.0
    assert get_scalar(parse_text("-10.0f")) == -10.0

    # scientific notation
    assert isclose(get_scalar(parse_text("1e-1f")), 1e-1)
    assert get_scalar(parse_text("1e+1f")) == 1e1
    assert isclose(get_scalar(parse_text("1E-1f")), 1e-1)
    assert get_scalar(parse_text("1E+1f")) == 1e1
    assert isclose(get_scalar(parse_text("1.0e-1f")), 1.0e-1)
    assert get_scalar(parse_text("1.0e+1f")) == 1.0e1
    assert isclose(get_scalar(parse_text("1.0E-1f")), 1.0e-1)
    assert get_scalar(parse_text("1.0E+1f")) == 1.0e1

    assert get_scalar(parse_text("3f16")) == 3.0
    assert get_scalar(parse_text("3f32")) == 3.0

    assert_parses_as("3f16", relay.const(3.0, "float16"))
    assert_parses_as("3f32", relay.const(3.0, "float32"))
    assert_parses_as("3f", relay.const(3.0, "float32"))
    assert_parses_as("3f64", relay.const(3.0, "float64"))

    with pytest.raises(tvm.error.DiagnosticError):
        # Unrepresentable
        parse_text("3.40283e+38f32")
    with pytest.raises(tvm.error.DiagnosticError):
        # Unrepresentable
        parse_text("65505f16")


def test_bool_literal():
    assert get_scalar(parse_text("True")) == True
    assert get_scalar(parse_text("False")) == False

    assert_parses_as("True", relay.const(True, "bool"))


def test_negative():
    # need to handle parsing non-literal operations
    # assert isinstance(parse_text("let %x = 1; -%x").body, relay.Call)
    assert get_scalar(parse_text("--10")) == 10
    assert get_scalar(parse_text("---10")) == -10


def test_bin_op():
    for bin_op in BINARY_OPS.keys():
        assert_parses_as(
            "1 {} 1".format(bin_op), BINARY_OPS.get(bin_op)(relay.const(1), relay.const(1))
        )


def test_parens():
    assert graph_equal(parse_text("1 * 1 + 1"), parse_text("(1 * 1) + 1"))
    assert not graph_equal(parse_text("1 * 1 + 1"), parse_text("1 * (1 + 1)"))


def test_op_assoc():
    assert graph_equal(parse_text("1 * 1 + 1 < 1 == 1"), parse_text("(((1 * 1) + 1) < 1) == 1"))
    assert graph_equal(parse_text("1 == 1 < 1 + 1 * 1"), parse_text("1 == (1 < (1 + (1 * 1)))"))


def test_vars():
    # var
    var = parse_text("let %foo = (); %foo")
    assert isinstance(var.body, relay.Var)
    assert var.body.name_hint == "foo"

    # global var
    global_var = parse_text("@foo")
    assert isinstance(global_var, relay.GlobalVar)
    assert global_var.name_hint == "foo"

    # operator id
    op = parse_text("add")
    assert isinstance(op, tvm.ir.Op)
    assert op.name == "add"

    # operator id with prefix
    op = parse_text("nn.global_avg_pool2d")
    assert isinstance(op, tvm.ir.Op)
    assert op.name == "nn.global_avg_pool2d"


def test_meta_ref():
    with pytest.raises(tvm.error.DiagnosticError):
        meta_op = parse_text("meta[type_key][1337]")
        assert meta_op.attrs.node_type_key == "type_key"
        assert meta_op.attrs.node_index == 1337


def test_let():
    assert_parses_as("let %x = 1; ()", relay.Let(X, relay.const(1), UNIT))

    assert_parses_as(
        """
        let %x = 1;
        let %y = 2;
        ()
        """,
        relay.Let(X, relay.const(1), relay.Let(Y, relay.const(2), UNIT)),
    )


def test_seq():
    assert_parses_as("(); ()", relay.Let(_, UNIT, UNIT))

    assert_parses_as("let %_ = 1; ()", relay.Let(X, relay.const(1), UNIT))


def test_graph():
    code = "%0 = (); %1 = 1; (%0, %0, %1)"
    assert_parses_as(code, relay.Tuple([UNIT, UNIT, relay.const(1)]))


def test_graph_single():
    assert_parses_as("%1 = (); %1", relay.Tuple([]))


def test_let_global_var():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text("let @x = 1; ()")


def test_let_op():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text("let x = 1; ()")


def test_tuple():
    assert_parses_as("()", relay.Tuple([]))

    assert_parses_as("(0,)", relay.Tuple([relay.const(0)]))

    assert_parses_as("(0, 1)", relay.Tuple([relay.const(0), relay.const(1)]))

    assert_parses_as("(0, 1, 2)", relay.Tuple([relay.const(0), relay.const(1), relay.const(2)]))


def test_tuple_proj():
    x = relay.var("x", shape=())
    assert_parses_as(
        "free_var %x: float32; %x((%x,).0, %x)",
        relay.Call(x, [relay.TupleGetItem(relay.Tuple([x]), 0), x]),
    )


def test_func():
    # 0 args
    assert_parses_as("fn () { 0 }", relay.Function([], relay.const(0), None, []))

    # 1 arg
    assert_parses_as("fn (%x) { %x }", relay.Function([X], X, None, []))

    # 2 args
    assert_parses_as("fn (%x, %y) { %x + %y }", relay.Function([X, Y], relay.add(X, Y), None, []))

    # annotations
    assert_parses_as("fn (%x: int32) -> int32 { %x }", relay.Function([X_ANNO], X_ANNO, int32, []))

    # Refactor the attribute syntax and printing.
    #
    # # attributes
    # assert_parses_as(
    #     "fn (n=5) { () }",
    #     relay.Function([], UNIT, None, None, tvm.ir.make_node("DictAttrs", n=relay.const(5)))
    # )


# TODO(@jmp): Crashes if %x isn't annnotated.
def test_defn():
    id_defn = parse_module(
        """
        def @id(%x: int32) -> int32 {
            %x
        }
        """
    )
    assert isinstance(id_defn, tvm.IRModule)


def test_recursive_call():
    id_defn = parse_module(
        """
        def @id(%x: int32) -> int32 {
            @id(%x)
        }
        """
    )
    assert isinstance(id_defn, tvm.IRModule)


def test_ifelse():
    assert_parses_as(
        """
        if (True) {
            0
        } else {
            1
        }
        """,
        relay.If(relay.const(True), relay.const(0), relay.const(1)),
    )


def test_ifelse_scope():
    with pytest.raises(tvm.error.DiagnosticError):
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


def test_ref():
    program = """
    #[version = "0.0.5"]
    def @main(%x: float32) {
        %0 = ref(%x);
        ref_write(%0, 1f);
        ref_read(%0)
    }
    """
    tvm.parser.parse(program)


def test_call():
    # select right function to call: simple ident case
    id_func = relay.Var("id")
    assert_parses_as(
        """
        let %id = fn (%x) { %x };
        10 * %id(10)
        """,
        relay.Let(
            id_func,
            relay.Function([X], X, None, []),
            relay.multiply(relay.const(10), relay.Call(id_func, [relay.const(10)])),
        ),
    )

    # 0 args
    constant = relay.Var("constant")
    assert_parses_as(
        """
        let %constant = fn () { 0 };
        %constant()
        """,
        relay.Let(
            constant,
            relay.Function([], relay.const(0), None, []),
            relay.Call(constant, [], None, None),
        ),
    )

    # 1 arg
    id_var = relay.Var("id")
    assert_parses_as(
        """
        let %id = fn (%x) { %x };
        %id(1)
        """,
        relay.Let(
            id_var,
            relay.Function([X], X, None, []),
            relay.Call(id_var, [relay.const(1)], None, None),
        ),
    )

    # 2 args
    multiply = relay.Var("multiply")
    assert_parses_as(
        """
        let %multiply = fn (%x, %y) { %x * %y };
        %multiply(0, 0)
        """,
        relay.Let(
            multiply,
            relay.Function([X, Y], relay.multiply(X, Y), None, []),
            relay.Call(multiply, [relay.const(0), relay.const(0)], None, None),
        ),
    )

    # anonymous function
    assert_parses_as(
        """
        (fn (%x) { %x })(0)
        """,
        relay.Call(relay.Function([X], X, None, []), [relay.const(0)], None, None),
    )

    # curried function
    curried_mult = relay.Var("curried_mult")
    assert_parses_as(
        """
        let %curried_mult =
            fn (%x) {
            fn (%y) {
                %x * %y
            }
            };
            %curried_mult(0);
            %curried_mult(0)(0)
        """,
        relay.Let(
            curried_mult,
            relay.Function([X], relay.Function([Y], relay.multiply(X, Y), None, []), None, []),
            relay.Let(
                _,
                relay.Call(curried_mult, [relay.const(0)], None, None),
                relay.Call(
                    relay.Call(curried_mult, [relay.const(0)], None, None),
                    [relay.const(0)],
                    None,
                    None,
                ),
            ),
        ),
    )

    # op
    assert_parses_as("abs(1)", relay.Call(relay.op.get("abs"), [relay.const(1)], None, None))


# Types


def test_incomplete_type():
    assert_parses_as("let %_ : _ = (); ()", relay.Let(_, UNIT, UNIT))


def test_builtin_types():
    for builtin_type in TYPES:
        parse_text("let %_ : {} = (); ()".format(builtin_type))


def test_tensor_type():
    assert_parses_as(
        "let %_ : Tensor[(), float32] = (); ()",
        relay.Let(relay.Var("_", relay.TensorType((), "float32")), UNIT, UNIT),
    )

    assert_parses_as(
        "let %_ : Tensor[(1), float32] = (); ()",
        relay.Let(relay.Var("_", relay.TensorType((1,), "float32")), UNIT, UNIT),
    )

    assert_parses_as(
        "let %_ : Tensor[(1, 1), float32] = (); ()",
        relay.Let(relay.Var("_", relay.TensorType((1, 1), "float32")), UNIT, UNIT),
    )

    assert_parses_as(
        "let %_ : Tensor[(?, 1), float32] = (); ()",
        relay.Let(relay.Var("_", relay.TensorType((tvm.tir.Any(), 1), "float32")), UNIT, UNIT),
    )


def test_function_type():
    assert_parses_as(
        """
        let %_: fn () -> int32 = fn () -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([], int32, [], [])),
            relay.Function([], relay.const(0), int32, []),
            UNIT,
        ),
    )

    assert_parses_as(
        """
        let %_: fn (int32) -> int32 = fn (%x: int32) -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([int32], int32, [], [])),
            relay.Function([relay.Var("x", int32)], relay.const(0), int32, []),
            UNIT,
        ),
    )

    assert_parses_as(
        """
        let %_: fn (int32, int32) -> int32 = fn (%x: int32, %y: int32) -> int32 { 0 }; ()
        """,
        relay.Let(
            relay.Var("_", relay.FuncType([int32, int32], int32, [], [])),
            relay.Function(
                [relay.Var("x", int32), relay.Var("y", int32)], relay.const(0), int32, []
            ),
            UNIT,
        ),
    )


def test_tuple_type():
    assert_parses_as(
        """
        let %_: () = (); ()
        """,
        relay.Let(relay.Var("_", relay.TupleType([])), UNIT, UNIT),
    )

    assert_parses_as(
        """
        let %_: (int32,) = (0,); ()
        """,
        relay.Let(relay.Var("_", relay.TupleType([int32])), relay.Tuple([relay.const(0)]), UNIT),
    )

    assert_parses_as(
        """
        let %_: (int32, int32) = (0, 1); ()
        """,
        relay.Let(
            relay.Var("_", relay.TupleType([int32, int32])),
            relay.Tuple([relay.const(0), relay.const(1)]),
            UNIT,
        ),
    )


def test_adt_defn():
    mod = tvm.IRModule()

    glob_typ_var = relay.GlobalTypeVar("Ayy")
    prog = relay.TypeData(glob_typ_var, [], [relay.Constructor("Nil", [], glob_typ_var)])
    mod[glob_typ_var] = prog
    assert_parse_module_as(
        """
        type Ayy { Nil }
        """,
        mod,
    )


def test_adt_any():
    code = """
    type my_dtype {
        my_cons(Tensor[(?, 1), uint16]),
    }
    """
    mod = parse_module(code)
    items = mod.type_definitions.items()
    global_type_var, type_data = items[0]
    assert global_type_var.name_hint == "my_dtype"
    ctors = type_data.constructors
    assert len(ctors) == 1
    my_cons = ctors[0]
    assert my_cons.name_hint == "my_cons"
    ty_shape = my_cons.inputs[0].shape
    assert isinstance(ty_shape[0], tvm.tir.Any)
    assert ty_shape[1] == 1


def test_empty_adt_defn():
    mod = tvm.IRModule()

    glob_typ_var = relay.GlobalTypeVar("Ayy")
    prog = relay.TypeData(glob_typ_var, [], [])
    mod[glob_typ_var] = prog
    assert_parse_module_as(
        """
        type Ayy { }
        """,
        mod,
    )


def test_multiple_cons_defn():
    mod = tvm.IRModule()

    list_var = relay.GlobalTypeVar("List")
    typ_var = relay.TypeVar("A")
    prog = relay.TypeData(
        list_var,
        [typ_var],
        [
            relay.Constructor("Cons", [typ_var, list_var(typ_var)], list_var),
            relay.Constructor("Nil", [], list_var),
        ],
    )
    mod[list_var] = prog
    assert_parse_module_as(LIST_DEFN, mod)


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
        ],
    )
    mod = tvm.IRModule()
    mod[glob_typ_var] = prog
    assert_parse_module_as(
        """
        type Either[A, B] {
          Left(A),
          Right(B),
        }
        """,
        mod,
    )


def test_match():
    # pair each match keyword with whether it specifies a complete match or not
    match_keywords = [("match", True), ("match?", False)]
    for (match_keyword, is_complete) in match_keywords:
        mod = tvm.IRModule()

        list_var = relay.GlobalTypeVar("List")
        typ_var = relay.TypeVar("A")
        cons_constructor = relay.Constructor("Cons", [typ_var, list_var(typ_var)], list_var)
        nil_constructor = relay.Constructor("Nil", [], list_var)
        list_def = relay.TypeData(list_var, [typ_var], [cons_constructor, nil_constructor])
        mod[list_var] = list_def

        length_var = relay.GlobalVar("length")
        typ_var = relay.TypeVar("A")
        input_type = list_var(typ_var)
        input_var = relay.Var("xs", input_type)
        rest_var = relay.Var("rest")
        cons_case = relay.Let(
            relay.var("", type_annotation=None),
            UNIT,
            relay.add(relay.const(1), relay.Call(length_var, [rest_var])),
        )
        body = relay.Match(
            input_var,
            [
                relay.Clause(
                    relay.PatternConstructor(
                        cons_constructor, [relay.PatternWildcard(), relay.PatternVar(rest_var)]
                    ),
                    cons_case,
                ),
                relay.Clause(relay.PatternConstructor(nil_constructor, []), relay.const(0)),
            ],
            complete=is_complete,
        )
        length_func = relay.Function([input_var], body, int32, [typ_var])
        mod[length_var] = length_func

        assert_parse_module_as(
            """
            %s

            def @length[A](%%xs: List[A]) -> int32 {
              %s (%%xs) {
                Cons(_, %%rest : List[A]) => {
                  ();
                  1 + @length(%%rest)
                },
                Nil => 0,
              }
            }
            """
            % (LIST_DEFN, match_keyword),
            mod,
        )


def test_adt_cons_expr():
    mod = tvm.IRModule()

    list_var = relay.GlobalTypeVar("List")
    typ_var = relay.TypeVar("A")
    cons_constructor = relay.Constructor("Cons", [typ_var, list_var(typ_var)], list_var)
    nil_constructor = relay.Constructor("Nil", [], list_var)
    list_def = relay.TypeData(list_var, [typ_var], [cons_constructor, nil_constructor])
    mod[list_var] = list_def

    make_singleton_var = relay.GlobalVar("make_singleton")
    input_var = relay.Var("x", int32)
    make_singleton_func = relay.Function(
        [input_var], cons_constructor(input_var, nil_constructor()), list_var(int32)
    )
    mod[make_singleton_var] = make_singleton_func

    assert_parse_module_as(
        """
        %s

        def @make_singleton(%%x: int32) -> List[int32] {
          Cons(%%x, Nil)
        }
        """
        % LIST_DEFN,
        mod,
    )


def test_duplicate_adt_defn():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_module(
            """
            %s

            type List[A] {
            Cons(A, List[A]),
            Nil,
            }
            """
            % LIST_DEFN
        )


def test_duplicate_adt_cons():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text(
            """
            type Ayy { Lmao }
            type Haha { Lmao }
            """
        )


def test_duplicate_adt_cons_defn():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text(
            """
            type Ayy { Lmao }
            type Lmao { Ayy }
            """
        )


def test_duplicate_global_var():
    with pytest.raises(tvm.error.DiagnosticError):
        parse_text(
            """
            def @id[A](%x: A) -> A { x }
            def @id[A](%x: A) -> A { x }
            """
        )


def test_extern_adt_defn():
    mod = tvm.IRModule()

    extern_var = relay.GlobalTypeVar("T")
    typ_var = relay.TypeVar("A")
    extern_def = relay.TypeData(extern_var, [typ_var], [])
    mod[extern_var] = extern_def

    assert_parse_module_as(
        """
        extern type T[A]
        """,
        mod,
    )


def test_import_grad():
    mod = tvm.IRModule()
    mod.import_from_std("gradient.rly")


def test_mlp():
    mod, _ = relay.testing.mlp.get_workload(1)
    text = mod.astext()
    parsed_mod = tvm.parser.parse(text)
    tvm.ir.assert_structural_equal(mod, parsed_mod)


def inline_params(mod, params):
    main_fn = mod["main"]
    str_to_var = {}
    for param in main_fn.params:
        str_to_var[param.name_hint] = param

    bind_map = {}
    for param in params:
        bind_map[str_to_var[param]] = relay.const(params[param])

    body = relay.bind(main_fn.body, bind_map)
    main_fn = relay.Function(relay.analysis.free_vars(body), body)
    mod._add("main", main_fn, True)
    return mod


def test_mlp_inlined_params():
    mod, params = relay.testing.mlp.get_workload(1)
    mod = inline_params(mod, params)
    mod = relay.transform.InferType()(mod)
    text = mod.astext()
    parsed_mod = tvm.parser.parse(text)
    tvm.ir.assert_structural_equal(mod, parsed_mod)


def test_tuple_return_value():
    program = """
    type Box[T] {
        constructor(T)
    }

    def @example() {
        %0 = ();
        %1 = constructor(%0);
        %2 = constructor(0f);
        (%1, %2,)
    }
    """
    parse_module(program)


def test_parse_if_in_binding():
    program = """
    def @example(%b: bool) {
        %0 = if (%b) {
            1
        } else {
            0
        };
        %0
    }
    """
    parse_module(program)


def test_op_string_attr():
    call = parse_text(
        """
        free_var %x: Tensor[(1, 32, 32, 3), float32];
        free_var %y: Tensor[(1, 1, 3, 3), float32];
        nn.conv2d(%x, %y, data_layout="NHWC", kernel_layout="HWIO")
        """
    )

    assert isinstance(call.op, tvm.ir.Op)
    assert call.op.name == "nn.conv2d"
    assert call.attrs.data_layout == "NHWC"
    assert call.attrs.kernel_layout == "HWIO"


def test_load_prelude():
    mod = tvm.IRModule()
    mod.import_from_std("prelude.rly")
    tvm.parser.parse(mod.astext())


def test_call_attrs():
    def get_func(shape, dtype):
        x0 = relay.var("data", shape=shape, dtype=dtype)
        w0 = relay.var("weight", shape=shape, dtype=dtype)
        a = relay.nn.dense(x0, w0)
        b = relay.nn.relu(a)
        d = relay.add(b, relay.const(1.0, dtype=dtype))
        return relay.Function([x0, w0], d)

    # build relay graph
    shape = (2, 4)
    dtype = "float32"
    sub_func = get_func(shape, dtype)
    p0 = relay.var("p0", shape=shape, dtype=dtype)
    p1 = relay.var("p1", shape=shape, dtype=dtype)
    attr = tvm.ir.make_node("attrs.TestAttrs", name="func_call_attrs")
    call = relay.Call(sub_func, [p0, p1], attrs=attr)
    func = relay.Function([p0, p1], call)

    # build relay module
    mod = tvm.IRModule()
    mod["main"] = func
    mod = tvm.relay.transform.InferType()(mod)

    # assert equal
    program = """
    def @main(%p0: Tensor[(2, 4), float32], %p1: Tensor[(2, 4), float32]) {
    %2 = fn (%data: Tensor[(2, 4), float32], %weight: Tensor[(2, 4), float32]) {
        %0 = nn.dense(%data, %weight, units=None);
        %1 = nn.relu(%0);
        add(%1, 1f)
    };
    %2(%p0, %p1, name="func_call_attrs", attrs_type_key="attrs.TestAttrs")
    }
    """
    parsed = parse_module(program)
    assert_graph_equal(parsed, mod)


def test_tokenize_inf():
    x = relay.var("x", shape=(3, 4), dtype="float32")
    y = relay.clip(x, -np.inf, np.inf)

    f = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(f)

    mod = relay.transform.AnnotateSpans()(mod)


def test_func_attrs():
    attrs = tvm.ir.make_node("DictAttrs", **{"Primitive": 1, "relay.reshape_only": 1})
    x = relay.var("x", shape=(2, 3))
    func = relay.Function([x], relay.reshape(x, (-1,)), attrs=attrs)
    assert_parses_as(func.astext(), func)


def test_init_module_and_metatable():
    init_metatable = {"relay.Constant": [relay.const(np.random.rand(2, 3), dtype="float32")]}
    init_module = tvm.parser.fromtext(
        SEMVER
        + """
            def @f(%y : Tensor[(2, 3), float32]) -> Tensor[(2, 3), float32] {
              negative(%y)
            }
        """,
    )
    mod = tvm.parser.parse(
        SEMVER
        + """
            def @main(%x: Tensor[(2, 3), float32]) {
              add(@f(%x), meta[relay.Constant][0])
            }
        """,
        "from_string",
        init_module,
        init_metatable,
    )
    roundtrip(mod)


if __name__ == "__main__":
    tvm.testing.main()
