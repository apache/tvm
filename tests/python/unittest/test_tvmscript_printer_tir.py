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
import pytest

from tvm.ir import GlobalVar, PointerType, PrimType, Range, TupleType
from tvm.script.printer import script
from tvm.tir import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    Add,
    And,
    Broadcast,
    BufferLoad,
    BufferRegion,
    Call,
    Cast,
    CommReducer,
    Div,
    FloatImm,
    FloorDiv,
    FloorMod,
    IntImm,
    IterVar,
    Let,
    Mul,
    Not,
    Or,
    Ramp,
    Reduce,
    Select,
    Shuffle,
    SizeVar,
    StringImm,
    Sub,
    Var,
    decl_buffer,
)


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        # no actual content
        return "\n"

    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    cleaned_lines = "\n".join(line[spaces_to_remove:] for line in s.splitlines())
    if not cleaned_lines.endswith("\n"):
        cleaned_lines += "\n"
    return cleaned_lines


def as_tir_script(node):
    return script(node, "tir", {"tir": "T"})


@pytest.mark.parametrize(
    "ty, expected",
    [
        pytest.param(
            PrimType("int8"),
            """
            T.int8
            """,
            id="int",
        ),
        pytest.param(
            PrimType("float32"),
            """
            T.float32
            """,
            id="float",
        ),
        pytest.param(
            PointerType(PrimType("int32")),
            """
            T.Ptr(T.int32)
            """,
            id="pointer",
        ),
        pytest.param(
            PointerType(PrimType("int32"), "global"),
            """
            T.Ptr(T.int32, "global")
            """,
            id="with_scope",
        ),
        pytest.param(
            TupleType([]),
            """
            None
            """,
            id="none",
        ),
    ],
)
def test_type(ty, expected):
    assert format_script(expected) == as_tir_script(ty)


@pytest.mark.parametrize("var_type", [Var, SizeVar])
def test_var(var_type):
    var = var_type("x", "int8")

    assert as_tir_script(var) == format_script(
        """
        x: T.int8
        x
        """
    )


@pytest.mark.parametrize(
    "buffer, expected",
    [
        pytest.param(
            decl_buffer((5, 10), name="b"),
            """
            b: T.Buffer(shape=(5, 10))
            b
            """,
            id="simple",
        ),
        pytest.param(
            decl_buffer((5), name=""),
            """
            buf: T.Buffer(shape=(5,))
            buf
            """,
            id="no_name",
        ),
        pytest.param(
            decl_buffer((SizeVar("m", "int"), SizeVar("n", "int")), dtype="int8"),
            """
            m: T.int32
            n: T.int32
            buffer: T.Buffer("int8", shape=(m, n))
            buffer
            """,
            id="symbolic_shape",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="int8",
                data=Var("p", PointerType(PrimType("int16"), "local")),
                strides=[2, 5],
                elem_offset=2,
                data_alignment=16,
                offset_factor=2,
                scope="local",
            ),
            """
            p: T.Ptr(T.int16, "local")
            buffer: T.Buffer("int8", shape=(4, 10), data=p, strides=(2, 5), elem_offset=2, scope="local", align=16, offset_factor=2)
            buffer
            """,
            id="all_param",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="bool",
            ),
            """
            buffer: T.Buffer("bool", shape=(4, 10))
            buffer
            """,
            id="bool_different_ptr_type",
        ),
    ],
)
def test_buffer(buffer, expected):
    assert as_tir_script(buffer) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            StringImm("test"),
            """
            "test"
            """,
            id="string",
        ),
        pytest.param(
            StringImm(""),
            """
            ""
            """,
            id="empty",
        ),
        pytest.param(
            StringImm("test1\ntest2\n"),
            r"""
            "test1\ntest2\n"
            """,
            id="multiline",
        ),
    ],
)
def test_string_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            IntImm("int32", 1),
            """
            1
            """,
            id="default-dtype",
        ),
        pytest.param(
            IntImm("int8", 0),
            """
            T.int8(0)
            """,
            id="int8",
        ),
        pytest.param(
            IntImm("int64", -1),
            """
            T.int64(-1)
            """,
            id="int64",
        ),
        pytest.param(
            IntImm("bool", 1),
            """
            True
            """,
            id="boolean-true",
        ),
        pytest.param(
            IntImm("bool", 0),
            """
            False
            """,
            id="boolean-true",
        ),
    ],
)
def test_int_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            FloatImm("float32", 1.5),
            """
            T.float32(1.5)
            """,
            id="f32",
        ),
        pytest.param(
            FloatImm("float16", 1.5),
            """
            T.float16(1.5)
            """,
            id="f16",
        ),
    ],
)
def test_float_imm(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Cast("float32", Var("x", dtype="int32")),
            """
            x: T.int32
            T.cast(x, "float32")
            """,
            id="with-var",
        ),
        pytest.param(
            Cast("int8", IntImm("int16", 1)),
            """
            T.cast(T.int16(1), "int8")
            """,
            id="with-var",
        ),
    ],
)
def test_cast(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Add(1, 0),
            """
            1 + 0
            """,
            id="add",
        ),
        pytest.param(
            Sub(1, 0),
            """
            1 - 0
            """,
            id="sub",
        ),
        pytest.param(
            Mul(-2.5, 1.5),
            """
            T.float32(-2.5) * T.float32(1.5)
            """,
            id="mul",
        ),
        pytest.param(
            Div(5, 2),
            """
            5 / 2
            """,
            id="div",
        ),
        pytest.param(
            FloorDiv(3, 2),
            """
            3 // 2
            """,
            id="floor-div",
        ),
        pytest.param(
            FloorMod(IntImm("int8", 5), IntImm("int8", 2)),
            """
            T.int8(5) % T.int8(2)
            """,
            id="floor-mod",
        ),
        pytest.param(
            LT(0, 1),
            """
            0 < 1
            """,
            id="lt",
        ),
        pytest.param(
            LE(1.0, 5.0),
            """
            T.float32(1) <= T.float32(5)
            """,
            id="le",
        ),
        pytest.param(
            GT(1, 0),
            """
            1 > 0
            """,
            id="gt",
        ),
        pytest.param(
            GE(Var("n", "int32"), 0),
            """
            n: T.int32
            n >= 0
            """,
            id="ge",
        ),
        pytest.param(
            EQ(Var("n", "int32"), Var("m", "int32")),
            """
            n: T.int32
            m: T.int32
            n == m
            """,
            id="eq",
        ),
        pytest.param(
            NE(1, 0),
            """
            1 != 0
            """,
            id="ne",
        ),
        pytest.param(
            And(IntImm("bool", 1), IntImm("bool", 0)),
            """
            True and False
            """,
            id="and",
        ),
        pytest.param(
            Or(IntImm("bool", 1), IntImm("bool", 0)),
            """
            True or False
            """,
            id="or",
        ),
    ],
)
def test_binary_op(node, expected):
    assert as_tir_script(node) == format_script(expected)


def test_not():
    assert as_tir_script(Not(IntImm("bool", 1))) == format_script(
        """
        not True
        """
    )


def test_select():
    node = Select(IntImm("bool", 1), 0, 1)
    assert as_tir_script(node) == format_script(
        """
        T.Select(True, 0, 1)
        """
    )


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            BufferLoad(decl_buffer((5, 10), name="b"), [0, 1]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0, 1]
            """,
            id="normal",
        ),
        pytest.param(
            BufferLoad(
                decl_buffer((5,), name="b"),
                [
                    0,
                ],
            ),
            """
            b: T.Buffer(shape=(5,))
            b[0]
            """,
            id="1d",
        ),
        pytest.param(
            BufferLoad(decl_buffer((), name="b"), []),
            """
            b: T.Buffer(shape=())
            b[()]
            """,
            id="0d",
        ),
    ],
)
def test_buffer_load(node, expected):
    assert as_tir_script(node) == format_script(expected)


def test_ramp():
    node = Ramp(0, 1, 8)
    expected = """
    T.ramp(0, 1, 8)
    """
    assert as_tir_script(node) == format_script(expected)


def test_broadcast():
    node = Broadcast(0, 4)
    expected = """
    T.broadcast(0, 4)
    """
    assert as_tir_script(node) == format_script(expected)


def test_let():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = Let(x, 1, x + y)
    # Not var definition for x because x isn't free variable in this expression
    expected = """
    y: T.int32
    T.let(x, 1, x + y)
    """
    assert as_tir_script(node) == format_script(expected)


def test_call_tir_op():
    node = Call("float64", "tir.exp", [0.0])
    expected = """
    T.exp(T.float32(0))
    """
    assert as_tir_script(node) == format_script(expected)


def test_call_global_var():
    f_var = GlobalVar("test_f")
    node = Call("float32", f_var, [0, 1])
    expected = """
    test_f(0, 1)
    """
    assert as_tir_script(node) == format_script(expected)


def test_shuffle():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = Shuffle([x, 1, 10], [0, 1, y])
    expected = """
    x: T.int32
    y: T.int32
    T.shuffle([x, 1, 10], [0, 1, y])
    """
    assert as_tir_script(node) == format_script(expected)


def test_comm_reducer_single_value():
    x = Var("x", "int32")
    y = Var("y", "int32")
    node = CommReducer([x], [y], [x + y], [0])
    expected = """
    T.comm_reducer(lambda x, y: x + y, [0])
    """
    assert as_tir_script(node) == format_script(expected)


def test_comm_reducer_multi_value():
    x0 = Var("x0", "int32")
    x1 = Var("x1", "int32")
    y0 = Var("y0", "int32")
    y1 = Var("y1", "int32")
    node = CommReducer([x0, x1], [y0, y1], [x0 + y0, x1 * y1], [0, 1])
    expected = """
    T.comm_reducer(lambda x0, x1, y0, y1: (x0 + y0, x1 * y1), [0, 1])
    """
    assert as_tir_script(node) == format_script(expected)


def test_reduce():
    x = Var("x", "int32")
    y = Var("y", "int32")
    m = Var("m", "int32")
    comm_reducer = CommReducer([x], [y], [x + y], [0])
    node = Reduce(comm_reducer, [m], [IterVar(None, Var("i", "int32"), 2)], True, 0)
    expected = """
    i: T.int32
    m: T.int32
    T.reduce(T.comm_reducer(lambda x, y: x + y, [0]), [m], [T.iter_var(i, None, "CommReduce", "")], 0)
    """
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            Range(0, 1),
            """
            _[0:1]
            """,
            id="normal",
        ),
        pytest.param(
            Range(10),
            """
            _[0:10]
            """,
            id="one-arg",
        ),
    ],
)
def test_range(node, expected):
    assert as_tir_script(node) == format_script(expected)


@pytest.mark.parametrize(
    "node, expected",
    [
        pytest.param(
            BufferRegion(decl_buffer((5, 10), name="b"), [Range(0, 4), Range(0, 9)]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0:4, 0:9]
            """,
            id="normal",
        ),
        pytest.param(
            BufferRegion(decl_buffer((5, 10), name="b"), [Range(0, 1), Range(5, 9)]),
            """
            b: T.Buffer(shape=(5, 10))
            b[0, 5:9]
            """,
            id="scalar-range",
        ),
        pytest.param(
            BufferRegion(decl_buffer((5,), name="b"), [Range(0, 3)]),
            """
            b: T.Buffer(shape=(5,))
            b[0:3]
            """,
            id="1d",
        ),
        pytest.param(
            BufferRegion(decl_buffer((), name="b"), []),
            """
            b: T.Buffer(shape=())
            b[()]
            """,
            id="0d",
        ),
    ],
)
def test_buffer_region(node, expected):
    assert as_tir_script(node) == format_script(expected)
