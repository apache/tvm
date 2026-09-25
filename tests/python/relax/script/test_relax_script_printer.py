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
# ruff: noqa: E501, F841
"""Relax script printer."""

from __future__ import annotations

import pytest
from tvm_ffi.access_path import AccessPath

import tvm
import tvm.testing
from tvm import IRModule, relax, tirx
from tvm.ir import Range
from tvm.relax import TensorType
from tvm.relax.distributed import DeviceMesh, DTensorType, Placement
from tvm.runtime.script_printer import PrinterConfig, _script
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def _assert_print(obj, expected):
    if not isinstance(obj, str):
        obj = obj.script(verbose_expr=True)
    obj = obj.strip()
    assert obj == expected.strip(), "\n" + obj


def test_constant():
    constant = R.dist.const(
        1,
        ty=R.DTensor((), "float32", device_mesh=DeviceMesh((2, 2), Range(0, 4)), placement="R, R"),
    )
    assert (
        constant.__str__()
        == """R.dist.const(1.0, R.DTensor((), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "R, R"))"""
    )


def test_dtensor_type():
    tensor_ty1 = TensorType((32, 32), "float32")
    tensor_ty2 = TensorType((32, 32), None)
    obj0 = DTensorType(tensor_ty1, DeviceMesh((2, 2), Range(0, 4)), Placement.from_text("S[1], R"))
    assert (
        obj0.__str__()
        == """R.DTensor((32, 32), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[1], R")"""
    )

    obj1 = DTensorType(tensor_ty2, DeviceMesh((2, 2), Range(0, 4)), Placement.from_text("S[1], R"))
    assert (
        obj1.__str__()
        == """R.DTensor((32, 32), device_mesh=R.device_mesh((2, 2), R.Range(0, 4)), placement="S[1], R")"""
    )

    obj2 = DTensorType(tensor_ty2, DeviceMesh((2, 2), [0, 1, 2, 3]), Placement.from_text("S[1], R"))
    assert (
        obj2.__str__()
        == """R.DTensor((32, 32), device_mesh=R.device_mesh((2, 2), [0, 1, 2, 3]), placement="S[1], R")"""
    )


@I.ir_module
class TestModule:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2, 2), I.Range(0, 4)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @Ts.prim_func
    def tir_func(
        x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                y[vi, vj] = x[vi, vj] + 1.0

    @R.function
    def foo(
        x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
    ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
        gv0 = R.dist.call_tir(
            TestModule.tir_func,
            x,
            R.DTensor(
                shape=(128, 128), dtype="float32", device_mesh="mesh[0]", placement="S[0], R"
            ),
        )
        return gv0


def test_func():
    _assert_print(
        TestModule["foo"],
        """
# from tvm.script import relax as R

@R.function
def foo(x: R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R")) -> R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R"):
    gv0 = R.dist.call_tir(tir_func, (x,), out_ty=R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R"))
    return gv0
            """,
    )


def test_module():
    _assert_print(
        TestModule,
        """
from __future__ import annotations

# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"device_num": 10})
    I.module_global_infos({"mesh": [R.device_mesh((2, 2), I.Range(0, 4)), R.device_mesh((1,), I.Range(4, 5))]})
    @Ts.prim_func
    def tir_func(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), y: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock(""):
                v, v_1 = Ts.axis.remap("SS", [i, j])
                Ts.reads(x[v, v_1])
                Ts.writes(y[v, v_1])
                y[v, v_1] = x[v, v_1] + T.float32(1.0)

    @R.function
    def foo(x: R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R")) -> R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R"):
        cls = Module
        gv0 = R.dist.call_tir(cls.tir_func, (x,), out_ty=R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R"))
        return gv0
    """,
    )


def _assert_print_lines(obj, expected):
    if not isinstance(obj, str):
        obj = obj.script(verbose_expr=True)
    obj = obj.strip()
    # compare line by line in case there is trailing whitespace in the _middle_
    for obj_line, expected_line in zip(obj.splitlines(), expected.strip().splitlines()):
        assert obj_line.strip() == expected_line.strip(), "\n" + obj


def test_function():
    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):  # type: ignore
        R.func_attr({"some_attr": 1})
        return a

    _assert_print_lines(
        func,
        """
# from tvm.script import relax as R

@R.function
def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
    R.func_attr({"some_attr": 1})
    return a""",
    )


def test_function_dependent_shape_source_spans():
    n = tirx.Var("n", "int64")
    cast = tirx.Cast("int64", n)
    x = relax.Var("x", relax.TensorType([cast], "float32"))
    ret_ty = relax.TensorType(dtype="float32", ndim=1)
    func = relax.Function([x, n], x, ret_ty=ret_ty).with_attr("global_symbol", "main")
    cast_path = (
        AccessPath.root()
        .attr("params")
        .array_item(0)
        .attr("ty")
        .attr("shape")
        .attr("values")
        .array_item(0)
    )

    def render(path):
        config = PrinterConfig(
            verbose_expr=True,
            num_context_lines=0,
            path_to_underline=[path],
            extra_config={"render_invisible_path_info": True},
        )
        first = _script(func, config)
        assert first.count("Access path:") == 1
        lines = first.splitlines()
        definition_index = next(i for i, line in enumerate(lines) if "def main" in line)
        assert "Access path:" not in lines[definition_index]
        return lines[definition_index], lines[definition_index + 1]

    expression = 'T.Cast("int64", n)'
    definition, underline = render(cast_path)
    expression_start = definition.index(expression)
    assert underline[expression_start : expression_start + len(expression)] == "^" * len(expression)
    assert underline.strip() == "^" * len(expression)

    dtype_literal = '"int64"'
    definition, underline = render(cast_path.attr("dtype"))
    dtype_start = definition.index(dtype_literal)
    assert underline[dtype_start : dtype_start + len(dtype_literal)] == "^" * len(dtype_literal)
    assert underline.strip() == "^" * len(dtype_literal)

    definition, underline = render(cast_path.attr("value"))
    variable_start = definition.index(expression) + expression.rindex("n")
    assert underline[variable_start] == "^"
    assert underline.strip() == "^"


def test_lone_private_function():
    @R.function(private=True)
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):  # type: ignore
        R.func_attr({"some_attr": 1})
        return a

    # name prints as main because without a global symbol, the printer cannot assume a name
    _assert_print_lines(
        func,
        """
# from tvm.script import relax as R

@R.function(private=True)
def main(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
    R.func_attr({"some_attr": 1})
    return a""",
    )


def test_extern_func():
    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):  # type: ignore
        return a

    obj = IRModule(
        {
            "func": func,
            "my_ext": relax.ExternFunc("my_ext"),
        }
    )
    _assert_print_lines(
        obj,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    my_ext = R.ExternFunc("my_ext")
    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
        return a
""",
    )


def test_extern_func_with_ty():
    obj = IRModule(
        {
            "my_ext": relax.ExternFunc(
                "my_ext",
                relax.FuncType([], relax.TensorType(dtype="float32", ndim=2), purity=True),
            ),
        }
    )
    _assert_print_lines(
        obj,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    my_ext = R.ExternFunc("my_ext", R.Callable((), R.Tensor(dtype="float32", ndim=2), True))
""",
    )


def test_extern_func_with_ty_roundtrip():
    mod = IRModule(
        {
            "my_ext": relax.ExternFunc(
                "my_ext",
                relax.FuncType([], relax.TensorType(dtype="float32", ndim=2), purity=True),
            ),
        }
    )
    roundtrip = tvm.script.from_source(
        mod.script(verbose_expr=True),
        extra_vars={
            "I": tvm.script.ir,
            "R": tvm.script.relax,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
        },
    )
    tvm.ir.assert_structural_equal(mod, roundtrip)


def test_nested_function():
    @I.ir_module
    class NestedFunction:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            @R.function
            def nested(y: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                return y

            z = nested(x)
            return z

    _assert_print_lines(
        NestedFunction,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        # from tvm.script import relax as R

        @R.function
        def nested(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            return y

        z: R.Tensor((), dtype="int32") = nested(x)
        return z
""",
    )


def test_object_ty():
    obj = relax.AnyType()
    _assert_print_lines(
        obj,
        "R.Any",
    )


def test_shape_ty_0():
    obj = relax.ShapeType(ndim=-1)
    _assert_print_lines(obj, "R.Shape(ndim=-1)")


def test_shape_ty_1():
    obj = relax.ShapeType([1, 2, 3])
    _assert_print_lines(obj, "R.Shape([1, 2, 3])")


def test_shape_ty_2():
    obj = relax.ShapeType([1, tirx.Var("a", "int64"), 3])
    _assert_print_lines(
        obj,
        """
a = I.dynamic("a", dtype="int64")
R.Shape([1, a, 3])""",
    )


def test_tensor_ty():
    obj = relax.TensorType(
        shape=relax.ShapeExpr([1, tirx.Var("a", "int64"), 3]),
        dtype="float32",
    )
    _assert_print_lines(
        obj,
        """
a = I.dynamic("a", dtype="int64")
R.Tensor((1, a, 3), dtype="float32")
""",
    )


def test_tuple_ty_empty():
    obj = relax.TupleType([])
    _assert_print_lines(obj._relax_script(), "R.Tuple")  # pylint: disable=protected-access


def test_tuple_ty():
    obj = relax.TupleType(
        [
            tvm.ir.PrimType("float32"),
            relax.AnyType(),
            relax.ShapeType([1, tirx.Var("a", "int64"), 3]),
        ]
    )
    _assert_print_lines(
        obj._relax_script(),  # pylint: disable=protected-access
        """
R.Tuple(T.float32, R.Any, R.Shape([1, a, 3]))
""",
    )


def test_func_ty():
    obj = relax.FuncType(
        params=[
            tvm.ir.PrimType("float32"),
            relax.AnyType(),
            relax.ShapeType([1, tirx.Var("a", "int64"), 3]),
            tvm.ir.PrimType("int64"),
        ],
        ret=relax.TensorType(
            shape=relax.ShapeExpr([1, 2, 3]),
            dtype="float32",
        ),
    )
    _assert_print_lines(
        obj,
        'a = I.dynamic("a", dtype="int64")\n'
        "R.Callable((T.float32, R.Any, R.Shape([1, a, 3]), T.int64), "
        'R.Tensor((1, 2, 3), dtype="float32"), True)',
    )


def test_shape_type():
    obj = relax.ShapeType(ndim=3)
    _assert_print_lines(obj, "R.Shape(ndim=3)")


def test_object_type():
    obj = relax.AnyType()
    _assert_print_lines(obj, "R.Any")


def test_dyn_tensor_type():
    obj = relax.TensorType()
    _assert_print_lines(obj, 'R.Tensor(dtype="float32")')


def test_packed_func_type():
    obj = relax.PackedFuncType()
    _assert_print_lines(obj, "R.PackedFunc")


def test_tuple_type():
    obj = relax.TupleType([relax.ShapeType(ndim=3), relax.AnyType()])
    _assert_print_lines(
        obj._relax_script(),  # pylint: disable=protected-access
        "R.Tuple(R.Shape(ndim=3), R.Any)",
    )


def test_func_type():
    obj = relax.FuncType(
        params=[
            relax.AnyType(),
            relax.ShapeType(ndim=3),
        ],
        ret=relax.TensorType(
            ndim=3,
            dtype="float32",
        ),
    )
    _assert_print_lines(
        obj._relax_script(),  # pylint: disable=protected-access
        'R.Callable((R.Any, R.Shape(ndim=3)), R.Tensor(dtype="float32", ndim=3), True)',
    )


def test_prim_value():
    @R.function
    def func() -> T.int64:
        return R.prim_value(1)

    _assert_print_lines(
        func,
        """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import relax as R

@R.function
def func() -> T.int64:
    return 1""",
    )

    @R.function
    def float_func() -> T.float32:
        return R.prim_value(T.float32(1.0))

    float_script = float_func.script(verbose_expr=True)
    assert "R.prim_value" not in float_script
    assert "return T.float32(" in float_script
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(
            float_script,
            extra_vars={
                "I": tvm.script.ir,
                "R": tvm.script.relax,
                "T": tvm.script.tirx,
                "Ts": tvm.script.s_tir,
            },
        ),
        float_func,
    )


def test_primitive_bindings_roundtrip_without_prim_value_marker():
    @R.function
    def func(n: T.int64) -> T.int64:
        plus_one = n + 1
        alias = R.prim_value(plus_one)
        return alias

    for show_all_ty in [False, True]:
        source = func.script(show_all_ty=show_all_ty)
        assert "R.prim_value" not in source
        assert "plus_one: T.int64" in source
        assert "alias: T.int64" in source
        tvm.ir.assert_structural_equal(
            tvm.script.from_source(
                source,
                extra_vars={
                    "I": tvm.script.ir,
                    "R": tvm.script.relax,
                    "T": tvm.script.tirx,
                    "Ts": tvm.script.s_tir,
                },
            ),
            func,
        )


def test_data_type_imm():
    obj = tvm.ir.GenericConst(tvm.DataType("float32"), tvm.relax.AnyType())
    _assert_print_lines(obj, 'R.dtype("float32")')


def test_var():
    obj = relax.Var("a", relax.TensorType([1, tirx.Var("x", "int64"), 3], "float32"))
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
a""",
    )


def test_dataflow_var():
    obj = relax.DataflowVar("a", relax.TensorType([1, tirx.Var("x", "int64"), 3], "float32"))
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
a""",
    )


def test_tuple():
    obj = relax.Tuple(
        [
            relax.Var("a", relax.TensorType([1, tirx.Var("x", "int64"), 3], "float32")),
            relax.Var("b", relax.TensorType([1, tirx.Var("y", "int64"), 3], "float32")),
            relax.Var("c", relax.TensorType([1, tirx.Var("z", "int64"), 3], "float32")),
        ]
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
y = I.dynamic("y", dtype="int64")
b: R.Tensor((1, y, 3), dtype="float32")
z = I.dynamic("z", dtype="int64")
c: R.Tensor((1, z, 3), dtype="float32")
(a, b, c)
""",
    )


def test_tuple_get_item():
    obj = relax.TupleGetItem(
        relax.Tuple(
            [
                relax.Var("a", relax.TensorType([1, tirx.Var("x", "int64"), 3], "float32")),
                relax.Var("b", relax.TensorType([1, tirx.Var("y", "int64"), 3], "float32")),
                relax.Var("c", relax.TensorType([1, tirx.Var("z", "int64"), 3], "float32")),
            ]
        ),
        0,
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
y = I.dynamic("y", dtype="int64")
b: R.Tensor((1, y, 3), dtype="float32")
z = I.dynamic("z", dtype="int64")
c: R.Tensor((1, z, 3), dtype="float32")
(a, b, c)[0]
""",
    )


def test_shape_expr():
    obj = relax.ShapeExpr([1, 2, 3])
    _assert_print_lines(obj, "R.shape([1, 2, 3])")


def test_call():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3], "float32"))
    o0 = relax.call_tir(relax.GlobalVar("tir_func"), args=(a, x), out_ty=a.ty)
    o1 = relax.call_dps_packed("my_dps_func", args=a, out_ty=a.ty)
    _assert_print_lines(
        o0,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
R.call_tir(tir_func, (a, x), out_ty=R.Tensor((1, x, 3), dtype="float32"))
""",
    )
    _assert_print_lines(
        o1,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
R.call_dps_packed("my_dps_func", (a,), out_ty=R.Tensor((1, x, 3), dtype="float32"))
""",
    )


def test_call_tir_with_grad():
    x = tirx.Var("x", "int64")
    v0 = relax.Var("v0", R.Tensor([54, 96], "float32"))
    v1 = relax.call_tir_with_grad(
        relax.GlobalVar("tir_func"),
        (v0,),
        R.Tensor((54, 96), "float32"),
        te_grad_name="grad_func",
        te_grad_kwargs={"k": 1.0, "x": x},
    )

    _assert_print_lines(
        v1,
        """
v0: R.Tensor((54, 96), dtype="float32")
x = I.dynamic("x", dtype="int64")
R.call_tir_with_grad(tir_func, (v0,), out_ty=R.Tensor((54, 96), dtype="float32"), te_grad_name="grad_func", te_grad_kwargs={"k": 1.0, "x": x})
""",
    )


def test_call_tir_inplace():
    x = relax.Var("x", R.Tensor((32, 32), dtype="int32"))
    y = relax.Var("y", R.Tensor((32, 32), dtype="int32"))
    t = tirx.Var("t", ty="int64")
    call = relax.call_tir_inplace(
        relax.GlobalVar("tir_func"),
        (
            x,
            y,
            t,
        ),
        inplace_indices=[-1, 0],
        out_ty=[R.Tensor((32, 32), dtype="int32"), R.Tensor((32, 32), dtype="int32")],
    )
    _assert_print_lines(
        call,
        """
x: R.Tensor((32, 32), dtype="int32")
y: R.Tensor((32, 32), dtype="int32")
t = I.dynamic("t", dtype="int64")
R.call_tir_inplace(tir_func, (x, y, t), out_ty=[R.Tensor((32, 32), dtype="int32"), R.Tensor((32, 32), dtype="int32")], inplace_indices=[-1, 0])
        """,
    )


def test_seq_expr():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3], "float32"))
    b = relax.DataflowVar("b", relax.TensorType([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorType([1, x, 3], "float32"))

    obj = relax.SeqExpr(
        blocks=[
            relax.DataflowBlock(
                bindings=[
                    relax.VarBinding(b, relax.op.sin(a)),
                    relax.VarBinding(c, relax.op.sin(b)),
                ]
            ),
        ],
        body=c,
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
with R.dataflow():
    b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
    c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
    R.output(c)
c
""",
    )


def test_binding_block():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3], "float32"))
    b = relax.Var("b", relax.TensorType([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorType([1, x, 3], "float32"))
    obj = relax.BindingBlock(
        bindings=[
            relax.VarBinding(b, relax.op.sin(a)),
            relax.VarBinding(c, relax.op.sin(b)),
        ]
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
""",
    )


def test_dataflow_block():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3], "float32"))
    b = relax.DataflowVar("b", relax.TensorType([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorType([1, x, 3], "float32"))
    obj = relax.DataflowBlock(
        bindings=[
            relax.VarBinding(b, relax.op.sin(a)),
            relax.VarBinding(c, relax.op.sin(b)),
        ]
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
with R.dataflow():
    b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
    c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
    R.output(c)
""",
    )


def test_match_cast():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3]))
    b = relax.Var("b", relax.TensorType([1, 5, 3]))
    obj = relax.MatchCast(
        var=b,
        value=a,
        ty=b.ty,
    )
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, 5, 3), dtype="float32") = R.match_cast(a, R.Tensor((1, 5, 3), dtype="float32"))
""",
    )


def test_var_binding():
    x = tirx.Var("x", "int64")
    a = relax.Var("a", relax.TensorType([1, x, 3], "float32"))
    b = relax.Var("b", relax.TensorType([1, x, 3], "float32"))
    obj = relax.VarBinding(b, relax.op.sin(a))
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
""",
    )


def test_if():
    a = relax.Var("a", relax.TensorType([], "bool"))
    b = relax.Var("b", relax.TensorType([1, 2, 3], "float32"))
    c = relax.Var("c", relax.TensorType([1, 2, 3], "float32"))
    obj = relax.If(
        a,
        relax.SeqExpr([], b),
        relax.SeqExpr([], c),
    )
    _assert_print_lines(
        obj,
        """
a: R.Tensor((), dtype="bool")
if a:
    b: R.Tensor((1, 2, 3), dtype="float32")
    b
else:
    c: R.Tensor((1, 2, 3), dtype="float32")
    c
""",
    )


def test_builtin_keywords():
    x = tirx.Var("x", "int64")
    a = relax.Var("R", relax.TensorType([1, x, 3], "float32"))
    b = relax.Var("T", relax.TensorType([1, x, 3], "float32"))
    obj = relax.VarBinding(b, relax.op.sin(a))
    _assert_print_lines(
        obj,
        """
x = I.dynamic("x", dtype="int64")
R_1: R.Tensor((1, x, 3), dtype="float32")
T_1: R.Tensor((1, x, 3), dtype="float32") = R.sin(R_1)
""",
    )


def test_module_cross_func_call():
    @I.ir_module
    class TestModule:
        @Ts.prim_func
        def tir_func(
            x: T.Buffer((T.int64(128),), "float32"), y: T.Buffer((T.int64(128),), "float32")
        ):
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor((128,), "float32")) -> R.Tensor((128,), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, R.Tensor((128,), dtype="float32"))
            return gv0

    # default behavior
    _assert_print_lines(
        TestModule,
        """
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts
# from tvm.script import relax as R

@I.ir_module
class Module:
    @Ts.prim_func
    def tir_func(x: T.Buffer((T.int64(128),), "float32"), y: T.Buffer((T.int64(128),), "float32")):
        T.evaluate(0)

    @R.function
    def foo(x: R.Tensor((128,), dtype="float32")) -> R.Tensor((128,), dtype="float32"):
        cls = Module
        gv0 = R.call_tir(cls.tir_func, (x,), out_ty=R.Tensor((128,), dtype="float32"))
        return gv0
""",
    )

    # empty module alias
    module_str = TestModule.script(module_alias="")
    _assert_print_lines(
        module_str,
        """
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts
# from tvm.script import relax as R

@I.ir_module
class Module:
    @Ts.prim_func
    def tir_func(x: T.Buffer((T.int64(128),), "float32"), y: T.Buffer((T.int64(128),), "float32")):
        T.evaluate(0)

    @R.function
    def foo(x: R.Tensor((128,), dtype="float32")) -> R.Tensor((128,), dtype="float32"):
        gv0 = R.call_tir(Module.tir_func, (x,), out_ty=R.Tensor((128,), dtype="float32"))
        return gv0
""",
    )


def test_assert_op():
    @I.ir_module
    class AssertOpMod:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), x, format="x: {}")
            return x

    _assert_print_lines(
        AssertOpMod,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        R.assert_op(R.const(False, "bool"), x, format=R.str("x: {}"))
        return x
""",
    )


def test_print():
    @I.ir_module
    class PrintMod:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print(x, format="x: {}")
            return x

    _assert_print_lines(
        PrintMod,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(pure=False)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        R.print(x, format=R.str("x: {}"))
        return x
""",
    )


def test_private_function():
    @I.ir_module
    class AddMod:
        @R.function(private=True)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y: R.Tensor((), dtype="int32") = R.add(x, x)
            return y

    _assert_print_lines(
        AddMod,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        y: R.Tensor((), dtype="int32") = R.add(x, x)
        return y
""",
    )


def test_directly_construct_private_funcs():
    # public
    @R.function
    def foo(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        y: R.Tensor((), dtype="int32") = R.add(x, x)
        return y

    # private
    @R.function(private=True)
    def bar(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        y: R.Tensor((), dtype="int32") = R.multiply(x, x)
        return y

    # public but there's another attribute
    @R.function
    def baz(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        R.func_attr({"relax.force_pure": True})
        y: R.Tuple = R.print(format="Hi there!")
        z: R.Tensor((), dtype="int32") = R.add(x, x)
        return z

    # private with an attribute
    @R.function(private=True)
    def quux(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        R.func_attr({"relax.force_pure": True})
        y: R.Tuple = R.print(format="Lol")
        z: R.Tensor((), dtype="int32") = R.multiply(x, x)
        return z

    obj = IRModule(
        {
            "foo": foo,
            "bar": bar,
            "baz": baz,
            "quux": quux,
        }
    )
    _assert_print_lines(
        obj,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function(private=True)
    def bar(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        y: R.Tensor((), dtype="int32") = R.multiply(x, x)
        return y

    @R.function
    def baz(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        R.func_attr({"relax.force_pure": True})
        R.print(format=R.str("Hi there!"))
        z: R.Tensor((), dtype="int32") = R.add(x, x)
        return z

    @R.function
    def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        y: R.Tensor((), dtype="int32") = R.add(x, x)
        return y

    @R.function(private=True)
    def quux(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
        R.func_attr({"relax.force_pure": True})
        R.print(format=R.str("Lol"))
        z: R.Tensor((), dtype="int32") = R.multiply(x, x)
        return z
""",
    )


def test_reused_extern_func():
    """An ExternFunc used in a variable binding should be explicit"""

    @R.function
    def func(x: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        extern_func = R.ExternFunc("extern_func")
        y = R.call_dps_packed(extern_func, (x,), out_ty=R.Tensor((128, 128), dtype="float32"))
        z = R.call_dps_packed(extern_func, (y,), out_ty=R.Tensor((128, 128), dtype="float32"))
        return z

    _assert_print_lines(
        func,
        """
# from tvm.script import relax as R

@R.function
def func(x: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
    extern_func: R.Callable = R.ExternFunc("extern_func")
    y = R.call_dps_packed(extern_func, (x,), out_ty=R.Tensor((128, 128), dtype="float32"))
    z = R.call_dps_packed(extern_func, (y,), out_ty=R.Tensor((128, 128), dtype="float32"))
    return z
                  """,
    )


def test_inline_extern_func():
    """An ExternFunc used in-line may be printed as a string"""

    @R.function
    def func(x: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        y = R.call_dps_packed(
            R.ExternFunc("extern_func"), (x,), out_ty=R.Tensor((128, 128), dtype="float32")
        )
        z = R.call_dps_packed(
            R.ExternFunc("extern_func"), (y,), out_ty=R.Tensor((128, 128), dtype="float32")
        )
        return z

    _assert_print_lines(
        func,
        """
# from tvm.script import relax as R

@R.function
def func(x: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
    y = R.call_dps_packed("extern_func", (x,), out_ty=R.Tensor((128, 128), dtype="float32"))
    z = R.call_dps_packed("extern_func", (y,), out_ty=R.Tensor((128, 128), dtype="float32"))
    return z
                  """,
    )


def test_hide_inferable_ty():
    """Redundant type annotations can be omitted

    When `show_all_ty=False`, TVMScript type annotations that
    provide redundant type can be omitted.
    """

    @R.function
    def func(A: R.Tensor([10, 20], "float32"), B: R.Tensor(ndim=2, dtype="float32")):
        # R.match_cast has the type as an argument, so it can
        # be omitted from the variable annotation.
        B2 = R.match_cast(B, R.Tensor([10, 20], "float32"))

        # Call nodes may have inferable shapes from their arguments.
        C = R.add(A, B2)

        # Trivial bindings can be inferred to have the same struct
        # info as the RHS.
        D = C

        # Here, the type cannot be omitted.  `R.add(D,B)` has
        # type `R.Tensor(ndim=2)`, but the variable has a shape
        # `R.Tensor([10,20])`.  This is compatible, so it is not an
        # error to have this annotation, but it is not inferrable from
        # the RHS.  Therefore, it must still be printed.
        E: R.Tensor([10, 20], "float32") = R.add(D, B)

        # The return type can be inferred from function body, but is
        # still always printed in the TVMScript.  When parsing an
        # IRModule with functions calling each other, the return type
        # of each callee must be available for use in the caller's
        # shape inference.
        return E

    _assert_print_lines(
        func.script(show_all_ty=False),
        """
# from tvm.script import relax as R

@R.function
def func(A: R.Tensor((10, 20), dtype="float32"), B: R.Tensor(dtype="float32", ndim=2)) -> R.Tensor((10, 20), dtype="float32"):
    B2 = R.match_cast(B, R.Tensor((10, 20), dtype="float32"))
    C = R.add(A, B2)
    D = C
    E: R.Tensor((10, 20), dtype="float32") = R.add(D, B)
    return E""",
    )


def relax_extern_func():
    @R.function
    def func(A: R.Tensor([10, 20], "float32")):
        func = R.ExternFunc("dummy_func")

        B: R.Tensor([10, 20], "float32") = R.call_dps_packed(
            func, [A], out_ty=R.Tensor([10, 20], "float32")
        )

        C: R.Tensor(ndim=2, dtype="float32") = R.call_dps_packed(
            func, [B], out_ty=R.Tensor([10, 20], "float32")
        )

        return C

    return func


@pytest.mark.parametrize("show_all_ty", [True, False], ids=["show_all_ty", "hide_inferable_ty"])
def test_extern_func_roundtrip(show_all_ty):
    original = relax_extern_func()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True, show_all_ty=show_all_ty),
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
