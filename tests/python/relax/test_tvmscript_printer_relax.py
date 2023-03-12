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
# pylint: disable=missing-docstring
import tvm
import tvm.testing
from tvm import IRModule, relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def _assert_print(obj, expected):
    if not isinstance(obj, str):
        obj = obj.script(verbose_expr=True)
    obj = obj.strip()
    assert obj == expected.strip(), "\n" + obj


def test_function():
    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):  # type: ignore
        return a

    _assert_print(
        func,
        """
# from tvm.script import relax as R

@R.function
def main(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
    return a""",
    )


def test_extern_func():
    @R.function
    def relax_func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):  # type: ignore
        return a

    obj = IRModule(
        {
            "func": relax_func,
            "my_ext": relax.ExternFunc("my_ext"),
        }
    )
    _assert_print(
        obj,
        """
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    "my_ext"
    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
        return a
""",
    )


def test_object_struct_info():
    obj = relax.ObjectStructInfo()
    _assert_print(
        obj,
        "R.Object",
    )


def test_prim_struct_info():
    obj = relax.PrimStructInfo("float32")
    _assert_print(obj, 'R.Prim("float32")')


def test_shape_struct_info_0():
    obj = relax.ShapeStructInfo(ndim=-1)
    _assert_print(obj, "R.Shape(ndim=-1)")


def test_shape_struct_info_1():
    obj = relax.ShapeStructInfo([1, 2, 3])
    _assert_print(obj, "R.Shape([1, 2, 3])")


def test_shape_struct_info_2():
    obj = relax.ShapeStructInfo([1, tir.Var("a", "int64"), 3])
    _assert_print(
        obj,
        """
a = T.int64()
R.Shape([1, a, 3])""",
    )


def test_tensor_struct_info():
    obj = relax.TensorStructInfo(
        shape=relax.ShapeExpr([1, tir.Var("a", "int64"), 3]),
        dtype="float32",
    )
    _assert_print(
        obj,
        """
a = T.int64()
R.Tensor((1, a, 3), dtype="float32")
""",
    )


def test_tuple_struct_info_empty():
    obj = relax.TupleStructInfo([])
    _assert_print(obj, "R.Tuple")


def test_tuple_struct_info():
    obj = relax.TupleStructInfo(
        [
            relax.PrimStructInfo("float32"),
            relax.ObjectStructInfo(),
            relax.ShapeStructInfo([1, tir.Var("a", "int64"), 3]),
        ]
    )
    _assert_print(
        obj,
        """
a = T.int64()
R.Tuple(R.Prim("float32"), R.Object, R.Shape([1, a, 3]))
""",
    )


def test_func_struct_info():
    obj = relax.FuncStructInfo(
        params=[
            relax.PrimStructInfo("float32"),
            relax.ObjectStructInfo(),
            relax.ShapeStructInfo([1, tir.Var("a", "int64"), 3]),
        ],
        ret=relax.TensorStructInfo(
            shape=relax.ShapeExpr([1, 2, 3]),
            dtype="float32",
        ),
    )
    _assert_print(
        obj,
        """
a = T.int64()
R.Callable((R.Prim("float32"), R.Object, R.Shape([1, a, 3])), R.Tensor((1, 2, 3), dtype="float32"))
""",
    )


def test_shape_type():
    obj = relax.ShapeType(ndim=3)
    _assert_print(obj, "R.Shape(ndim=3)")


def test_object_type():
    obj = relax.ObjectType()
    _assert_print(obj, "R.Object")


def test_dyn_tensor_type():
    obj = relax.DynTensorType()
    _assert_print(obj, 'R.Tensor(ndim=-1, dtype="float32")')


def test_packed_func_type():
    obj = relax.PackedFuncType()
    _assert_print(obj, "R.PackedFunc")


def test_tuple_type():
    obj = relax.TupleType([relax.ShapeType(ndim=3), relax.ObjectType()])
    _assert_print(
        obj._relax_script(),  # pylint: disable=protected-access
        "R.Tuple(R.Shape(ndim=3), R.Object)",
    )


def test_func_type():
    obj = relax.FuncType(
        arg_types=[
            relax.ObjectType(),
            relax.ShapeType(ndim=3),
        ],
        ret_type=relax.DynTensorType(
            ndim=3,
            dtype="float32",
        ),
    )
    _assert_print(
        obj._relax_script(),  # pylint: disable=protected-access
        'R.Callable((R.Object, R.Shape(ndim=3)), R.Tensor(ndim=3, dtype="float32"))',
    )


def test_prim_value():
    obj = relax.PrimValue(1)
    _assert_print(obj, "R.prim_value(1)")


def test_string_imm():
    obj = relax.StringImm("hello")
    _assert_print(obj, 'R.str("hello")')


def test_data_type_imm():
    obj = relax.DataTypeImm("float32")
    _assert_print(obj, 'R.dtype("float32")')


def test_var():
    obj = relax.Var("a", relax.TensorStructInfo([1, tir.Var("x", "int64"), 3], "float32"))
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
a""",
    )


def test_dataflow_var():
    obj = relax.DataflowVar("a", relax.TensorStructInfo([1, tir.Var("x", "int64"), 3], "float32"))
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
a""",
    )


def test_tuple():
    obj = relax.Tuple(
        [
            relax.Var("a", relax.TensorStructInfo([1, tir.Var("x", "int64"), 3], "float32")),
            relax.Var("b", relax.TensorStructInfo([1, tir.Var("y", "int64"), 3], "float32")),
            relax.Var("c", relax.TensorStructInfo([1, tir.Var("z", "int64"), 3], "float32")),
        ]
    )
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
y = T.int64()
b: R.Tensor((1, y, 3), dtype="float32")
z = T.int64()
c: R.Tensor((1, z, 3), dtype="float32")
(a, b, c)
""",
    )


def test_tuple_get_item():
    obj = relax.TupleGetItem(
        relax.Tuple(
            [
                relax.Var("a", relax.TensorStructInfo([1, tir.Var("x", "int64"), 3], "float32")),
                relax.Var("b", relax.TensorStructInfo([1, tir.Var("y", "int64"), 3], "float32")),
                relax.Var("c", relax.TensorStructInfo([1, tir.Var("z", "int64"), 3], "float32")),
            ]
        ),
        0,
    )
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
y = T.int64()
b: R.Tensor((1, y, 3), dtype="float32")
z = T.int64()
c: R.Tensor((1, z, 3), dtype="float32")
(a, b, c)[0]
""",
    )


def test_shape_expr():
    obj = relax.ShapeExpr([1, 2, 3])
    _assert_print(obj, "R.shape([1, 2, 3])")


def test_call():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3], "float32"))
    o0 = relax.call_tir(relax.GlobalVar("tir_func"), args=a, out_sinfo=a.struct_info, tir_vars=[x])
    o1 = relax.call_dps_packed("my_dps_func", args=a, out_sinfo=a.struct_info)
    _assert_print(
        o0,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
R.call_tir(tir_func, (a,), out_sinfo=R.Tensor((1, x, 3), dtype="float32"), tir_vars=R.shape([x]))
""",
    )
    _assert_print(
        o1,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
R.call_dps_packed("my_dps_func", (a,), out_sinfo=R.Tensor((1, x, 3), dtype="float32"))
""",
    )


def test_seq_expr():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3], "float32"))
    b = relax.DataflowVar("b", relax.TensorStructInfo([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorStructInfo([1, x, 3], "float32"))

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
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
with R.dataflow():
    b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
    c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
    R.output(c)
c
""",
    )


def test_binding_block():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorStructInfo([1, x, 3], "float32"))
    obj = relax.BindingBlock(
        bindings=[
            relax.VarBinding(b, relax.op.sin(a)),
            relax.VarBinding(c, relax.op.sin(b)),
        ]
    )
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
""",
    )


def test_dataflow_block():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3], "float32"))
    b = relax.DataflowVar("b", relax.TensorStructInfo([1, x, 3], "float32"))
    c = relax.Var("c", relax.TensorStructInfo([1, x, 3], "float32"))
    obj = relax.DataflowBlock(
        bindings=[
            relax.VarBinding(b, relax.op.sin(a)),
            relax.VarBinding(c, relax.op.sin(b)),
        ]
    )
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
with R.dataflow():
    b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
    c: R.Tensor((1, x, 3), dtype="float32") = R.sin(b)
    R.output(c)
""",
    )


def test_match_cast():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3]))
    b = relax.Var("b", relax.TensorStructInfo([1, 5, 3]))
    obj = relax.MatchCast(
        var=b,
        value=a,
        struct_info=b.struct_info,
    )
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, 5, 3), dtype="float32") = R.match_cast(a, R.Tensor((1, 5, 3), dtype="float32"))
""",
    )


def test_var_binding():
    x = tir.Var("x", "int64")
    a = relax.Var("a", relax.TensorStructInfo([1, x, 3], "float32"))
    b = relax.Var("b", relax.TensorStructInfo([1, x, 3], "float32"))
    obj = relax.VarBinding(b, relax.op.sin(a))
    _assert_print(
        obj,
        """
x = T.int64()
a: R.Tensor((1, x, 3), dtype="float32")
b: R.Tensor((1, x, 3), dtype="float32") = R.sin(a)
""",
    )


def test_if():
    a = relax.Var("a", relax.TensorStructInfo([], "bool"))
    b = relax.Var("b", relax.TensorStructInfo([1, 2, 3], "float32"))
    c = relax.Var("c", relax.TensorStructInfo([1, 2, 3], "float32"))
    obj = relax.If(
        a,
        relax.SeqExpr([], b),
        relax.SeqExpr([], c),
    )
    _assert_print(
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
    x = tir.Var("x", "int64")
    a = relax.Var("R", relax.TensorStructInfo([1, x, 3], "float32"))
    b = relax.Var("T", relax.TensorStructInfo([1, x, 3], "float32"))
    obj = relax.VarBinding(b, relax.op.sin(a))
    _assert_print(
        obj,
        """
x = T.int64()
R_1: R.Tensor((1, x, 3), dtype="float32")
T_1: R.Tensor((1, x, 3), dtype="float32") = R.sin(R_1)
""",
    )


def test_module_cross_func_call():
    @I.ir_module
    class TestModule:
        @T.prim_func
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
    _assert_print(
        TestModule,
        """
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_func(x: T.Buffer((T.int64(128),), "float32"), y: T.Buffer((T.int64(128),), "float32")):
        T.evaluate(0)

    @R.function
    def foo(x: R.Tensor((128,), dtype="float32")) -> R.Tensor((128,), dtype="float32"):
        cls = Module
        gv0 = R.call_tir(cls.tir_func, (x,), out_sinfo=R.Tensor((128,), dtype="float32"))
        return gv0
""",
    )

    # empty module alias
    module_str = TestModule.script(module_alias="")
    _assert_print(
        module_str,
        """
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def tir_func(x: T.Buffer((T.int64(128),), "float32"), y: T.Buffer((T.int64(128),), "float32")):
        T.evaluate(0)

    @R.function
    def foo(x: R.Tensor((128,), dtype="float32")) -> R.Tensor((128,), dtype="float32"):
        gv0 = R.call_tir(Module.tir_func, (x,), out_sinfo=R.Tensor((128,), dtype="float32"))
        return gv0
""",
    )


if __name__ == "__main__":
    tvm.testing.main()
