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
# ruff: noqa: F841

import pytest

import tvm.script
import tvm.testing
from tvm import relax
from tvm.ir import assert_structural_equal
from tvm.relax.testing.runtime_builtin import MakeShapeCode, MatchShapeCode
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

# note: we expected RemovePurityChecking to be run first, so we force purity in most test cases


def _assert_runtime_prim_value_seeds_slot(mod, var_name):
    func = mod["main"]
    runtime_var = None
    match_inputs = []
    for block in func.body.blocks:
        for binding in block.bindings:
            if runtime_var is None and binding.var.name_hint == var_name:
                runtime_var = binding.var
            if not isinstance(binding, relax.VarBinding):
                continue
            value = binding.value
            if (
                isinstance(value, relax.Call)
                and isinstance(value.op, relax.ExternFunc)
                and value.op.global_symbol == "vm.builtin.match_prim_value"
            ):
                match_inputs.append(value.args[0])

    assert runtime_var is not None
    assert any(value.same_as(runtime_var) for value in match_inputs)


def _assert_vm_codegen_succeeds(mod):
    tvm.get_global_func("relax.VMCodeGen")(relax.ExecBuilder(), mod)


def test_const_shape_arg():
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            return x

        @T.prim_func(s_tir=True)
        def extra_func(H: T.Buffer(T.int64(4), "int64")):
            """Extra function, checks if the pass preserves it."""
            H[T.int64(1)] = H[T.int64(0)] + T.int64(1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_shape_info", x, 2, "", ty_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, -1, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_IMM,
                1,
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                "",
                ty_args=[R.Tuple()],
            )
            return x

        @T.prim_func(s_tir=True)
        def extra_func(H: T.Buffer(T.int64(4), "int64")):
            H[T.int64(1)] = H[T.int64(0)] + T.int64(1)

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_static_fn_check():
    """Check static shape and function."""
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(f: R.Callable([R.Any], R.Any), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            return y

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(f: R.Callable([R.Any], R.Any), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_func_info", f, "", ty_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, 2, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_IMM,
                1,
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                "",
                ty_args=[R.Tuple()],
            )
            return y

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_simple_symbolic_shape():
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(["n", 2, "m"], "float32")):
            R.func_attr({"relax.force_pure": True})
            return x

    sindex = {
        "n": 0,
        "m": 1,
    }

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(["n", 2, "m"], "float32")):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                3,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                3,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_IMM,
                2,
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            return x

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_symbolic_compute():
    MS = MatchShapeCode
    MK = MakeShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(["n", "m"], "float32"), y: R.Tensor(ndim=3, dtype=None)) -> R.Shape(
            ndim=3
        ):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            k = T.int64()
            z = R.match_cast(y, R.Tensor([k, m, k + 1], dtype=None))
            return R.shape([k + 1, m, 2])

    # slot assignment:
    # 0: n, 1: m, 2:k, 3: k+1
    sindex = {"n": 0, "m": 1, "k": 2, "k+1": 3}

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True, s_tir=True)
        def shape_func(H: T.Buffer(T.int64(4), "int64")):
            # generated compute function
            T.func_attr({"tirx.is_host_func": True})
            H[T.int64(sindex["k+1"])] = H[T.int64(sindex["k"])] + T.int64(1)

        @R.function
        def main(x: R.Tensor(["n", "m"], "float32"), y: R.Tensor(ndim=3, dtype=None)) -> R.Shape(
            ndim=3
        ):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            k = T.int64()
            cls = Expected
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(4)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            gv = R.null_value()
            _ = R.call_packed("vm.builtin.check_tensor_info", y, 3, gv, "", ty_args=[R.Tuple()])
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                3,
                MS.STORE_TO_HEAP,
                sindex["k"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                MS.NO_OP,
                0,
                "",
                ty_args=[R.Tuple()],
            )
            _ = cls.shape_func(shape_heap)
            # extra assertion on y's shape after shape computation
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y,
                shape_heap,
                3,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["k"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["k+1"],
                "",
                ty_args=[R.Tuple()],
            )
            z = R.match_cast(y, R.Tensor([k, m, k + 1], dtype=None))
            # construct shape value for return
            s = R.call_packed(
                "vm.builtin.make_shape",
                shape_heap,
                3,
                MK.LOAD_SHAPE,
                sindex["k+1"],
                MK.LOAD_SHAPE,
                sindex["m"],
                MK.USE_IMM,
                2,
                ty_args=[R.Shape(ndim=3)],
            )
            return s

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_tuple_handling():
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tuple(
                R.Tensor(["n", "m"], "float32"), R.Tuple(R.Shape, R.Tensor(["n", "k"], "int32"))
            ),
        ):
            R.func_attr({"relax.force_pure": True})
            return x

    # slot assignment:
    sindex = {"n": 0, "m": 1, "k": 2}

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tuple(
                R.Tensor(["n", "m"], "float32"), R.Tuple(R.Shape, R.Tensor(["n", "k"], "int32"))
            ),
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(3)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            # recursively unpack tuple for static info check
            _ = R.call_packed("vm.builtin.check_tuple_info", x, 2, "", ty_args=[R.Tuple()])
            t0 = x[0]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t0,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            t1 = x[1]
            _ = R.call_packed("vm.builtin.check_tuple_info", t1, 2, "", ty_args=[R.Tuple()])
            t1x0 = t1[0]
            _ = R.call_packed("vm.builtin.check_shape_info", t1x0, -1, "", ty_args=[R.Tuple()])
            t1x1 = t1[1]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t1x1,
                2,
                R.dtype("int32"),
                "",
                ty_args=[R.Tuple()],
            )
            # match shape checks.
            _ = R.call_packed(
                "vm.builtin.match_shape",
                t0,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                t1x1,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["k"],
                "",
                ty_args=[R.Tuple()],
            )
            return x

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_return_match_check():
    """Test when return body is not same as ret_ty, runtime match check needed."""
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(["n", "m"], "float32"), y: R.Any) -> R.Tuple(
            R.Tensor(["n", "m"], "float32")
        ):
            R.func_attr({"relax.force_pure": True})
            return y

    # slot assignment:
    sindex = {
        "n": 0,
        "m": 1,
    }

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(["n", "m"], "float32"), y: R.Any) -> R.Tuple(
            R.Tensor(["n", "m"], "float32")
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", ty_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.STORE_TO_HEAP,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed("vm.builtin.check_tuple_info", y, 1, "", ty_args=[R.Tuple()])
            # emit runtime function call since y do not have the right type.
            y1 = R.call_packed("vm.builtin.tuple_getitem", y, 0, ty_args=[R.Any])
            # run check
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                y1,
                2,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            # shape check
            _ = R.call_packed(
                "vm.builtin.match_shape",
                y1,
                shape_heap,
                2,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["m"],
                "",
                ty_args=[R.Tuple()],
            )

            return y

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_return_match_check_with_new_expr():
    """Like test_return_match_check, but requires a computation

    When return body is not same as ret_ty, a runtime match
    check is required.  This match check may require a symbolic
    expression to be computed.
    """
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor(["n", "n"], "float32")) -> R.Tensor(["n * n"], "float32"):
            R.func_attr({"relax.force_pure": True})
            out = R.call_packed("flatten_matrix", x, ty_args=R.Any)
            return out

    # slot assignment:
    sindex = {
        "n": 0,
        "n * n": 1,
    }

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(["n", "n"], "float32")) -> R.Tensor(["n * n"], "float32"):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", ty_args=[R.Tuple()]
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                2,
                MS.STORE_TO_HEAP,
                sindex["n"],
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n"],
                "",
                ty_args=[R.Tuple()],
            )

            _ = Expected.shape_func(shape_heap)

            out = R.call_packed("flatten_matrix", x, ty_args=R.Any)
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                out,
                1,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_shape",
                out,
                shape_heap,
                1,
                MS.ASSERT_EQUAL_TO_LOAD,
                sindex["n * n"],
                "",
                ty_args=[R.Tuple()],
            )
            return out

        @T.prim_func(private=True, s_tir=True)
        def shape_func(H: T.Buffer(T.int64(2), "int64")):
            # generated compute function
            T.func_attr({"tirx.is_host_func": True})
            H[T.int64(sindex["n * n"])] = H[T.int64(sindex["n"])] * H[T.int64(sindex["n"])]

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_symbolic_shape_multiple_function():
    MS = MatchShapeCode
    MK = MakeShapeCode

    @I.ir_module
    class Before:
        @R.function
        def fn1(A: R.Tensor(("m", "n"), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            n = T.int64()
            return A

        @R.function
        def fn2(A: R.Tensor(("n", "m"), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            n = T.int64()
            m = T.int64()
            return A

    # slot assignment:
    sindex_fn1 = {
        "m": 0,
        "n": 1,
    }
    sindex_fn2 = {
        "n": 0,
        "m": 1,
    }

    @I.ir_module
    class Expected:
        @R.function
        def fn1(A: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            n = T.int64()
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(2),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                A,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                A,
                shape_heap,
                R.prim_value(2),
                MS.STORE_TO_HEAP,
                sindex_fn1["m"],
                MS.STORE_TO_HEAP,
                sindex_fn1["n"],
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return A

        @R.function
        def fn2(A: R.Tensor(("n", "m"), dtype="float32")) -> R.Tensor(("n", "m"), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            n = T.int64()
            m = T.int64()
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(2),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _2: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                A,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _3: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                A,
                shape_heap,
                R.prim_value(2),
                MS.STORE_TO_HEAP,
                sindex_fn2["n"],
                MS.STORE_TO_HEAP,
                sindex_fn2["m"],
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return A

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_check_lifted_weights():
    MS = MatchShapeCode

    @I.ir_module
    class Before:
        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((16, 16), dtype="float32"))) -> R.Tuple(
            R.Tensor((16, 16), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            return params

        @R.function
        def main(x: R.Tensor((16, 16), "float32"), param_0: R.Tensor((16, 16), dtype="float32")):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            return (x, param_0)

    @I.ir_module
    class Expected:
        @R.function
        def main_transform_params(params: R.Tuple(R.Tensor((16, 16), dtype="float32"))) -> R.Tuple(
            R.Tensor((16, 16), dtype="float32")
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap: R.Any = R.null_value()
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tuple_info",
                params,
                R.prim_value(1),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            gv: R.Tensor((16, 16), dtype="float32") = params[0]
            _1: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                gv,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _2: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                gv,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return params

        @R.function
        def main(
            x: R.Tensor((16, 16), dtype="float32"), param_0: R.Tensor((16, 16), dtype="float32")
        ) -> R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((16, 16), dtype="float32")):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            shape_heap: R.Any = R.null_value()
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            return (x, param_0)

    before = Before
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    expected = Expected
    assert_structural_equal(after, expected)


def test_check_weights_with_dynamic_shape():
    MS = MatchShapeCode

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"),
            params: R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor(("n",), "float32")),
        ):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            n = T.int64()
            param_0 = params[0]
            param_1 = params[1]
            return (x, param_0, param_1)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float32"),
            params: R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor(("n",), "float32")),
        ):
            n = T.int64()
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            shape_heap: R.Tensor(dtype="int64", ndim=1) = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                (R.prim_value(1),),
                ty_args=(R.Tensor(dtype="int64", ndim=1),),
            )
            _: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                R.prim_value(2),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _1: R.Tuple = R.call_packed(
                "vm.builtin.check_tuple_info",
                params,
                R.prim_value(2),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _param_1: R.Tensor((n,), dtype="float32") = params[1]
            _2: R.Tuple = R.call_packed(
                "vm.builtin.check_tensor_info",
                _param_1,
                R.prim_value(1),
                R.dtype("float32"),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _3: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                x,
                shape_heap,
                R.prim_value(2),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                MS.ASSERT_EQUAL_TO_IMM,
                R.prim_value(16),
                R.str(""),
                ty_args=(R.Tuple,),
            )
            _4: R.Tuple = R.call_packed(
                "vm.builtin.match_shape",
                _param_1,
                shape_heap,
                MS.STORE_TO_HEAP,
                R.prim_value(1),
                R.prim_value(0),
                R.str(""),
                ty_args=(R.Tuple,),
            )

            param_0 = params[0]
            param_1 = params[1]
            return (x, param_0, param_1)

    before = Before
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    print(after)
    expected = Expected
    assert_structural_equal(after, expected)


def test_runtime_prim_value_used_as_shape_dimension():
    MS = MatchShapeCode
    MK = MakeShapeCode

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), axis: R.Prim("int64")) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            n: R.Prim("int64") = R.call_pure_packed(
                "test.identity", axis, ty_args=(R.Prim("int64"),)
            )
            out = R.reshape(A, R.shape([n]))
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), axis: R.Prim("int64")) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                ty_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                A,
                1,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.check_prim_value_info",
                axis,
                R.dtype("int64"),
                "",
                ty_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.match_prim_value",
                axis,
                shape_heap,
                1,
                0,
                "",
                ty_args=[R.Tuple()],
            )
            gv: R.Prim("int64") = R.call_packed(
                "vm.builtin.make_prim_value",
                shape_heap,
                MK.LOAD_SHAPE,
                0,
                ty_args=[R.Prim("int64")],
            )
            n: R.Prim("int64") = R.call_pure_packed("test.identity", gv, ty_args=(R.Prim("int64"),))
            _ = R.call_packed(
                "vm.builtin.match_prim_value",
                n,
                shape_heap,
                MS.STORE_TO_HEAP,
                1,
                "",
                ty_args=[R.Tuple()],
            )
            shape = R.call_packed(
                "vm.builtin.make_shape",
                shape_heap,
                1,
                MK.LOAD_SHAPE,
                1,
                ty_args=[R.Shape(ndim=1)],
            )
            out = R.reshape(A, shape)
            return out

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    assert_structural_equal(after, Expected)


def test_primitive_dataflow_alias_stays_runtime_value():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Prim("float32")) -> R.Prim("float32"):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Prim("float32") = R.prim_value(x)
                gv: R.Prim("float32") = R.call_pure_packed(
                    "test.identity", lv, ty_args=(R.Prim("float32"),)
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Prim("float32")) -> R.Prim("float32"):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed(
                "vm.builtin.check_prim_value_info",
                x,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            with R.dataflow():
                lv: R.Prim("float32") = R.prim_value(x)
                gv: R.Prim("float32") = R.call_pure_packed(
                    "test.identity", lv, ty_args=(R.Prim("float32"),)
                )
                R.output(gv)
            return gv

    dataflow_var = Before["main"].body.blocks[0].bindings[0].var
    assert isinstance(dataflow_var, relax.DataflowVar)
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    assert_structural_equal(after, Expected)


def test_non_shape_primitive_alias_stays_runtime_value():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Prim("float32")) -> R.Prim("float32"):
            R.func_attr({"relax.force_pure": True})
            alias: R.Prim("float32") = R.prim_value(x)
            return alias

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Prim("float32")) -> R.Prim("float32"):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed(
                "vm.builtin.check_prim_value_info",
                x,
                R.dtype("float32"),
                "",
                ty_args=[R.Tuple()],
            )
            alias: R.Prim("float32") = R.prim_value(x)
            return alias

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    assert_structural_equal(after, Expected)


def test_signed_int64_varbinding_roots_lower():
    @I.ir_module
    class Undemanded:
        @R.function
        def main(x: R.Prim("int64")) -> R.Prim("int64"):
            R.func_attr({"relax.force_pure": True})
            alias: R.Prim("int64") = R.prim_value(x)
            return alias

    @I.ir_module
    class Demanded:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), x: R.Prim("int64")) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            alias: R.Prim("int64") = R.prim_value(x)
            out = R.reshape(A, R.shape([alias]))
            return out

    @I.ir_module
    class ConstantDemanded:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1)) -> R.Tensor(dtype="float32", ndim=1):
            R.func_attr({"relax.force_pure": True})
            extent: R.Prim("int64") = R.prim_value(5)
            out = R.reshape(A, R.shape([extent]))
            return out

    for before in [Undemanded, Demanded, ConstantDemanded]:
        after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
        relax.analysis.well_formed(after)


def test_signed_int64_dataflow_var_stays_runtime_value():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), x: R.Prim("int64")) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Prim("int64") = R.prim_value(x)
                gv: R.Prim("int64") = R.call_pure_packed(
                    "test.identity", lv, ty_args=(R.Prim("int64"),)
                )
                R.output(gv)
            out = R.reshape(A, R.shape([gv]))
            return out

    dataflow_var = Before["main"].body.blocks[0].bindings[0].var
    assert isinstance(dataflow_var, relax.DataflowVar)
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    relax.analysis.well_formed(after)
    _assert_runtime_prim_value_seeds_slot(after, "gv")


def test_symbolic_shape_var_dataflow_runtime_value_compiles():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(("n",), dtype="float32")) -> R.Prim("int64"):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Prim("int64") = R.prim_value(n)  # noqa: F821
                gv: R.Prim("int64") = R.prim_value(lv)
                R.output(gv)
            return gv

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    relax.analysis.well_formed(after)
    _assert_vm_codegen_succeeds(after)


def test_primitive_tuple_get_item_stays_runtime_value():
    @I.ir_module
    class Before:
        @R.function
        def main(values: R.Tuple(R.Prim("int64"))) -> R.Shape(ndim=1):
            R.func_attr({"relax.force_pure": True})
            extent: R.Prim("int64") = values[0]
            out = R.shape([extent])
            return out

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    relax.analysis.well_formed(after)
    _assert_runtime_prim_value_seeds_slot(after, "extent")
    _assert_vm_codegen_succeeds(after)


def test_demanded_signed_int64_match_cast_seeds_shape_slot():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), x: R.Any) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            matched: R.Prim("int64") = R.match_cast(x, R.Prim("int64"))
            out = R.reshape(A, R.shape([matched]))
            return out

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    relax.analysis.well_formed(after)
    _assert_runtime_prim_value_seeds_slot(after, "matched")


def test_demanded_signed_int64_dataflow_match_cast_seeds_shape_slot():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Any) -> R.Shape(ndim=1):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                matched: R.Prim("int64") = R.match_cast(x, R.Prim("int64"))
                R.output(matched)
            out = R.shape([matched])
            return out

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    relax.analysis.well_formed(after)
    runtime_var = None
    check_inputs = []
    for block in after["main"].body.blocks:
        for binding in block.bindings:
            if runtime_var is None and binding.var.name_hint == "matched":
                runtime_var = binding.var
            if isinstance(binding, relax.VarBinding) and isinstance(binding.value, relax.Call):
                if isinstance(block, relax.DataflowBlock):
                    assert not isinstance(binding.value.op, relax.ExternFunc)
                elif (
                    isinstance(binding.value.op, relax.ExternFunc)
                    and binding.value.op.global_symbol == "vm.builtin.check_prim_value_info"
                ):
                    check_inputs.append(binding.value.args[0])
    assert runtime_var is not None
    assert any(value.same_as(runtime_var) for value in check_inputs)
    _assert_runtime_prim_value_seeds_slot(after, "matched")
    _assert_vm_codegen_succeeds(after)


def test_dataflow_match_cast_check_stays_in_if_branch():
    prim_type = tvm.ir.PrimType("int64")
    cond = relax.Var("cond", tvm.ir.PrimType("bool"))
    x = relax.Var("x", relax.AnyType())
    matched = relax.Var("matched", prim_type)
    true_branch = relax.SeqExpr(
        [relax.DataflowBlock([relax.MatchCast(matched, x, prim_type)])],
        tvm.tirx.IntImm("int64", 0),
    )
    false_branch = relax.SeqExpr([], tvm.tirx.IntImm("int64", 1))
    result = relax.Var("result", prim_type)
    body = relax.SeqExpr(
        [relax.BindingBlock([relax.VarBinding(result, relax.If(cond, true_branch, false_branch))])],
        result,
    )
    func = relax.Function([cond, x], body, prim_type).with_attrs(
        {"global_symbol": "main", "relax.force_pure": True}
    )
    before = relax.transform.Normalize()(tvm.IRModule({"main": func}))
    relax.analysis.well_formed(before)

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    relax.analysis.well_formed(after)
    _assert_vm_codegen_succeeds(after)

    if_expr = next(
        binding.value
        for block in after["main"].body.blocks
        for binding in block.bindings
        if isinstance(binding, relax.VarBinding) and isinstance(binding.value, relax.If)
    )
    branch_checks = [
        binding.value
        for block in if_expr.true_branch.blocks
        for binding in block.bindings
        if isinstance(binding, relax.VarBinding)
        and isinstance(binding.value, relax.Call)
        and isinstance(binding.value.op, relax.ExternFunc)
        and binding.value.op.global_symbol == "vm.builtin.check_prim_value_info"
    ]
    assert len(branch_checks) == 1
    assert branch_checks[0].args[0].name_hint == "matched"


def test_internal_dataflow_match_cast_is_rejected():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Any) -> R.Prim("int64"):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Prim("int64") = R.match_cast(x, R.Prim("int64"))
                gv: R.Prim("int64") = R.prim_value(lv)
                R.output(gv)
            return gv

    with pytest.raises(ValueError, match="MatchCast bound to an internal DataflowVar"):
        relax.transform.VMShapeLower(emit_err_ctx=False)(Before)


def test_runtime_primitive_if_result_seeds_shape_slot():
    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor(dtype="float32", ndim=1),
            cond: R.Prim("bool"),
            x: R.Prim("int64"),
            y: R.Prim("int64"),
        ) -> R.Tensor(dtype="float32", ndim=1):
            R.func_attr({"relax.force_pure": True})
            if cond:
                out: R.Prim("int64") = R.prim_value(x)
            else:
                out: R.Prim("int64") = R.prim_value(y)
            result = R.reshape(A, R.shape([out]))
            return result

    after = relax.transform.VMShapeLower(emit_err_ctx=False)(Before)
    _assert_runtime_prim_value_seeds_slot(after, "out")


def test_compute_prim_value_arithmetic_result_seeds_shape_slot():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(dtype="float32", ndim=1), n: R.Prim("int64")) -> R.Tensor(
            dtype="float32", ndim=1
        ):
            R.func_attr({"relax.force_pure": True})
            extent: R.Prim("int64") = R.prim_value(n + 1)
            out = R.reshape(A, R.shape([extent]))
            return out

    computed = relax.transform.ComputePrimValue()(Before)
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(computed)
    _assert_runtime_prim_value_seeds_slot(after, "extent")


if __name__ == "__main__":
    tvm.testing.main()
