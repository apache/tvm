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

import tvm.script
import tvm.testing
from tvm import relax
from tvm.ir import assert_structural_equal
from tvm.relax.testing.runtime_builtin import MakeShapeCode, MatchShapeCode
from tvm.script import relax as R
from tvm.script import tir as T

# note: we expected RemovePurityChecking to be run first, so we force purity in most test cases


def test_const_shape_arg():
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            return x

        @T.prim_func
        def extra_func(H: T.Buffer(T.int64(4), "int64")):
            """Extra function, checks if the pass preserves it."""
            H[T.int64(1)] = H[T.int64(0)] + T.int64(1)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Shape([1, 2]), y: R.Shape):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_shape_info", x, 2, "", sinfo_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, -1, "", sinfo_args=[R.Tuple()])
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
                sinfo_args=[R.Tuple()],
            )
            return x

        @T.prim_func
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
        def main(f: R.Callable([R.Object], R.Object), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            return y

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(f: R.Callable([R.Object], R.Object), y: R.Shape([1, 2])):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.null_value()
            _ = R.call_packed("vm.builtin.check_func_info", f, "", sinfo_args=[R.Tuple()])
            _ = R.call_packed("vm.builtin.check_shape_info", y, 2, "", sinfo_args=[R.Tuple()])
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
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                3,
                R.dtype("float32"),
                "",
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
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
        def main(
            x: R.Tensor(["n", "m"], "float32"), y: R.Tensor(ndim=3, dtype=None)
        ) -> R.Shape(ndim=3):
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
        @T.prim_func(private=True)
        def shape_func(H: T.Buffer(T.int64(4), "int64")):
            # generated compute function
            T.func_attr({"tir.is_host_func": 1})
            H[T.int64(sindex["k+1"])] = H[T.int64(sindex["k"])] + T.int64(1)

        @R.function
        def main(
            x: R.Tensor(["n", "m"], "float32"), y: R.Tensor(ndim=3, dtype=None)
        ) -> R.Shape(ndim=3):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            k = T.int64()
            cls = Expected
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(4)],
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                x,
                2,
                R.dtype("float32"),
                "",
                sinfo_args=[R.Tuple()],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", y, 3, R.dtype(""), "", sinfo_args=[R.Tuple()]
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
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Shape(ndim=3)],
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
            )
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
            )
        ):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(3)],
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            # recursively unpack tuple for static info check
            _ = R.call_packed("vm.builtin.check_tuple_info", x, 2, "", sinfo_args=[R.Tuple()])
            t0 = x[0]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t0,
                2,
                R.dtype("float32"),
                "",
                sinfo_args=[R.Tuple()],
            )
            t1 = x[1]
            _ = R.call_packed("vm.builtin.check_tuple_info", t1, 2, "", sinfo_args=[R.Tuple()])
            t1x0 = t1[0]
            _ = R.call_packed("vm.builtin.check_shape_info", t1x0, -1, "", sinfo_args=[R.Tuple()])
            t1x1 = t1[1]
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                t1x1,
                2,
                R.dtype("int32"),
                "",
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
            )
            return x

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


def test_return_match_check():
    """Test when return body is not same as ret_struct_info, runtime match check needed."""
    MS = MatchShapeCode

    @tvm.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["n", "m"], "float32"), y: R.Object
        ) -> R.Tuple(R.Tensor(["n", "m"], "float32")):
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
        def main(
            x: R.Tensor(["n", "m"], "float32"), y: R.Object
        ) -> R.Tuple(R.Tensor(["n", "m"], "float32")):
            R.func_attr({"relax.force_pure": True})
            shape_heap = R.call_builtin_with_ctx(
                "vm.builtin.alloc_shape_heap",
                [R.prim_value(2)],
                sinfo_args=[R.Tensor(ndim=1, dtype="int64")],
            )
            _ = R.call_packed(
                "vm.builtin.check_tensor_info", x, 2, R.dtype("float32"), "", sinfo_args=[R.Tuple()]
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
                sinfo_args=[R.Tuple()],
            )
            _ = R.call_packed("vm.builtin.check_tuple_info", y, 1, "", sinfo_args=[R.Tuple()])
            # emit runtime function call since y do not have the right type.
            y1 = R.call_packed("vm.builtin.tuple_getitem", y, 0, sinfo_args=[R.Object])
            # run check
            _ = R.call_packed(
                "vm.builtin.check_tensor_info",
                y1,
                2,
                R.dtype("float32"),
                "",
                sinfo_args=[R.Tuple()],
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
                sinfo_args=[R.Tuple()],
            )

            return y

    before = Before
    expected = Expected
    after = relax.transform.VMShapeLower(emit_err_ctx=False)(before)
    assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
