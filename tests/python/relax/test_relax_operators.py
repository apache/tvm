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

import sys
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm._ffi.base import TVMError
from tvm.script import ir as I, relax as R, tir as T

exec_mode = tvm.testing.parameter("bytecode", "compiled")


@tvm.script.ir_module
class InputModule:
    @R.function
    def foo(x: R.Tensor(("m", "n"), "int64")):
        y = R.unique(x, sorted=False)
        y_sorted = R.unique(x)
        return y, y_sorted


def run_cpu(mod, func_name, *args, exec_mode):
    if isinstance(mod, relax.Function):
        func = mod
        args = [func_name, *args]
        func_name = func.attrs["global_symbol"]
        mod = tvm.IRModule.from_expr(func)

    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    return vm[func_name](*args)


def test_unique(exec_mode):
    # TODO(prakalp): also add test for compiling and running on cuda device.
    data_numpy = np.random.randint(0, 16, (16, 16))
    data = tvm.nd.array(data_numpy)
    result, result_sorted = run_cpu(InputModule, "foo", data, exec_mode=exec_mode)

    expected_output_sorted, indices = np.unique(data_numpy, return_index=True)
    expected_output = [data_numpy.flatten()[index] for index in sorted(indices)]

    np.testing.assert_array_equal(expected_output_sorted, result_sorted.numpy())
    np.testing.assert_array_equal(expected_output, result.numpy())


@tvm.script.ir_module
class PrintTest:
    @R.function(pure=False)
    def foo(x: R.Tensor((), "int32")):
        # results have to be bound, but we don't use them
        # TODO: We should allow calls whose results are not bound for side effects;
        #       it would be easy syntactic sugar to add.
        p1 = R.print(x)
        p2 = R.print(x, format="Number: {}")
        t = (x, x)
        p3 = R.print(t, format="Tuple: {}")
        p4 = R.print(x, t)
        p5 = R.print(x, x, format="Custom print: {} {}")
        p6 = R.print(x, t, format="Another print: {} {}")
        return x


def test_print(exec_mode):
    try:
        stdout = sys.stdout
        with tempfile.TemporaryFile(mode="w+") as test_out:
            sys.stdout = test_out
            run_cpu(
                PrintTest,
                "foo",
                tvm.nd.array(np.array(1).astype("int32")),
                exec_mode=exec_mode,
            )
            test_out.seek(0)
            printed_text = str(test_out.read())
            expected = "1\nNumber: 1\nTuple: (1, 1)\n1 (1, 1)\nCustom print: 1 1\nAnother print: 1 (1, 1)\n"
            assert printed_text in expected, ("printed_text is ", printed_text)
    finally:
        sys.stdout = stdout


def test_assert_passes(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(True))
        return x

    run_cpu(func, tvm.nd.array(np.array(1).astype("int32")), exec_mode=exec_mode)


def test_assert_passes_with_format_args(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(True), x, format="You won't see me")
        return x

    run_cpu(func, tvm.nd.array(np.array(1).astype("int32")), exec_mode=exec_mode)


def test_assert_fails(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False))
        return x

    with pytest.raises(AssertionError, match="Assertion Failed"):
        run_cpu(func, tvm.nd.array(np.array(1).astype("int32")), exec_mode=exec_mode)


def test_assert_fails_with_message(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), format="I failed...")
        return x

    with pytest.raises(AssertionError, match="I failed..."):
        run_cpu(func, tvm.nd.array(np.array(1).astype("int32")), exec_mode=exec_mode)


def test_assert_fails_with_args(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), [x, x])
        return x

    with pytest.raises(AssertionError, match="5, 5"):
        run_cpu(func, tvm.nd.array(np.array(5).astype("int32")), exec_mode=exec_mode)


def test_assert_fails_with_formatted_args(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), x, format="Number: {}")
        return x

    with pytest.raises(AssertionError, match="Number: 6"):
        run_cpu(func, tvm.nd.array(np.array(6).astype("int32")), exec_mode=exec_mode)


def test_assert_on_argument_passes(exec_mode):
    @R.function(pure=False)
    def func(condition: R.Tensor((), "bool"), x: R.Tensor((), "int32")):
        _ = R.assert_op(condition)
        return x

    condition = tvm.nd.array(np.array(True))
    x = tvm.nd.array(np.array(5).astype("int32"))
    run_cpu(func, condition, x, exec_mode=exec_mode)


def test_assert_on_argument_fails(exec_mode):
    @R.function(pure=False)
    def func(condition: R.Tensor((), "bool"), x: R.Tensor((), "int32")):
        _ = R.assert_op(condition)
        return x

    condition = tvm.nd.array(np.array(False))
    x = tvm.nd.array(np.array(5).astype("int32"))
    with pytest.raises(AssertionError):
        run_cpu(func, condition, x, exec_mode=exec_mode)


def test_assert_on_symbolic_var_passes(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor(["N"], "int32")):
        N = T.int64()
        _ = R.assert_op(R.prim_value(N % 8 == 0))
        return x

    x = tvm.nd.array(np.arange(8, dtype="int32"))
    run_cpu(func, x, exec_mode=exec_mode)


def test_assert_on_symbolic_var_fails(exec_mode):
    @R.function(pure=False)
    def func(x: R.Tensor(["N"], "int32")):
        N = T.int64()
        _ = R.assert_op(R.prim_value(N % 8 == 0))
        return x

    x = tvm.nd.array(np.arange(10, dtype="int32"))
    with pytest.raises(AssertionError):
        run_cpu(func, x, exec_mode=exec_mode)


@tvm.script.ir_module
class ShapeOfTest:
    @R.function
    def get_shape(t: R.Tensor(ndim=-1, dtype="int32")) -> R.Shape(ndim=-1):
        return R.shape_of(t)

    @R.function
    def get_constrained_shape(t: R.Tensor(ndim=1, dtype="int32")) -> R.Shape(ndim=1):
        # require the input tensor to have rank 1
        return R.shape_of(t)

    @R.function
    def get_scalar_shape() -> R.Shape(()):
        x: R.Tensor((), "int32") = R.const(1, dtype="int32")
        return R.shape_of(x)

    @R.function
    def get_constant_shape() -> R.Shape((2, 2)):
        x: R.Tensor((2, 2), "int32") = R.const(
            np.array([[1, 2], [3, 4]], dtype="int32"), dtype="int32"
        )
        return R.shape_of(x)


def test_op_shape_of(exec_mode):
    unit_shape = run_cpu(ShapeOfTest, "get_scalar_shape", exec_mode=exec_mode)
    assert unit_shape == tvm.runtime.ShapeTuple([])

    const_shape = run_cpu(ShapeOfTest, "get_constant_shape", exec_mode=exec_mode)
    assert const_shape == tvm.runtime.ShapeTuple([2, 2])

    scalar_shape = run_cpu(
        ShapeOfTest, "get_shape", tvm.nd.array(np.array(1, dtype="int32")), exec_mode=exec_mode
    )
    assert scalar_shape == tvm.runtime.ShapeTuple([])

    tensor_shape = run_cpu(
        ShapeOfTest,
        "get_shape",
        tvm.nd.array(np.zeros((1, 2, 3)).astype("int32")),
        exec_mode=exec_mode,
    )
    assert tensor_shape == tvm.runtime.ShapeTuple([1, 2, 3])

    constrained_shape = run_cpu(
        ShapeOfTest,
        "get_constrained_shape",
        tvm.nd.array(np.zeros((1,)).astype("int32")),
        exec_mode=exec_mode,
    )
    assert constrained_shape == tvm.runtime.ShapeTuple([1])


@tvm.script.ir_module
class ShapeToTensorTest:
    @R.function
    def const_shape(shape: R.Shape(ndim=-1)) -> R.Tensor(ndim=-1):
        return R.shape_to_tensor(shape)

    @R.function
    def symbolic_shape(shape: R.Shape(("m", "n"))) -> R.Tensor(ndim=-1):
        m = T.int64()
        n = T.int64()
        return R.shape_to_tensor(shape)


def test_op_shape_to_tensor(exec_mode):
    # Check struct info
    isinstance(ShapeToTensorTest["const_shape"].body.struct_info, tvm.relax.TensorStructInfo)
    assert ShapeToTensorTest["const_shape"].body.struct_info.ndim == 1
    isinstance(ShapeToTensorTest["symbolic_shape"].body.struct_info, tvm.relax.TensorStructInfo)
    assert ShapeToTensorTest["symbolic_shape"].body.struct_info.ndim == 1

    # Check its functionality
    out2d = run_cpu(
        ShapeToTensorTest, "const_shape", tvm.runtime.ShapeTuple([3, 2]), exec_mode=exec_mode
    )
    assert isinstance(out2d, tvm.runtime.ndarray.NDArray)
    assert np.array_equal(out2d.numpy(), np.array([3, 2]))

    out3d = run_cpu(
        ShapeToTensorTest, "const_shape", tvm.runtime.ShapeTuple([3, 3, 2]), exec_mode=exec_mode
    )
    assert isinstance(out3d, tvm.runtime.ndarray.NDArray)
    assert np.array_equal(out3d.numpy(), np.array([3, 3, 2]))

    out4d = run_cpu(
        ShapeToTensorTest, "const_shape", tvm.runtime.ShapeTuple([3, 3, 2, 2]), exec_mode=exec_mode
    )
    assert isinstance(out4d, tvm.runtime.ndarray.NDArray)
    assert np.array_equal(out4d.numpy(), np.array([3, 3, 2, 2]))

    outs = run_cpu(
        ShapeToTensorTest, "symbolic_shape", tvm.runtime.ShapeTuple([3, 2]), exec_mode=exec_mode
    )
    assert isinstance(outs, tvm.runtime.ndarray.NDArray)
    assert np.array_equal(outs.numpy(), np.array([3, 2]))


def test_op_call_pure_packed(exec_mode):
    @tvm.script.ir_module
    class CallPureTest:
        @R.function
        def pure_copy(x: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.copy", x, sinfo_args=(R.Tensor((3, 4), dtype="float32"))
            )
            return z

    np.random.seed(0)  # to avoid flakiness
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(CallPureTest, "pure_copy", tvm.nd.array(arr), exec_mode=exec_mode)
    assert (copy_found.numpy() == arr).all()


def test_op_call_inplace_packed(exec_mode):
    # in this case we can use the same test as above
    @tvm.script.ir_module
    class CallInplaceTest:
        @R.function
        def pure_copy(x: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "vm.builtin.copy",
                x,
                inplace_indices=0,
                sinfo_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    @tvm.register_func("test.inplace.add", override=True)
    def inplace_add(a, b):
        arr_a = a.numpy()
        arr_b = b.numpy()
        for i in range(len(arr_a)):
            for j in range(len(arr_a[i])):
                arr_a[i][j] = arr_a[i][j] + arr_b[i][j]
        a.copyfrom(arr_a)
        return a

    @tvm.script.ir_module
    class CallInplaceAddTest:
        @R.function
        def inplace_add(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "test.inplace.add",
                x,
                y,
                inplace_indices=0,
                sinfo_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(1)  # to avoid flakiness
    arr_a = np.random.rand(3, 4).astype("float32")
    arr_b = np.random.rand(3, 4).astype("float32")
    sum = arr_a + arr_b
    tvm_arr_a = tvm.nd.array(arr_a)
    result = run_cpu(
        CallInplaceAddTest, "inplace_add", tvm_arr_a, tvm.nd.array(arr_b), exec_mode=exec_mode
    )
    assert result == tvm_arr_a
    assert (result.numpy() == sum).all()

    @tvm.register_func("test.inplace.tuple_add", override=True)
    def inplace_tuple_add(a, b):
        arr_a = a.numpy()
        arr_b = b.numpy()
        c = tvm.nd.array(arr_a + arr_b)
        for i in range(len(arr_a)):
            for j in range(len(arr_a[i])):
                arr_a[i][j] = arr_a[i][j] + arr_b[i][j]
        a.copyfrom(arr_a)
        return tvm.runtime.container.ADT(0, [a, c])

    @tvm.script.ir_module
    class CallInplaceTuple:
        @R.function
        def inplace_tuple(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "test.inplace.tuple_add",
                x,
                y,
                inplace_indices=[0, -1],
                sinfo_args=(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(2)  # to avoid flakiness
    arr_a = np.random.rand(3, 4).astype("float32")
    arr_b = np.random.rand(3, 4).astype("float32")
    sum = arr_a + arr_b
    tvm_arr_a = tvm.nd.array(arr_a)
    tvm_arr_b = tvm.nd.array(arr_b)
    result = run_cpu(CallInplaceTuple, "inplace_tuple", tvm_arr_a, tvm_arr_b, exec_mode=exec_mode)
    assert result[0] == tvm_arr_a
    assert (result[0].numpy() == sum).all()
    assert result[1] != tvm_arr_a and result[1] != tvm_arr_b
    assert (result[1].numpy() == sum).all()


def test_op_to_device(exec_mode):
    @tvm.script.ir_module
    class CallToDevice:
        @R.function
        def to_dev(x: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.to_device",
                x,
                1,
                0,
                sinfo_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(0)  # to avoid flakiness
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(CallToDevice, "to_dev", tvm.nd.array(arr), exec_mode=exec_mode)
    assert (copy_found.numpy() == arr).all()


def test_op_to_vdevice(exec_mode):
    @tvm.script.ir_module
    class ToVDevice:
        I.module_global_infos({"vdevice": [I.vdevice("llvm")]})

        @R.function
        def to_vdev(x: R.Tensor((3, 4), "float32")):
            dst_vdev = tvm.ir.VDevice("llvm", 0, "global")
            ret = R.to_vdevice(x, "llvm")
            return ret

    np.random.seed(0)
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(ToVDevice, "to_vdev", tvm.nd.array(arr), exec_mode=exec_mode)
    assert (copy_found.numpy() == arr).all()


def test_scalar_tensor_as_branch_condition(exec_mode):
    """The condition of a branch may be a scalar tensor"""

    @R.function
    def func(condition: R.Tensor((), "bool")):
        if condition:
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, tvm.nd.array(np.array(True)), exec_mode=exec_mode)
    assert res == 5

    res = run_cpu(func, tvm.nd.array(np.array(False)), exec_mode=exec_mode)
    assert res == 10


def test_prim_value_as_branch_condition(exec_mode):
    """The condition may be a PrimValue"""

    @R.function
    def func(condition: R.Prim("bool")):
        if condition:
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, True, exec_mode=exec_mode)
    assert res == 5

    res = run_cpu(func, False, exec_mode=exec_mode)
    assert res == 10


def test_computed_prim_value_as_branch_condition(exec_mode):
    """The R.Prim condition may be computed within the function"""

    @R.function
    def func(x: R.Tensor(["N"], "int64")):
        N = T.int64()
        if R.prim_value(N % 16 == 0):
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, tvm.nd.array(np.arange(16)), exec_mode=exec_mode)
    assert res == 5

    res = run_cpu(func, tvm.nd.array(np.arange(20)), exec_mode=exec_mode)
    assert res == 10


if __name__ == "__main__":
    tvm.testing.main()
