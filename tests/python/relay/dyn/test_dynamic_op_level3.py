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
""" Support level3 operator test cases.
"""
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay, te
from tvm.relay.testing import check_grad, run_infer_type

executor_kind = tvm.testing.parameter("debug", "vm")


def verify_func(executor_kind, func, data, ref_res, target_device=tvm.testing.enabled_targets()):
    assert isinstance(data, list)
    for target, dev in target_device:
        mod = tvm.ir.IRModule.from_expr(func)
        op_res = relay.create_executor(
            executor_kind, mod=mod, device=dev, target=target
        ).evaluate()(*data)
        if isinstance(op_res, tvm.runtime.container.ADT):
            assert len(op_res) == len(
                ref_res
            ), "Outputs from TVM and Python implementation must be equal "
            for op_result, ref_result in zip(op_res, ref_res):
                tvm.testing.assert_allclose(op_result.numpy(), ref_result, rtol=1e-5)
        else:
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)
        relay.backend.te_compiler.get().clear()


def check_on_vm(target, dev, args, expected_result, mod):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on Relay VM.
    """
    rts_result = relay.create_executor("vm", device=dev, target=target, mod=mod).evaluate()(*args)
    tvm.testing.assert_allclose(expected_result, rts_result.numpy())


@tvm.testing.uses_gpu
def test_dyn_reshape(executor_kind):
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType((len(newshape),), "int64"))
        z = relay.reshape(x, y)

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        x_data = np.ones(shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        check_grad(
            run_infer_type(func),
            inputs=[x_data, np.array(newshape).astype("int64")],
            test_inputs=[x_data],
            eps=1e-3,
        )
        verify_func(executor_kind, func, [x_data, np.array(newshape).astype("int64")], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))
    verify_reshape((2, 3, 4), (4, 0, 2), (4, 3, 2))
    verify_reshape((2, 3, 4), (2, 0, 0), (2, 3, 4))
    verify_reshape((2, 3, 4), (0, -1), (2, 12))
    verify_reshape((2, 3, 4), (-1, 0), (8, 3))
    verify_reshape((2, 3, 4), (-3, 4), (6, 4))
    verify_reshape((2, 3, 4, 5), (-3, -3), (6, 20))
    verify_reshape((2, 3, 4), (0, -3), (2, 12))


@tvm.testing.uses_gpu
def test_dyn_shape_reshape(executor_kind):
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        check_grad(run_infer_type(func), inputs=[x_data, y_data], eps=1e-3)
        verify_func(executor_kind, func, [x_data, y_data], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))


def test_squeeze(executor_kind):
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        assert axis is not None
        np_axis = tuple(axis)
        axis = relay.var("axis", relay.TensorType([len(axis)], "int64"))
        squeeze = relay.squeeze(x, axis=axis)
        func = relay.Function([x, axis], squeeze)
        x_data = np.random.random_sample(shape).astype(dtype)
        ref_res = np.squeeze(x_data, axis=np_axis)
        verify_func(executor_kind, func, [x_data, np.array(np_axis).astype("int64")], ref_res)

    verify_squeeze((1, 3, 1), "float32", [0])
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


@tvm.testing.uses_gpu
def test_dyn_expand_dims(executor_kind):
    def verify_expand_dims(
        dshape, dtype, oshape, axis, num_newaxis, target_device=tvm.testing.enabled_targets()
    ):
        # Use 1 to avoid issues with invalid buffer sizes
        x = relay.Var("x", relay.TensorType(dshape, dtype))
        y = relay.var("axis", shape=[], dtype="int64")
        z = relay.expand_dims(x, axis=y, num_newaxis=num_newaxis)
        func = relay.Function([x, y], z)

        data_np = np.random.uniform(size=dshape).astype(dtype)
        axis_np = np.array(axis).astype("int64")
        ref_res = data_np.reshape(oshape)
        verify_func(executor_kind, func, [data_np, axis_np], ref_res, target_device=target_device)

    for dtype in ["float16", "float32"]:
        verify_expand_dims((2, 2), dtype, (2, 2, 1), 2, 1)
        verify_expand_dims((2, 2), dtype, (2, 1, 2), 1, 1)
        verify_expand_dims((2, 2), dtype, (1, 2, 2), 0, 1)

        # TODO (AndrewZhaoLuo): investigate why runtimes in non-llvm are extremely slow
        # for multiple new axis
        llvm_target_only = [x for x in tvm.testing.enabled_targets() if "llvm" in x]
        verify_expand_dims((2, 2), dtype, (2, 2, 1, 1), 2, 2, target_device=llvm_target_only)
        verify_expand_dims((2, 2), dtype, (2, 1, 1, 1, 2), 1, 3, target_device=llvm_target_only)
        verify_expand_dims((2, 2), dtype, (1, 1, 1, 1, 2, 2), 0, 4, target_device=llvm_target_only)


@tvm.testing.uses_gpu
def test_dyn_tile(executor_kind):
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        r = relay.var("reps", relay.TensorType((len(reps),), "float32"))
        z = relay.tile(x, r)

        func = relay.Function([x, r], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)
        reps_data = np.array(reps).astype("float32")
        verify_func(executor_kind, func, [x_data, np.array(reps).astype("float32")], ref_res)

    verify_tile((2, 3, 4), (3, 2, 1))
    verify_tile((2, 3, 4), (1, 2))
    verify_tile((2, 3), (3, 2, 1))


@tvm.testing.uses_gpu
def test_dyn_zeros_ones(executor_kind):
    def verify_zeros_ones(shape, dtype):
        for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
            rank = len(shape)
            dyn_shape = relay.Var("shape", relay.ty.TensorType((rank,), "int64"))
            y = op(dyn_shape, dtype)
            yy = run_infer_type(y)
            assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * rank, dtype)

            func = relay.Function([dyn_shape], y)
            ref_res = ref(shape, dtype)
            verify_func(
                executor_kind, func, [np.array(shape).astype("int64")], ref_res.astype("int64")
            )

    verify_zeros_ones((1, 3), "int64")
    verify_zeros_ones((8, 9, 1, 2), "float32")


@tvm.testing.uses_gpu
def test_dyn_full(executor_kind):
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        rank = len(src_shape)
        dyn_src_shape = relay.var("dyn_scr_shape", relay.ty.TensorType((rank,), "int64"))
        z = relay.full(x, dyn_src_shape, dtype)
        func = relay.Function([x, dyn_src_shape], z)
        ref_res = np.full(src_shape, fill_value).astype(dtype)

        verify_func(
            executor_kind,
            func,
            [np.array(fill_value).astype(dtype), np.array(src_shape).astype("int64")],
            ref_res,
        )

    verify_full(4, (1, 3, 4, 4), "int32")
    verify_full(4, (1, 3, 4, 4), "int64")
    verify_full(4.0, (2, 50), "float32")


@tvm.testing.uses_gpu
def test_dyn_sparse_to_dense(executor_kind):
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)
        output_shape_data = np.array(output_shape)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        output_shape_var = relay.var(
            "output_shape", relay.TensorType(output_shape_data.shape, str(output_shape_data.dtype))
        )
        if default_value is None:
            args = [a, b, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c, output_shape_var]
            d = relay.sparse_to_dense(a, output_shape_var, b, c)

        zz = run_infer_type(d)
        assert len(zz.checked_type.shape) == len(output_shape)

        func = relay.Function(args, d)

        if default_value is None:
            arguments = [sparse_indices_data, sparse_values_data, output_shape_data]
        else:
            arguments = [
                sparse_indices_data,
                sparse_values_data,
                default_value_data,
                output_shape_data,
            ]

        verify_func(executor_kind, func, arguments, xpected)

    verify_sparse_to_dense(1, 3, 0, [5], [0, 3, 0, 0, 0])  # scalar
    verify_sparse_to_dense([0, 1, 4], [3, 3, 3], 0, [5], [3, 3, 0, 0, 3])  # vector
    verify_sparse_to_dense(
        [[0, 0], [1, 2]], [1, 2], 0, [3, 4], [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    )  # nXd
    verify_sparse_to_dense(
        [[0, 0, 0], [1, 2, 3]],
        [1, 2],
        4,
        [2, 3, 4],
        [[[1, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]], [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 2]]],
    )  # nXd
    verify_sparse_to_dense(
        [0, 1, 4], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1]
    )  # floats
    # default value not specified
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])


@pytest.mark.parametrize(
    "sparse_indices, sparse_values, dense_shape, default_value",
    [
        (
            np.array([[0, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([5, 6], dtype=np.int64),
            np.array([10], dtype=np.int64),
        ),
        (
            np.array([[1, 1, 1], [1, 3, 1], [2, 0, 5], [3, 1, 6]], dtype=np.int64),
            np.array([1, 2, 3, 4], dtype=np.int64),
            np.array([7, 7, 7], dtype=np.int64),
            np.array([5], dtype=np.int64),
        ),
        (
            np.array([[1], [2]], dtype=np.int64),
            np.array([7, 8], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([4], dtype=np.int64),
        ),
        (
            np.ones((0, 1), dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([5], dtype=np.int64),
            np.array([4], dtype=np.int64),
        ),
        (
            np.ones((0, 3), dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([9, 3, 7], dtype=np.int64),
            np.array([100], dtype=np.int64),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.int64, np.int32])
@pytest.mark.parametrize("use_dyn", [True, False])
def test_sparse_fill_empty_rows(
    sparse_indices, sparse_values, dense_shape, default_value, dtype, use_dyn, executor_kind
):
    def ref_sparse_fill_empty_rows(
        sparse_indices: np.ndarray,
        sparse_values: np.ndarray,
        dense_shape: np.ndarray,
        default_value: np.ndarray,
    ) -> None:
        """
        This function calculates the expected output of sparse_fill_empty_rows operator given the
        inputs.
        """

        def check_add_rows(current_idx, limit_idx):
            while current_idx < limit_idx:
                new_sparse_indices.append([current_idx] + [0] * (num_cols - 1))
                new_sparse_values.append(default_value[0])
                empty_row_indicator[current_idx] = True
                current_idx += 1

            return current_idx

        current_idx = 0
        new_sparse_indices = []
        new_sparse_values = []
        empty_row_indicator = [False for _ in range(dense_shape[0])]
        num_cols = sparse_indices.shape[1]
        for sparse_row, sparse_value in zip(sparse_indices, sparse_values):
            limit_idx = sparse_row[0]
            current_idx = check_add_rows(current_idx, limit_idx)
            new_sparse_indices.append(list(sparse_row))
            new_sparse_values.append(sparse_value)
            current_idx = limit_idx + 1

        check_add_rows(current_idx, dense_shape[0])
        return new_sparse_indices, new_sparse_values, empty_row_indicator

    def verify_sparse_fill_empty_rows(
        sparse_indices_np: np.ndarray,
        sparse_values_np: np.ndarray,
        dense_shape_np: np.ndarray,
        default_value_np: np.ndarray,
    ) -> None:
        """
        This function verifies the relay output of sparse_fill_empty_rows with its expected output.
        """
        if use_dyn:
            sparse_indices = relay.var(
                "sparse_indices",
                shape=[relay.Any(), relay.Any()],
                dtype=str(sparse_indices_np.dtype),
            )
            sparse_values = relay.var(
                "sparse_values",
                shape=[relay.Any()],
                dtype=str(sparse_values_np.dtype),
            )
            dense_shape = relay.var(
                "dense_shape",
                shape=[relay.Any()],
                dtype=str(dense_shape_np.dtype),
            )
            default_value = relay.var(
                "default_value",
                shape=[relay.Any()],
                dtype=str(default_value_np.dtype),
            )
        else:
            sparse_indices = relay.var(
                "sparse_indices",
                relay.TensorType(sparse_indices_np.shape, str(sparse_indices_np.dtype)),
            )
            sparse_values = relay.var(
                "sparse_values",
                relay.TensorType(sparse_values_np.shape, str(sparse_values_np.dtype)),
            )
            dense_shape = relay.var(
                "dense_shape",
                relay.TensorType(dense_shape_np.shape, str(dense_shape_np.dtype)),
            )
            default_value = relay.var(
                "default_value",
                relay.TensorType(default_value_np.shape, str(default_value_np.dtype)),
            )
        z = relay.sparse_fill_empty_rows(sparse_indices, sparse_values, dense_shape, default_value)
        func = relay.Function([sparse_indices, sparse_values, dense_shape, default_value], z)
        ref_res = ref_sparse_fill_empty_rows(
            sparse_indices_np,
            sparse_values_np,
            dense_shape_np,
            default_value_np,
        )
        (
            new_sparse_indices_infer_type,
            new_sparse_values_infer_type,
            empty_row_indicator_infer_type,
        ) = run_infer_type(z)

        assert new_sparse_indices_infer_type.checked_type.dtype == sparse_indices_np.dtype
        assert new_sparse_values_infer_type.checked_type.dtype == sparse_indices_np.dtype
        assert empty_row_indicator_infer_type.checked_type.dtype == "bool"

        verify_func(
            executor_kind,
            func,
            [sparse_indices_np, sparse_values_np, dense_shape_np, default_value_np],
            ref_res,
            [("llvm", tvm.cpu())],
        )

    verify_sparse_fill_empty_rows(
        sparse_indices.astype(dtype),
        sparse_values.astype(dtype),
        dense_shape.astype(dtype),
        default_value.astype(dtype),
    )


def test_dyn_copy():
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    mod = tvm.relay.fromtext(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(?, 3), int64]) -> Tensor[(?, 3), int64] {
          copy(%x)
        }
        """
    )
    x_data = np.random.rand(15, 3).astype("int64")
    expected = x_data
    check_on_vm(target, dev, [x_data], expected, mod)


def test_dyn_copy_scalar():
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    mod = tvm.relay.fromtext(
        """
        #[version = "0.0.5"]
        def @main(%x: int32, %y: Tensor[(?), int32]) -> Tensor[(?), int32] {
          %0 = copy(%x);
          %1 = expand_dims(%0, axis=0);
          %2 = (%y, %1);
          concatenate(%2)
        }
        """
    )
    x_data = 3
    y_data = np.random.rand(7).astype("int32")
    expected = np.concatenate((y_data, np.expand_dims(x_data, axis=0)))
    check_on_vm(target, dev, [x_data, y_data], expected, mod)


def test_dyn_cast():
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()
    mod = tvm.relay.fromtext(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(?, 3), int64]) -> Tensor[(?, 3), int32] {
          cast(%x, dtype="int32")
        }
        """
    )
    x_data = np.random.rand(15, 3).astype("int64")
    expected = x_data.astype("int32")
    check_on_vm(target, dev, [x_data], expected, mod)


if __name__ == "__main__":
    tvm.testing.main()
