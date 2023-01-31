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
""" Support level6 operator test cases.
"""
import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.topi.testing import searchsorted_ref
import tvm.testing

executor_kind = tvm.testing.parameter("graph", "vm")


@tvm.testing.uses_gpu
def test_sort():
    def verify_sort(shape, axis, is_ascend, is_dyn=False, in_dtype="float32"):
        if is_dyn:
            x = relay.var("x", relay.TensorType([relay.Any()] * len(shape), in_dtype))
        else:
            x = relay.var("x", relay.TensorType(shape, in_dtype))
        z = relay.sort(x, axis=axis, is_ascend=is_ascend)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=shape).astype(in_dtype)
        if is_ascend:
            ref_res = np.sort(x_data, axis=axis)
        else:
            ref_res = -np.sort(-x_data, axis=axis)

        if is_dyn:
            backend = "vm"
        else:
            backend = "graph"
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(backend, mod=mod, device=dev, target=target).evaluate()(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    for is_dyn in [False, True]:
        verify_sort((2, 3, 4), axis=0, is_ascend=False, is_dyn=is_dyn)
        verify_sort((1, 4, 6), axis=1, is_ascend=True, is_dyn=is_dyn)
        verify_sort((3, 5, 6), axis=-1, is_ascend=False, is_dyn=is_dyn)
        verify_sort((3, 2000, 6), axis=1, is_ascend=False, is_dyn=is_dyn)
        verify_sort((1, 122640), axis=1, is_ascend=False, is_dyn=is_dyn)
        verify_sort((1, 122640), axis=1, is_ascend=False, is_dyn=is_dyn, in_dtype="float16")


@tvm.testing.uses_gpu
def test_argsort():
    def verify_argsort(shape, axis, is_ascend, dtype, is_dyn=False, in_dtype="float32"):
        if is_dyn:
            x = relay.var("x", relay.TensorType([relay.Any()] * len(shape), in_dtype))
        else:
            x = relay.var("x", relay.TensorType(shape, in_dtype))
        z = relay.argsort(x, axis=axis, is_ascend=is_ascend, dtype=dtype)
        func = relay.Function([x], z)
        x_data = np.random.uniform(size=shape).astype(in_dtype)
        if is_ascend:
            ref_res = np.argsort(x_data, axis=axis, kind="stable")
        else:
            ref_res = np.argsort(-x_data, axis=axis, kind="stable")

        if is_dyn:
            backend = "vm"
        else:
            backend = "graph"
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(backend, mod=mod, device=dev, target=target).evaluate()(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res.astype(dtype), rtol=1e-5)

    for is_dyn in [False, True]:
        for dtype in ["int32", "int64", "float32", "float64"]:
            verify_argsort((2, 3, 4), axis=0, is_ascend=False, dtype=dtype, is_dyn=is_dyn)
            verify_argsort((1, 4, 6), axis=1, is_ascend=True, dtype=dtype, is_dyn=is_dyn)
        dtype = "int32"
        verify_argsort((3, 5, 6), axis=-1, is_ascend=False, dtype=dtype, is_dyn=is_dyn)
        verify_argsort((3, 6000, 6), axis=1, is_ascend=False, dtype=dtype, is_dyn=is_dyn)
        verify_argsort((1000, 1, 1), axis=0, is_ascend=False, dtype=dtype, is_dyn=is_dyn)
        verify_argsort((1, 122640), axis=1, is_ascend=False, dtype=dtype, is_dyn=is_dyn)
        verify_argsort(
            (1, 122640), axis=1, is_ascend=False, dtype=dtype, is_dyn=is_dyn, in_dtype="float16"
        )


@tvm.testing.uses_gpu
def test_topk(executor_kind):
    def verify_topk(k, axis, ret_type, is_ascend, dtype, in_dtype="float32"):
        shape = (20, 100)
        x = relay.var("x", relay.TensorType(shape, in_dtype))
        out = relay.topk(x, k, axis, ret_type, is_ascend, dtype)
        if isinstance(out, relay.expr.TupleWrapper):
            out = out.astuple()
        func = relay.Function([x], out)
        np_data = np.random.uniform(size=shape).astype(in_dtype)
        if is_ascend:
            np_indices = np.argsort(np_data, axis=axis, kind="stable")
        else:
            np_indices = np.argsort(-np_data, axis=axis, kind="stable")
        kk = k if k >= 1 else shape[axis]
        if axis == 0:
            np_indices = np_indices[:kk, :]
            np_values = np.zeros(np_indices.shape).astype(in_dtype)
            for i in range(shape[1]):
                np_values[:, i] = np_data[np_indices[:, i], i]
        else:
            np_indices = np_indices[:, :kk]
            np_values = np.zeros(np_indices.shape).astype(in_dtype)
            for i in range(shape[0]):
                np_values[i, :] = np_data[i, np_indices[i, :]]
        np_indices = np_indices.astype(dtype)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                np_data
            )
            if ret_type == "both":
                tvm.testing.assert_allclose(op_res[0].numpy(), np_values)
                tvm.testing.assert_allclose(op_res[1].numpy(), np_indices)
            elif ret_type == "values":
                tvm.testing.assert_allclose(op_res.numpy(), np_values)
            else:
                tvm.testing.assert_allclose(op_res.numpy(), np_indices)

    np.random.seed(0)
    for k in [0, 1, 5]:
        for axis in [0, -1, 1]:
            for ret_type in ["both", "values", "indices"]:
                verify_topk(k, axis, ret_type, True, "int64")
                verify_topk(k, axis, ret_type, False, "float32")
                verify_topk(k, axis, ret_type, False, "int64", "float16")


@tvm.testing.uses_gpu
def test_searchsorted():
    def verify_searchsorted(right, dtype):
        shape = (8, 9, 10)
        values_shape = shape[:-1] + (10,)
        sorted_sequence = relay.var("sorted_sequence", relay.TensorType(shape, "float32"))
        values = relay.var("sorted_sequence", relay.TensorType(values_shape, "float32"))
        out = relay.searchsorted(sorted_sequence, values, right, dtype)
        func = relay.Function([sorted_sequence, values], out)
        sorted_sequence_np = np.sort(np.random.randn(*shape).astype("float32"), axis=-1)
        values_np = np.random.randn(*values_shape).astype("float32")
        np_indices = searchsorted_ref(sorted_sequence_np, values_np, right, dtype)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                sorted_sequence_np, values_np
            )
            np.testing.assert_equal(op_res.numpy(), np_indices)

    verify_searchsorted(False, "int32")
    verify_searchsorted(True, "int64")


if __name__ == "__main__":
    tvm.testing.main()
