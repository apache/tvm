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

import functools
import sys
import pytest

import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.tir.stmt_functor import post_order_visit
from tvm.driver.build_module import schedule_to_module

dtype = tvm.testing.parameter("int32")


def flatten_all_indices(preflatten_shape):
    def mapping(*indices):
        output = 0
        for index, size in zip(indices, preflatten_shape):
            output = output * size + index
        return [output]

    return mapping


def unpack_flattened_indices(preflatten_shape):
    def mapping(i):
        output = []
        for dim in reversed(preflatten_shape):
            output.append(i % dim)
            i //= dim
        return output[::-1]

    return mapping


def traverse(s, op, callback):
    visited = set()

    def _traverse(op):
        if op in visited:
            return
        visited.add(op)
        for tensor in op.input_tensors:
            _traverse(tensor.op)
        callback(op)

    _traverse(op)


class TestCompareAgainstExplicitReshape:
    A_definition_style = tvm.testing.parameter(
        "explicit_reshape",
        "transform_layout",
    )
    B_definition_style = tvm.testing.parameter(
        "explicit_reshape",
        "transform_layout",
    )

    reordered_shape = tvm.testing.parameter((2, 3, 4))

    @tvm.testing.fixture
    def n_items(self, reordered_shape):
        return functools.reduce(lambda x, y: x * y, reordered_shape, 1)

    @tvm.testing.fixture
    def fphysical_layout(self, reordered_shape):
        return unpack_flattened_indices(reordered_shape)

    @tvm.testing.fixture
    def fcompute(self, A_definition_style, B_definition_style, reordered_shape, n_items, dtype):
        assert A_definition_style in ["explicit_reshape", "transform_layout"]
        assert B_definition_style in ["explicit_reshape", "transform_layout"]

        def func():
            if A_definition_style == "explicit_reshape":
                A_input = te.placeholder(shape=reordered_shape, name="A_input", dtype=dtype)
                A = te.compute(
                    shape=(n_items,),
                    fcompute=lambda i: A_input[
                        i // (reordered_shape[1] * reordered_shape[2]),
                        (i // reordered_shape[2]) % reordered_shape[1],
                        i % reordered_shape[2],
                    ],
                    name="A",
                )

            elif A_definition_style == "transform_layout":
                A = te.placeholder(shape=(n_items,), name="A", dtype=dtype)
                A_input = A

            B = te.compute(shape=A.shape, fcompute=lambda i: A[i], name="B")

            if B_definition_style == "explicit_reshape":
                B_output = te.compute(
                    shape=reordered_shape,
                    fcompute=lambda i, j, k: B[
                        i * reordered_shape[1] * reordered_shape[2] + j * reordered_shape[2] + k
                    ],
                    name="B_output",
                )
            elif B_definition_style == "transform_layout":
                B_output = B

            return A_input, B_output

        return func

    @tvm.testing.fixture
    def fschedule(self, A_definition_style, B_definition_style, fphysical_layout):
        def func(outs):
            outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
            s = te.create_schedule([x.op for x in outs])

            def callback(op):
                if (op.name == "A" and A_definition_style == "transform_layout") or (
                    op.name == "B" and B_definition_style == "transform_layout"
                ):
                    s[op].transform_layout(fphysical_layout)

            traverse(s, outs[0].op, callback)
            return s

        return func

    @tvm.testing.parametrize_targets("llvm")
    def test_external_reshape(
        self, target, dev, fcompute, fschedule, n_items, reordered_shape, dtype
    ):
        A, B = fcompute()
        s = fschedule(B)

        func = tvm.build(s, [A, B], target=target, name="copy_reshape")

        a_np = np.arange(n_items).reshape(reordered_shape).astype(dtype)
        b_np = np.arange(n_items).reshape(reordered_shape).astype(dtype)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.empty(b_np.shape, dtype=dtype, device=dev)

        func(a, b)

        tvm.testing.assert_allclose(b.numpy(), b_np)

    @tvm.testing.parametrize_targets("llvm")
    def test_internal_reshape(self, target, dev, n_items, reordered_shape, dtype, fphysical_layout):
        # The reshaping of the buffer gets flattened away in
        # StorageFlatten.  Therefore, testing the behavior by running only
        # ApplyLayoutTransforms.
        logical_shape = (n_items,)
        A = te.placeholder(logical_shape, name="A", dtype=dtype)
        B = te.compute(shape=logical_shape, fcompute=lambda i: A[i], name="B")
        C = te.compute(shape=logical_shape, fcompute=lambda i: B[i], name="C")

        s = te.create_schedule(C.op)
        s[B].transform_layout(fphysical_layout)

        mod = schedule_to_module(s, [A, C])
        body = mod["main"].body

        def walk_buffer_interactions(stmt, callback):
            buffer_classes = [
                tvm.tir.BufferLoad,
                tvm.tir.BufferStore,
                tvm.tir.BufferRealize,
            ]

            def inner(node):
                if (type(node) in buffer_classes) and node.buffer.name == "B":
                    callback(node)

            post_order_visit(stmt, inner)

        # All references to the buffer are the same object
        def check_references():
            buffer_object = None

            def inner(node):
                nonlocal buffer_object
                if buffer_object is None:
                    buffer_object = node.buffer
                else:
                    assert node.buffer.same_as(buffer_object)

            return inner

        # The buffer has the expected shape.
        def check_shape(expected_shape):
            def inner(node):
                assert tuple(node.buffer.shape) == expected_shape

            return inner

        # Before the transform, the buffer should be in the logical shape.
        walk_buffer_interactions(body, check_references())
        walk_buffer_interactions(body, check_shape(logical_shape))

        mod = tvm.tir.transform.ApplyLayoutTransforms()(mod)
        body = mod["main"].body

        # After the transform, the buffer should be in the physical shape.
        walk_buffer_interactions(body, check_references())
        walk_buffer_interactions(body, check_shape(reordered_shape))


class Test2DPhysicalLayout:
    transform_A = tvm.testing.parameter(
        "1d_A",
        "2d_A",
        "2d_rev_A",
        "3d_A",
    )
    transform_B = tvm.testing.parameter(
        "1d_B",
        "2d_B",
        "2d_rev_B",
        "3d_B",
    )

    @staticmethod
    def extract_logical_indices(stmt):
        output = {}

        # Since the for loops can be reordered by the layout
        # transformation, identify the loop corresponding to each
        # pre-transformation axis based on the iteration extent.
        def callback(node):
            if isinstance(node, tvm.tir.For):
                output[node.loop_var] = node.extent.value

        post_order_visit(stmt, callback)
        return sorted(output, key=output.get)

    def get_transform(self, name):
        name = name[:-2]
        if name == "1d":
            return None
        elif name == "2d":
            return lambda i, j, k: [i, j, te.AXIS_SEPARATOR, k]
        elif name == "2d_rev":
            return lambda i, j, k: [k, j, te.AXIS_SEPARATOR, i]
        elif name == "3d":
            return lambda i, j, k: [i, te.AXIS_SEPARATOR, j, te.AXIS_SEPARATOR, k]
        else:
            raise ValueError(f"Unknown transformation: {name}")

    def transform_indices(self, name, logical_shape, logical_index_vars):
        name = name[:-2]

        i, j, k = logical_index_vars

        if name == "1d":
            return [i * (logical_shape[1] * logical_shape[2]) + j * logical_shape[2] + k]
        elif name == "2d":
            return [i * logical_shape[1] + j, k]
        elif name == "2d_rev":
            return [k * logical_shape[1] + j, i]
        elif name == "3d":
            return [i, j, k]
        else:
            raise ValueError(f"Unknown transformation: {name}")

    def test_2d_physical(self, dtype, transform_A, transform_B):
        logical_shape = (2, 3, 4)
        A = te.placeholder(shape=logical_shape, dtype=dtype, name="A")
        B = te.compute(shape=A.shape, fcompute=lambda i, j, k: A[i, j, k], name="B")

        s = te.create_schedule(B.op)

        func = self.get_transform(transform_A)
        if func:
            s[A].transform_layout(func)

        func = self.get_transform(transform_B)
        if func:
            s[B].transform_layout(func)

        # If the two buffers are accessed with the same indices, CSE
        # will replace them with a Let binding.  Since this makes it
        # harder to test what the transformed indices are, disabling
        # the CSE pass for this test.
        with tvm.transform.PassContext(disabled_pass=["tir.CommonSubexprElimTIR"]):
            mod = tvm.lower(s, [A, B])

        logical_index_vars = self.extract_logical_indices(mod["main"].body)
        expected_indices_A = self.transform_indices(transform_A, logical_shape, logical_index_vars)
        expected_indices_B = self.transform_indices(transform_B, logical_shape, logical_index_vars)

        def callback(node):
            if type(node) in [tvm.tir.BufferLoad, tvm.tir.BufferStore]:
                name = node.buffer.name
                if name == "A":
                    expected_indices = expected_indices_A
                elif name == "B":
                    expected_indices = expected_indices_B
                else:
                    raise RuntimeError(f"Unexpected buffer: {name}")

                tvm.ir.assert_structural_equal(expected_indices, node.indices)

        post_order_visit(mod["main"].body, callback)


class TestTransformedSchedules:
    logical_shape = tvm.testing.parameter((4, 6, 40))

    transform_names = [
        None,
        "reverse",
        "flatten_all",
        "factor_last_by_4",
    ]

    transform_A = tvm.testing.parameter(by_dict={f"A_{t}": t for t in transform_names})
    transform_B = tvm.testing.parameter(
        by_dict={f"B_{t}": t for t in transform_names if t is not None}
    )

    after_transform = tvm.testing.parameter(None)

    def make_transform(self, logical_shape, transform_name):
        if transform_name is None:
            return lambda *indices: indices
        elif transform_name == "reverse":
            return lambda *indices: indices[::-1]
        elif transform_name == "flatten_all":
            return flatten_all_indices(logical_shape)
        elif transform_name == "factor_last_by_4":
            return lambda *indices, n: [*indices, n // 4, n % 4]
        else:
            raise NotImplementedError(f"Unknown transformation {transform_name}")

    def make_transformed_shape(self, logical_shape, transform_name):
        if transform_name is None:
            return logical_shape
        elif transform_name == "reverse":
            return logical_shape[::-1]
        elif transform_name == "flatten_all":
            num_elements = functools.reduce(lambda x, y: x * y, logical_shape, 1)
            return [num_elements]
        elif transform_name == "factor_last_by_4":
            *indices, n = logical_shape
            return [*indices, n // 4, 4]
        else:
            raise NotImplementedError(f"Unknown transformation {transform_name}")

    @tvm.testing.fixture
    def expected_loop_order(self, logical_shape, transform_B, after_transform):
        shape = self.make_transformed_shape(logical_shape, transform_B)

        if after_transform == "reorder":
            shape = shape[::-1]

        elif after_transform == "split":
            shape = [
                *shape[:-1],
                2,
                shape[-1] // 2,
            ]

        elif after_transform == "fuse":
            fused_size = shape[0] if transform_B == "flatten_all" else shape[0] * shape[1]
            shape = [fused_size, *shape[2:]]

        return shape

    @tvm.testing.fixture
    def schedule(self, logical_shape, dtype, transform_A, transform_B, after_transform):
        A = te.placeholder(shape=logical_shape, dtype=dtype, name="A")
        B = te.compute(shape=A.shape, fcompute=lambda i, j, k: A[i, j, k], name="B")

        s = te.create_schedule(B.op)

        if transform_A:
            s[A].transform_layout(self.make_transform(logical_shape, transform_A))

        iter_vars = s[B].transform_layout(self.make_transform(logical_shape, transform_B))
        iter_vars = list(iter_vars)

        if after_transform == "reorder":
            s[B].reorder(*iter_vars[::-1])

        elif after_transform == "split":
            s[B].split(iter_vars[-1], nparts=2)

        elif after_transform == "fuse":
            to_fuse = iter_vars[:2]
            s[B].fuse(*iter_vars[:2])

        return {
            "schedule": s,
            "tensors": [A, B],
            "iter_vars": iter_vars,
        }

    def compare_tir_loop_order(self, stmt, expected_loop_order):
        def collect_loops(node):
            output = []

            def callback(node):
                if isinstance(node, tvm.tir.For):
                    output.append(node)

            post_order_visit(node, callback)
            return output[::-1]

        loops = collect_loops(stmt)
        loop_order = [loop.extent for loop in loops]

        np.testing.assert_array_equal(loop_order, expected_loop_order)

    def test_tir_loop_order(self, schedule, expected_loop_order):
        func = tvm.lower(schedule["schedule"], schedule["tensors"])["main"]
        self.compare_tir_loop_order(func.body, expected_loop_order)

    def test_te_loop_order(self, schedule, expected_loop_order):
        s = schedule["schedule"]
        A, B = schedule["tensors"]
        iter_vars = schedule["iter_vars"]

        # No reduction axis, so all leaf_iter_vars are over the data
        # array, and should have the new iteration variables.
        extents = [int(iter_var.dom.extent) for iter_var in s[B].leaf_iter_vars]
        np.testing.assert_array_equal(extents, expected_loop_order)

        # layout_transform should return the new iteration variables.
        extents = [int(iter_var.dom.extent) for iter_var in iter_vars]
        np.testing.assert_array_equal(extents, expected_loop_order)

    @pytest.mark.parametrize("after_transform", ["reorder", "split", "fuse"])
    def test_use_transformed_axes(
        self, schedule, expected_loop_order, transform_A, transform_B, after_transform
    ):
        s = schedule["schedule"]
        A, B = schedule["tensors"]

        func = tvm.lower(s, [A, B])["main"]
        self.compare_tir_loop_order(func.body, expected_loop_order)


class TestTransformCache:
    A_size = tvm.testing.parameter(16)

    transform_A = tvm.testing.parameter(by_dict={"transformA": True, "": False})
    transform_B = tvm.testing.parameter(by_dict={"transformB": True, "": False})
    cache_A = tvm.testing.parameter(by_dict={"cacheA": True, "": False})
    cache_B = tvm.testing.parameter(by_dict={"cacheB": True, "": False})

    @tvm.testing.fixture
    def schedule_args(self, target, A_size, transform_A, transform_B, cache_A, cache_B, dtype):
        A = te.placeholder(shape=[A_size], dtype=dtype, name="A")
        B = te.compute(A.shape, lambda i: A[i], name="B")
        s = te.create_schedule(B.op)

        requires_thread_bind = "gpu" in tvm.target.Target(target).keys
        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")
        thread_z = te.thread_axis("threadIdx.z")

        if cache_A:
            AA = s.cache_read(A, "shared", [B])
            if requires_thread_bind:
                s[AA].bind(AA.op.axis[0], thread_x)

        if cache_B:
            BB = s.cache_write(B, "shared")
            if requires_thread_bind:
                s[BB].bind(BB.op.axis[0], thread_y)

        if transform_A:
            A_axis = s[A].transform_layout(lambda i: [i // 4, i % 4])

        if transform_B:
            B_axis = s[B].transform_layout(lambda i: [i // 4, i % 4])
        else:
            B_axis = B.op.axis

        if requires_thread_bind:
            s[B].bind(B_axis[0], thread_z)

        return [s, [A, B]]

    @tvm.testing.fixture
    def ref_data(self, A_size, dtype, transform_A, transform_B):
        a_np = (100 * np.random.uniform(size=A_size)).astype(dtype)
        b_np = a_np

        if transform_A:
            a_np = a_np.reshape((-1, 4))

        if transform_B:
            b_np = b_np.reshape((-1, 4))

        return a_np, b_np

    def test_lower(self, schedule_args):
        tvm.lower(*schedule_args)

    def test_execute(self, target, dev, schedule_args, ref_data, dtype):
        func = tvm.build(*schedule_args, target=target)

        a_np, b_np = ref_data
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.empty(b_np.shape, dtype=dtype, device=dev)

        func(a, b)

        if "int" in dtype:
            np.testing.assert_equal(b.numpy(), b_np)
        else:
            tvm.testing.assert_allclose(b.numpy(), b_np)


def test_transform_with_reduction():
    # To trigger this failure mode, the computation must use a
    # reduction axis,
    A = te.placeholder([16, 32, 64], dtype="float32", name="A")
    k = te.reduce_axis((0, A.shape[-1]), name="k")
    B = te.compute(A.shape[:-1], lambda i, j: te.sum(A[i, j, k], axis=[k]))
    s = te.create_schedule(B.op)

    # And the output of the computation must have a layout
    # transformation applied.
    s[B].transform_layout(lambda i, j: [j, i])

    # When present, the failure occurred during tvm.lower, during the
    # call to `tvm::te::PassDownBitMaskOr`.
    tvm.lower(s, [A, B])


shape, transform = tvm.testing.parameters(
    ([1, 8], lambda n, i: [i, n]),
    ([1, 1, 8], lambda i, j, k: [j, te.AXIS_SEPARATOR, i, k]),
    ([1, 1, 8], lambda i, j, k: [i, te.AXIS_SEPARATOR, j, k]),
)


def test_size_one_buffer(shape, transform):
    # This test is to catch a failure mode that occurred if a
    # transformation were applied to a te.compute buffer, and one of
    # the dimensions of the buffer was 1.  Prior to bugfix,
    # arith::DetectIterMap would fold the variable as a constant,
    # causing an error when attempting to solve for the variable using
    # arith::InverseAffineIterMap.

    dtype = "int8"
    A = te.placeholder(shape, dtype, name="A")
    B = te.compute(
        shape=A.shape,
        fcompute=lambda *indices: A[indices].astype(dtype),
        name="B",
    )
    s = te.create_schedule(B.op)

    # If layout transformation is on the output buffer, and any
    # dimension of the output buffer is 1, failure occurs in
    # CheckFusePattern.
    s[B].transform_layout(transform)


def test_non_divisible_transform_raises_error():
    A = te.placeholder([1, 3, 8, 8])
    B = te.compute(A.shape, lambda *indices: A[indices])
    s = te.create_schedule(B.op)

    transform = lambda n, c, h, w: [n, c // 4, h, w, c % 4]
    # Error occurs here, because the transformation would introduce
    # padding.  Padded transforms are supported in TIR-based
    # schedules.
    with pytest.raises(tvm.TVMError):
        s[B].transform_layout(transform)


if __name__ == "__main__":
    tvm.testing.main()
