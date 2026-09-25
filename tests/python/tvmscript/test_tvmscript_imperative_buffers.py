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
# pylint: disable=invalid-name, missing-docstring
"""Concrete-buffer indexing and imperative operator regression tests."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import te, tirx, topi
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.script import ir_builder as T


@pytest.mark.parametrize(
    "shape,index,expected",
    [
        ((6,), 7, [7]),
        ((2, 3), 4, [1, 1]),
        ((2, 3), 7, [2, 1]),
        ((2, 3), -1, [-1, 2]),
        ((2, 3), (1, 2), [1, 2]),
        ((2, 3, 4), 17, [1, 1, 1]),
        ((), 0, []),
        ((), (), []),
    ],
)
def test_concrete_buffer_indices(shape, index, expected):
    buffer = tirx.decl_buffer(shape, "float32")
    assert T.buffer_indices(buffer, index) == expected


def test_symbolic_buffer_indices():
    m, n, k = [tirx.Var(name, "int32") for name in ("m", "n", "k")]
    buffer = tirx.decl_buffer((m, n), "float32")
    actual = T.buffer_indices(buffer, k)
    for index, expected in zip(actual, [k // n, k % n]):
        tvm.ir.assert_structural_equal(index, expected)


@pytest.mark.parametrize("layout", [None, TileLayout(S[(2, 3) : (1, 2)])])
def test_flat_buffer_store_preserves_identity_and_emits_once(layout):
    buffer = tirx.decl_buffer((2, 3), "float32", strides=(5, 1), elem_offset=2, layout=layout)
    indices = T.buffer_indices(buffer, 4)
    load = buffer[indices]
    assert load.source.same_as(buffer)
    with IRBuilder() as ib:
        receipt = T.buffer_store(buffer, load + 1, indices)
        T.emit_(receipt)
    store = ib.get()
    assert isinstance(store, tirx.BufferStore)
    assert store.buffer.same_as(buffer)
    assert list(store.indices) == [1, 1]
    assert store.value.a.source.same_as(buffer)


def _check_stores(output):
    outputs = output if isinstance(output, tuple | list | tvm.ir.Array) else [output]
    stores = []

    for tensor in outputs:
        tvm_ffi.structural_walk(
            tensor.op.body,
            lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None,
        )
    assert stores
    assert all(len(store.indices) == len(store.buffer.shape) for store in stores)


@pytest.mark.parametrize("mode", ["update", "add", "mul", "min", "max"])
def test_imperative_scatter_nd(mode):
    _check_stores(
        topi.scatter_nd(
            te.placeholder((2, 3), "float32"),
            te.placeholder((1, 2), "int32"),
            te.placeholder((2, 3), "float32"),
            mode,
        )
    )


@pytest.mark.parametrize("mode", ["update", "add", "mul", "mean", "min", "max"])
def test_imperative_scatter_elements(mode):
    _check_stores(
        topi.scatter_elements(
            te.placeholder((2, 3), "float32"),
            te.placeholder((2, 2), "int32"),
            te.placeholder((2, 2), "float32"),
            axis=1,
            reduction=mode,
        )
    )


@pytest.mark.parametrize("accumulate", [False, True])
def test_imperative_index_put(accumulate):
    _check_stores(
        topi.index_put(
            te.placeholder((2, 3), "float32"),
            [te.placeholder((2,), "int32"), te.placeholder((2,), "int32")],
            te.placeholder((2,), "float32"),
            accumulate=accumulate,
        )
    )


def test_imperative_search_scan_signal_and_unique():
    data = te.placeholder((2, 4), "float32")
    _check_stores(topi.searchsorted(data, te.placeholder((2, 3), "float32")))
    _check_stores(topi.cumsum(data, axis=1))
    _check_stores(topi.cumprod(data, axis=None))
    _check_stores(topi.signal.dft(data, data, inverse=False))
    _check_stores(
        topi.signal.stft(
            data,
            4,
            1,
            4,
            te.placeholder((4,), "float32"),
            False,
            True,
            (2, 3, 1, 2),
        )
    )
    from tvm.topi.unique import _calc_adjacent_diff

    _check_stores(_calc_adjacent_diff(te.placeholder((4,), "int32")))


def test_imperative_sparse_reshape():
    _check_stores(
        topi.sparse_reshape(
            te.placeholder((3, 2), "int64"),
            te.placeholder((2,), "int64"),
            te.placeholder((2,), "int64"),
            (3, 2),
            (2,),
        )
    )


def test_imperative_nms():
    data = te.placeholder((2, 4, 6), "float32")
    valid_count, boxes, indices = topi.vision.get_valid_counts(data)
    _check_stores([valid_count, boxes, indices])
    _check_stores(
        topi.vision.non_max_suppression(
            data,
            te.placeholder((2,), "int32"),
            te.placeholder((2, 4), "int32"),
            max_output_size=3,
            return_indices=False,
        )
    )


@pytest.mark.parametrize(
    "kind", ["sort", "argsort", "searchsorted", "scan", "scatter_nd", "scatter_elements"]
)
def test_gpu_imperative_buffers(kind):
    data = te.placeholder((2, 4), "float32")
    with tvm.target.Target("cuda"):
        if kind == "sort":
            result = topi.gpu.sort(data)
        elif kind == "argsort":
            result = topi.gpu.argsort(data)
        elif kind == "searchsorted":
            result = topi.gpu.searchsorted(data, te.placeholder((2, 3), "float32"))
        elif kind == "scan":
            from tvm.topi.gpu.scan import exclusive_scan_ir

            body = exclusive_scan_ir(tirx.decl_buffer((2, 4)), tirx.decl_buffer((2, 4)))
            assert isinstance(body, tirx.Stmt)
            return
        elif kind == "scatter_nd":
            result = topi.gpu.scatter_nd(
                data, te.placeholder((1, 2), "int32"), te.placeholder((2, 4)), "add"
            )
        else:
            result = topi.gpu.scatter_elements(
                data,
                te.placeholder((2, 3), "int32"),
                te.placeholder((2, 3)),
                axis=1,
                reduction="add",
            )
    _check_stores(result)


def test_python_bool_control_conditions():
    with IRBuilder() as ib:
        with T.prim_func():
            with T.if_(True):
                with T.then_():
                    T.evaluate(T.call_extern("int32", "true_branch"))
            with T.while_(False):
                T.evaluate(T.call_extern("int32", "false_loop"))
            T.assert_(True, "ok")
    statements = list(ib.get().body.seq)
    assert isinstance(statements[0], tirx.IfThenElse)
    assert isinstance(statements[1], tirx.While)
    assert isinstance(statements[2], tirx.AssertStmt)
    assert [bool(stmt.condition.value) for stmt in statements] == [True, False, True]
    assert all(str(stmt.condition.ty) == "bool" for stmt in statements)


@pytest.mark.parametrize("declare", [False, True])
def test_concrete_mutable_scalar(declare):
    with IRBuilder() as ib:
        with T.prim_func():
            if declare:
                owner = T.alloc_buffer((4,), "int32", scope="local")
                scalar = T.decl_scalar("int32", owner.data, "local", elem_offset=2)
            else:
                scalar = T.alloc_scalar("int32", "local")
            assert type(scalar) is tvm.ir.TensorLoad
            named = T.decl_mutable_cell_(scalar, name="counter")
            assert named.same_as(scalar)
            assert scalar.source.name == "counter"
            T.emit_(T.set_mutable_cell_(scalar, 3))
            T.emit_(T.set_mutable_cell_(scalar, scalar + 1))
    stores = []
    tvm_ffi.structural_walk(
        ib.get().body,
        lambda node: stores.append(node) if isinstance(node, tirx.BufferStore) else None,
    )
    assert len(stores) == 2
    assert all(store.buffer.same_as(scalar.source) for store in stores)
    if declare:
        declaration = next(stmt for stmt in ib.get().body.seq if isinstance(stmt, tirx.DeclBuffer))
        assert declaration.buffer.same_as(scalar.source)
        tvm.ir.assert_structural_equal(declaration.data, owner.data)
        assert scalar.source.elem_offset == 2


@pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize(
    "kind",
    [
        "searchsorted",
        "scan",
        "scatter",
        "scatter_elements",
        "index_put",
        "dft",
        "sparse_reshape",
        "unique",
    ],
)
def test_imperative_buffers_numerical(kind):
    data = np.arange(6, dtype="float32").reshape(2, 3)
    if kind == "searchsorted":
        arrays = [data, np.array([[0.5, 2, 3], [2, 4, 6]], dtype="float32")]

        def build(x, y):
            return topi.searchsorted(x, y, right=True)

        expected = [
            np.stack([np.searchsorted(row, values, side="right") for row, values in zip(*arrays)])
        ]
    elif kind == "scan":
        arrays = [data]

        def build(x):
            return topi.cumsum(x, axis=1)

        expected = [np.cumsum(data, axis=1)]
    elif kind == "scatter":
        arrays = [data, np.array([[1, 0]], dtype="int32"), np.ones((2, 3), dtype="float32")]

        def build(x, i, u):
            return topi.scatter_nd(x, i, u, "add")

        expected = [data + 1]
    elif kind == "scatter_elements":
        arrays = [data, np.array([[0, 2], [2, 0]], dtype="int32"), np.ones((2, 2), dtype="float32")]

        def build(x, i, u):
            return topi.scatter_elements(x, i, u, axis=1, reduction="add")

        result = data.copy()
        np.add.at(result, (np.arange(2)[:, None], arrays[1]), arrays[2])
        expected = [result]
    elif kind == "index_put":
        arrays = [
            data,
            np.array([0, 1], dtype="int32"),
            np.array([2, 0], dtype="int32"),
            np.array([10, 20], dtype="float32"),
        ]

        def build(x, i, j, v):
            return topi.index_put(x, [i, j], v, accumulate=True)

        result = data.copy()
        np.add.at(result, (arrays[1], arrays[2]), arrays[3])
        expected = [result]
    elif kind == "dft":
        arrays = [data, np.zeros_like(data)]

        def build(x, y):
            return topi.signal.dft(x, y, inverse=False)

        result = np.fft.fft(data, axis=-1)
        expected = [result.real, result.imag]
    elif kind == "sparse_reshape":
        arrays = [
            np.array([[0, 0], [1, 0], [1, 2]], dtype="int64"),
            np.array([2, 3], dtype="int64"),
            np.array([3, 2], dtype="int64"),
        ]

        def build(i, old, new):
            return topi.sparse_reshape(i, old, new, (3, 2), (2,))

        expected = [np.array([[0, 0], [1, 1], [2, 1]], dtype="int64"), arrays[2]]
    else:
        from tvm.topi.unique import _calc_adjacent_diff

        arrays = [np.array([3, 5, 5, 8], dtype="int32")]
        build = _calc_adjacent_diff
        expected = [np.array([0, 2, 0, 3], dtype="int32")]
    inputs = [te.placeholder(array.shape, str(array.dtype)) for array in arrays]
    outputs = build(*inputs)
    outputs = list(outputs) if isinstance(outputs, list | tuple | tvm.ir.Array) else [outputs]
    executable = tvm.compile(te.create_prim_func([*inputs, *outputs]), target="llvm")
    actual = [tvm.runtime.empty(tuple(int(dim) for dim in out.shape), out.dtype) for out in outputs]
    executable(*[tvm.runtime.tensor(array) for array in arrays], *actual)
    for result, reference in zip(actual, expected):
        tvm.testing.assert_allclose(result.numpy(), reference, atol=1e-5, rtol=1e-5)
