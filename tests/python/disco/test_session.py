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
"""Basic tests for a Disco session"""
# pylint: disable=missing-docstring
import tempfile

import numpy as np

import tvm
from tvm import relax as rx
from tvm._ffi import register_func
from tvm.runtime import ShapeTuple, String
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def _numpy_to_worker_0(sess: di.Session, np_array: np.array, device):
    x_array = sess.empty(np_array.shape, "float32", device=device)
    host_array = tvm.nd.array(np_array, device=device)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def _numpy_from_worker_0(sess: di.Session, remote_array, shape, dtype):
    host_array = tvm.nd.empty(shape, dtype, device=tvm.cpu())
    sess.copy_from_worker_0(host_array, remote_array)
    sess.sync_worker_0()
    return host_array.numpy()


def test_int():
    num_workers = 4

    @register_func("tests.disco.add_one", override=True)
    def add_one(x: int) -> int:  # pylint: disable=invalid-name
        return x + 1

    sess = di.ThreadedSession(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.add_one")
    result: di.DRef = func(1)

    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == 2


def test_float():
    num_workers = 4

    @register_func("tests.disco.add_one_float", override=True)
    def add_one(x: float):  # pylint: disable=invalid-name
        return x + 0.5

    sess = di.ThreadedSession(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.add_one_float")
    result: di.DRef = func(1.5)

    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == 2.0


def test_ndarray():
    num_workers = 4

    @register_func("tests.disco.add_one_ndarray", override=True)
    def add_one(x: tvm.runtime.NDArray) -> tvm.runtime.NDArray:  # pylint: disable=invalid-name
        return tvm.nd.array(x.numpy() + 1)

    device = tvm.cpu(0)
    x_np = np.arange(6).astype("float32").reshape([2, 3])
    y_np = np.arange(6).astype("float32").reshape([2, 3]) + 1

    sess = di.ThreadedSession(num_workers=num_workers)
    x_disc = _numpy_to_worker_0(sess, x_np, device=device)
    y_disc = sess.get_global_func("tests.disco.add_one_ndarray")(x_disc)
    y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
    np.testing.assert_equal(y_nd, y_np)


def test_string():
    num_workers = 4

    @register_func("tests.disco.str", override=True)
    def my_str_func(x: str):  # pylint: disable=invalid-name
        return x + "_suffix"

    sess = di.ThreadedSession(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.str")
    result: di.DRef = func("hello")

    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == "hello_suffix"


def test_string_obj():
    num_workers = 4

    @register_func("tests.disco.str_obj", override=True)
    def my_str_func(x: String):  # pylint: disable=invalid-name
        assert isinstance(x, String)
        return String(x + "_suffix")

    sess = di.ThreadedSession(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.str_obj")
    result: di.DRef = func(String("hello"))

    for i in range(num_workers):
        value = result.debug_get_from_remote(i)
        assert isinstance(value, String)
        assert value == "hello_suffix"


def test_shape_tuple():
    num_workers = 4

    @register_func("tests.disco.shape_tuple", override=True)
    def my_str_func(x: ShapeTuple):  # pylint: disable=invalid-name
        assert isinstance(x, ShapeTuple)
        return ShapeTuple(list(x) + [4, 5])

    sess = di.ThreadedSession(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.shape_tuple")
    result: di.DRef = func(ShapeTuple([1, 2, 3]))

    for i in range(num_workers):
        value = result.debug_get_from_remote(i)
        assert isinstance(value, ShapeTuple)
        assert list(value) == [1, 2, 3, 4, 5]


def test_vm_module():
    num_workers = 4

    # pylint: disable=invalid-name
    @I.ir_module
    class TestMod:
        @T.prim_func
        def transpose(A: T.Buffer((8, 16), "float32"), B: T.Buffer((16, 8), "float32")):
            for i, j in T.grid(16, 8):
                with T.block("transpose"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def main(A: R.Tensor((8, 16), dtype="float32")) -> R.Tensor((16, 8), dtype="float32"):
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.transpose, (A,), out_sinfo=R.Tensor((16, 8), dtype="float32"))
                R.output(B)
            return B

    # pylint: enable=invalid-name
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        device = tvm.cpu()
        x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
        y_np = x_np.transpose()

        rx.build(TestMod, target="llvm").export_library(path)
        sess = di.ThreadedSession(num_workers=num_workers)
        mod = sess.load_vm_module(path, device=device)

        x_disc = _numpy_to_worker_0(sess, x_np, device=device)
        y_disc = mod["main"](x_disc)
        y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
        np.testing.assert_equal(y_nd, y_np)


def test_vm_multi_func():
    num_workers = 4

    # pylint: disable=invalid-name
    @I.ir_module
    class TestMod:
        @T.prim_func
        def t1(A: T.Buffer((8, 16), "float32"), B: T.Buffer((16, 8), "float32")):
            for i, j in T.grid(16, 8):
                with T.block("transpose"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @T.prim_func
        def t2(A: T.Buffer((16, 8), "float32"), B: T.Buffer((8, 16), "float32")):
            for i, j in T.grid(8, 16):
                with T.block("transpose"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def transpose_1(
            A: R.Tensor((8, 16), dtype="float32")
        ) -> R.Tensor((16, 8), dtype="float32"):
            R.func_attr({"global_symbol": "main"})
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.t1, (A,), out_sinfo=R.Tensor((16, 8), dtype="float32"))
                R.output(B)
            return B

        @R.function
        def transpose_2(
            A: R.Tensor((16, 8), dtype="float32")
        ) -> R.Tensor((8, 16), dtype="float32"):
            R.func_attr({"global_symbol": "main"})
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.t2, (A,), out_sinfo=R.Tensor((8, 16), dtype="float32"))
                R.output(B)
            return B

    # pylint: enable=invalid-name
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        device = tvm.cpu()
        x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
        y_np = x_np.transpose()

        rx.build(TestMod, target="llvm").export_library(path)
        sess = di.ThreadedSession(num_workers=num_workers)
        mod = sess.load_vm_module(path, device=device)

        x_disc = _numpy_to_worker_0(sess, x_np, device=device)
        y_disc = mod["transpose_1"](x_disc)
        z_disc = mod["transpose_2"](y_disc)
        y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
        z_nd = _numpy_from_worker_0(sess, z_disc, shape=x_np.shape, dtype=x_np.dtype)
        np.testing.assert_equal(y_nd, y_np)
        np.testing.assert_equal(z_nd, x_np)


if __name__ == "__main__":
    test_int()
    test_float()
    test_string()
    test_string_obj()
    test_shape_tuple()
    test_ndarray()
    test_vm_module()
    test_vm_multi_func()
