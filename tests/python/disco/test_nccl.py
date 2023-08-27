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
"""Tests for NCCL"""
import tempfile

import numpy as np

import tvm
from tvm import dlight as dl
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.runtime.relax_vm import VirtualMachine
from tvm.script import relax as R


def test_init():
    num_workers = 2
    devices = [1, 2]

    sess = di.ThreadedSession(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)


def test_allreduce():
    num_workers = 2
    devices = [1, 2]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)

    sess = di.ThreadedSession(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)
    d_array = sess.empty((3, 4), "float32")
    d_array.debug_copy_from(0, array_1)
    d_array.debug_copy_from(1, array_2)
    for op, np_op in [  # pylint: disable=invalid-name
        ("sum", np.add),
        ("prod", np.multiply),
        ("min", np.minimum),
        ("max", np.maximum),
        ("avg", lambda a, b: (a + b) * 0.5),
    ]:
        result = sess.allreduce(d_array, op=op)
        result = result.debug_get_from_remote(0).numpy()
        expected = np_op(array_1, array_2)
        np.testing.assert_equal(result, expected)


def test_broadcast_from_zero():
    num_workers = 2
    devices = [1, 2]
    array = np.arange(12, dtype="float32").reshape(3, 4)

    sess = di.ThreadedSession(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)
    d_array = sess.empty((3, 4), "float32")
    d_array.debug_copy_from(0, array)
    sess.broadcast_from_worker0(d_array)
    result = d_array.debug_get_from_remote(1).numpy()
    np.testing.assert_equal(result, array)


def test_mlp():  # pylint: disable=too-many-locals
    num_workers = 2
    devices = [1, 2]

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class MLP:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 128), "float32"),
            W2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((128, 128), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 128), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                R.output(lv2)
            return lv2

    @tvm.script.ir_module
    class ShardedMLP:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            x: R.Tensor((128, 128), "float32"),
            W1: R.Tensor((128, 64), "float32"),  # shard along axis 1
            W2: R.Tensor((64, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((128, 64), "float32") = R.matmul(x, W1)
                lv1: R.Tensor((128, 64), "float32") = R.nn.gelu(lv0)
                lv2: R.Tensor((128, 128), "float32") = R.matmul(lv1, W2)
                lv3: R.Tensor((128, 128), "float32") = R.ccl.allreduce(lv2, "sum")
                R.output(lv3)
            return lv3

    # pylint: enable=invalid-name
    target = tvm.target.Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
            "registers_per_block": 65536,
            "arch": "sm_80",
        }
    )

    def relax_build(mod, target):
        with target:
            mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return rx.build(mod, target="cuda")

    # pylint: disable=invalid-name
    X = np.random.randn(128, 128).astype("float32")
    W1 = np.random.randn(128, 128).astype("float32")
    W2 = np.random.randn(128, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(MLP, target), device=tvm.cuda(0))["main"](
        tvm.nd.array(X, device=tvm.cuda(0)),
        tvm.nd.array(W1, device=tvm.cuda(0)),
        tvm.nd.array(W2, device=tvm.cuda(0)),
    ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        relax_build(ShardedMLP, target).export_library(path)

        sess = di.ThreadedSession(num_workers=num_workers)
        sess.init_ccl("nccl", *devices)
        mod = sess.load_vm_module(path)

        d_X = sess.empty((128, 128), "float32")
        d_W1 = sess.empty((128, 64), "float32")
        d_W2 = sess.empty((64, 128), "float32")

        d_X.debug_copy_from(0, X)
        d_X.debug_copy_from(1, X)
        d_W1.debug_copy_from(0, W1[:, :64])
        d_W1.debug_copy_from(1, W1[:, 64:])
        d_W2.debug_copy_from(0, W2[:64, :])
        d_W2.debug_copy_from(1, W2[64:, :])
        d_Y = mod["main"](d_X, d_W1, d_W2)
        Y_result = tvm.nd.empty((128, 128), "float32", device=tvm.cuda(0))
        sess.copy_from_worker_0(Y_result, d_Y)
        sess.sync_worker_0()
        Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_init()
    test_broadcast_from_zero()
    test_allreduce()
    test_mlp()
