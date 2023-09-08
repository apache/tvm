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
import tvm.testing
from tvm import dlight as dl
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.runtime.relax_vm import VirtualMachine
from tvm.script import relax as R


def test_init():
    devices = [0, 1]

    sess = di.ThreadedSession(num_workers=len(devices))
    sess.init_ccl("nccl", *devices)


def test_allreduce():
    devices = [0, 1]
    array_1 = np.arange(12, dtype="float32").reshape(3, 4)
    array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)

    sess = di.ThreadedSession(num_workers=len(devices))
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
        dst_array = sess.empty((3, 4), "float32")
        sess.allreduce(d_array, dst_array, op=op)
        result = dst_array.debug_get_from_remote(0).numpy()
        expected = np_op(array_1, array_2)
        np.testing.assert_equal(result, expected)


def test_broadcast_from_worker0():
    devices = [0, 1]
    array = np.arange(12, dtype="float32").reshape(3, 4)

    sess = di.ThreadedSession(num_workers=len(devices))
    sess.init_ccl("nccl", *devices)
    d_array = sess.empty((3, 4), "float32")
    d_array.debug_copy_from(0, array)
    dst_array = sess.empty((3, 4), "float32")
    sess.broadcast_from_worker0(d_array, dst_array)
    result = dst_array.debug_get_from_remote(1).numpy()
    np.testing.assert_equal(result, array)


def test_scatter():
    devices = [0, 1]
    array = np.arange(36, dtype="float32").reshape(3, 4, 3)

    sess = di.ThreadedSession(num_workers=len(devices))
    sess.init_ccl("nccl", *devices)
    d_src = sess.empty((3, 4, 3), "float32")
    d_dst = sess.empty((3, 3, 2), "float32")

    d_src.debug_copy_from(0, array)
    sess.scatter_from_worker0(d_src, d_dst)

    np.testing.assert_equal(
        d_dst.debug_get_from_remote(0).numpy(),
        array.flat[:18].reshape(3, 3, 2),
    )
    np.testing.assert_equal(
        d_dst.debug_get_from_remote(1).numpy(),
        array.flat[18:].reshape(3, 3, 2),
    )


def test_gather():
    num_workers = 2
    devices = [1, 2]
    array = np.arange(36, dtype="float32")

    sess = di.ThreadedSession(num_workers=num_workers)
    sess.init_ccl("nccl", *devices)
    d_src = sess.empty((3, 3, 2), "float32")
    d_dst = sess.empty((3, 4, 3), "float32")

    d_src.debug_copy_from(0, array[:18])
    d_src.debug_copy_from(1, array[18:])

    sess.gather_to_worker0(d_src, d_dst)

    np.testing.assert_equal(
        d_dst.debug_get_from_remote(0).numpy(),
        array.reshape(3, 4, 3),
    )


def test_mlp():  # pylint: disable=too-many-locals
    devices = [0, 1]

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
                broadcast_x: R.Tensor((128, 128), "float32") = R.ccl.broadcast_from_worker0(x)
                lv0: R.Tensor((128, 64), "float32") = R.matmul(broadcast_x, W1)
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

        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        mod = sess.load_vm_module(path)

        d_X = sess.empty((128, 128), "float32")
        d_W1 = sess.empty((128, 64), "float32")
        d_W2 = sess.empty((64, 128), "float32")

        d_X.debug_copy_from(0, X)
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


def test_attention():  # pylint: disable=too-many-locals,too-many-statements
    devices = [0, 1]

    # pylint: disable=invalid-name
    @tvm.script.ir_module
    class Attention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 512), "float32"),
            Wk: R.Tensor((128, 512), "float32"),
            Wv: R.Tensor((128, 512), "float32"),
            Wo: R.Tensor((512, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                # q
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                R.output(lv16)
            return lv16

    @tvm.script.ir_module
    class ShardedAttention:  # pylint: disable=too-few-public-methods
        @R.function
        def main(  # pylint: disable=too-many-locals
            x: R.Tensor((1, 10, 128), "float32"),
            Wq: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wk: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wv: R.Tensor((128, 256), "float32"),  # shard along axis 1
            Wo: R.Tensor((256, 128), "float32"),  # shard along axis 0
        ) -> R.Tensor((128, 128), "float32"):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                broadcast_x: R.Tensor((1, 10, 128), "float32") = R.ccl.broadcast_from_worker0(x)
                # q
                lv0: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wq)
                lv1: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv0, [1, 10, 4, 64])
                lv2: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                # k
                lv3: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wk)
                lv4: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv3, [1, 10, 4, 64])
                lv5: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                # v
                lv6: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wv)
                lv7: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv6, [1, 10, 4, 64])
                lv8: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                # softmax(q @ k / sqrt(dk))
                lv9: R.Tensor((1, 4, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 4, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 4, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 4, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                # attn_weight @ v
                lv13: R.Tensor((1, 4, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 4, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 256), "float32") = R.reshape(lv14, [1, 10, 256])
                # attn_output @ o
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                lv17: R.Tensor((1, 10, 128), "float32") = R.ccl.allreduce(lv16, "sum")
                R.output(lv17)
            return lv17

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
    X = np.random.randn(1, 10, 128).astype("float32")
    Wq = np.random.randn(128, 512).astype("float32")
    Wk = np.random.randn(128, 512).astype("float32")
    Wv = np.random.randn(128, 512).astype("float32")
    Wo = np.random.randn(512, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(Attention, target), device=tvm.cuda(0))["main"](
        tvm.nd.array(X, device=tvm.cuda(0)),
        tvm.nd.array(Wq, device=tvm.cuda(0)),
        tvm.nd.array(Wk, device=tvm.cuda(0)),
        tvm.nd.array(Wv, device=tvm.cuda(0)),
        tvm.nd.array(Wo, device=tvm.cuda(0)),
    ).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        relax_build(ShardedAttention, target).export_library(path)

        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        mod = sess.load_vm_module(path)

        d_X = sess.empty((1, 10, 128), "float32")
        d_Wq = sess.empty((128, 256), "float32")
        d_Wk = sess.empty((128, 256), "float32")
        d_Wv = sess.empty((128, 256), "float32")
        d_Wo = sess.empty((256, 128), "float32")

        d_X.debug_copy_from(0, X)
        d_Wq.debug_copy_from(0, Wq[:, :256])
        d_Wq.debug_copy_from(1, Wq[:, 256:])
        d_Wk.debug_copy_from(0, Wk[:, :256])
        d_Wk.debug_copy_from(1, Wk[:, 256:])
        d_Wv.debug_copy_from(0, Wv[:, :256])
        d_Wv.debug_copy_from(1, Wv[:, 256:])
        d_Wo.debug_copy_from(0, Wo[:256, :])
        d_Wo.debug_copy_from(1, Wo[256:, :])
        d_Y = mod["main"](d_X, d_Wq, d_Wk, d_Wv, d_Wo)
        Y_result = tvm.nd.empty((1, 10, 128), "float32", device=tvm.cuda(0))
        sess.copy_from_worker_0(Y_result, d_Y)
        sess.sync_worker_0()
        Y_result = Y_result.numpy()
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()

