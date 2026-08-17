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
"""Collective (cpuccl) tests over ThreadedSession, ProcessSession and SocketSession.
Reuse the test cases from test_ccl.py"""

import socket
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.runtime.vm import VirtualMachine
from tvm.script import relax as R

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)

_CCL = "cpuccl"
_SOCKET_SESSION_TESTER = None


def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class SocketSessionTester:
    """Run a disco SocketSession with one local node and remote nodes."""

    def __init__(self, num_workers, num_nodes=2, num_groups=1, build_ring=True):
        self.sess = None
        self.remote_nodes = []
        assert num_workers % num_nodes == 0
        num_workers_per_node = num_workers // num_nodes
        server_host = "localhost"
        server_port = _get_free_port()
        server_exc = []

        def start_server():
            try:
                self.sess = di.SocketSession(
                    num_nodes,
                    num_workers_per_node,
                    num_groups,
                    server_host,
                    server_port,
                    build_ring,
                )
            except Exception as exc:  # pylint: disable=broad-except
                server_exc.append(exc)

        thread = threading.Thread(target=start_server)
        thread.start()

        cmd = "tvm.exec.disco_remote_socket_session"
        for _i in range(num_nodes - 1):
            self.remote_nodes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        cmd,
                        server_host,
                        str(server_port),
                        str(num_workers_per_node),
                    ],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )

        thread.join()
        if server_exc:
            raise server_exc[0]

    # Bound at class creation: module globals may already be cleared when
    # __del__ runs during interpreter shutdown.
    _TIMEOUT_EXPIRED = subprocess.TimeoutExpired

    def __del__(self):
        try:
            # Shut down the session first so remote nodes can exit gracefully.
            if self.sess is not None:
                self.sess.shutdown()
        finally:
            for node in self.remote_nodes:
                try:
                    node.wait(timeout=10)
                except self._TIMEOUT_EXPIRED:
                    node.kill()
                    node.wait()


def create_socket_session(num_workers, build_ring=True):
    global _SOCKET_SESSION_TESTER
    # Rebind (not `del`) so the global stays defined if the constructor raises.
    _SOCKET_SESSION_TESTER = None
    _SOCKET_SESSION_TESTER = SocketSessionTester(num_workers, build_ring=build_ring)
    assert _SOCKET_SESSION_TESTER.sess is not None
    return _SOCKET_SESSION_TESTER.sess


_all_session_kinds = [di.ThreadedSession, di.ProcessSession, create_socket_session]


def _session_id(session_kind):
    return session_kind.__name__


def _run_with_ccl_session(session_kind, devices, func):
    sess = session_kind(num_workers=len(devices), build_ring=True)
    try:
        sess.init_ccl(_CCL, *devices)
        return func(sess)
    finally:
        sess.shutdown()


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_init(session_kind):
    devices = [0, 1]

    def run_test(_sess):
        pass

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_allreduce(session_kind):
    devices = [0, 1]

    def run_test(sess):
        array_1 = np.arange(12, dtype="float32").reshape(3, 4)
        array_2 = np.arange(start=1, stop=-11, step=-1, dtype="float32").reshape(3, 4)
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

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_allgather(session_kind):
    devices = [0, 1]

    def run_test(sess):
        array = np.arange(36, dtype="float32")
        d_src = sess.empty((3, 3, 2), "float32")
        d_dst = sess.empty((3, 4, 3), "float32")
        d_src.debug_copy_from(0, array[:18])
        d_src.debug_copy_from(1, array[18:])
        sess.allgather(d_src, d_dst)
        np.testing.assert_equal(d_dst.debug_get_from_remote(0).numpy(), array.reshape(3, 4, 3))
        np.testing.assert_equal(d_dst.debug_get_from_remote(1).numpy(), array.reshape(3, 4, 3))

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
@pytest.mark.parametrize("use_explicit_output", [True, False])
def test_broadcast(session_kind, use_explicit_output):
    devices = [0, 1]

    def run_test(sess):
        array = np.arange(12, dtype="float32").reshape(3, 4)
        if use_explicit_output:
            src_array = sess.empty((3, 4), "float32", worker0_only=True)
            src_array.debug_copy_from(0, array)
            dst_array = sess.empty((3, 4), "float32")
            sess.broadcast_from_worker0(src_array, dst_array)
        else:
            dst_array = sess.broadcast(array)

        result = dst_array.debug_get_from_remote(1).numpy()
        np.testing.assert_equal(result, array)

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
@pytest.mark.parametrize("use_explicit_output", [True, False])
def test_scatter(session_kind, use_explicit_output):
    devices = [0, 1]

    def run_test(sess):
        array = np.arange(36, dtype="float32").reshape(2, 6, 3)
        if use_explicit_output:
            d_src = sess.empty((2, 6, 3), "float32", worker0_only=True)
            d_dst = sess.empty((6, 3), "float32")
            d_src.debug_copy_from(0, array)
            sess.scatter_from_worker0(d_src, d_dst)
        else:
            d_dst = sess.scatter(array)

        np.testing.assert_equal(d_dst.debug_get_from_remote(0).numpy(), array[0, :, :])
        np.testing.assert_equal(d_dst.debug_get_from_remote(1).numpy(), array[1, :, :])

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_scatter_with_implicit_reshape(session_kind):
    devices = [0, 1]

    def run_test(sess):
        array = np.arange(36, dtype="float32").reshape(3, 4, 3)
        d_src = sess.empty((3, 4, 3), "float32", worker0_only=True)
        d_dst = sess.empty((3, 3, 2), "float32")
        d_src.debug_copy_from(0, array)
        sess.scatter_from_worker0(d_src, d_dst)

        np.testing.assert_equal(
            d_dst.debug_get_from_remote(0).numpy(), array.flat[:18].reshape(3, 3, 2)
        )
        np.testing.assert_equal(
            d_dst.debug_get_from_remote(1).numpy(), array.flat[18:].reshape(3, 3, 2)
        )

    _run_with_ccl_session(session_kind, devices, run_test)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_gather(session_kind):
    devices = [0, 1]

    def run_test(sess):
        array = np.arange(36, dtype="float32")
        d_src = sess.empty((3, 3, 2), "float32")
        d_dst = sess.empty((3, 4, 3), "float32", worker0_only=True)
        d_src.debug_copy_from(0, array[:18])
        d_src.debug_copy_from(1, array[18:])
        sess.gather_to_worker0(d_src, d_dst)
        np.testing.assert_equal(d_dst.debug_get_from_remote(0).numpy(), array.reshape(3, 4, 3))

    _run_with_ccl_session(session_kind, devices, run_test)


def relax_build(mod):
    target = tvm.target.Target("llvm")
    with target:
        mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
        return tvm.compile(mod, target=target)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_mlp(session_kind):  # pylint: disable=too-many-locals
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

    dev = tvm.cpu(0)
    X = np.random.randn(128, 128).astype("float32")
    W1 = np.random.randn(128, 128).astype("float32")
    W2 = np.random.randn(128, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(MLP), device=dev)["main"](
        tvm.runtime.tensor(X, device=dev),
        tvm.runtime.tensor(W1, device=dev),
        tvm.runtime.tensor(W2, device=dev),
    ).numpy()

    def run_test(sess):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.so"
            relax_build(ShardedMLP).export_library(path)

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
            Y_result = tvm.runtime.empty((128, 128), "float32", device=dev)
            sess.copy_from_worker_0(Y_result, d_Y)
            sess.sync_worker_0()
            return Y_result.numpy()

    Y_result = _run_with_ccl_session(session_kind, devices, run_test)
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("session_kind", _all_session_kinds, ids=_session_id)
def test_attention(session_kind):  # pylint: disable=too-many-locals,too-many-statements
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
                lv0: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wq)
                lv1: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv0, [1, 10, 8, 64])
                lv2: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                lv3: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wk)
                lv4: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv3, [1, 10, 8, 64])
                lv5: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                lv6: R.Tensor((1, 10, 512), "float32") = R.matmul(x, Wv)
                lv7: R.Tensor((1, 10, 8, 64), "float32") = R.reshape(lv6, [1, 10, 8, 64])
                lv8: R.Tensor((1, 8, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                lv9: R.Tensor((1, 8, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 8, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 8, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 8, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                lv13: R.Tensor((1, 8, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 8, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 512), "float32") = R.reshape(lv14, [1, 10, 512])
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
                lv0: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wq)
                lv1: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv0, [1, 10, 4, 64])
                lv2: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv1, [0, 2, 1, 3])
                lv3: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wk)
                lv4: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv3, [1, 10, 4, 64])
                lv5: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv4, [0, 2, 1, 3])
                lv6: R.Tensor((1, 10, 256), "float32") = R.matmul(broadcast_x, Wv)
                lv7: R.Tensor((1, 10, 4, 64), "float32") = R.reshape(lv6, [1, 10, 4, 64])
                lv8: R.Tensor((1, 4, 10, 64), "float32") = R.permute_dims(lv7, [0, 2, 1, 3])
                lv9: R.Tensor((1, 4, 64, 10), "float32") = R.permute_dims(lv5, [0, 1, 3, 2])
                lv10: R.Tensor((1, 4, 10, 10), "float32") = R.matmul(lv2, lv9)
                lv11: R.Tensor((1, 4, 10, 10), "float32") = R.multiply(
                    lv10, R.const(1 / 8, "float32")
                )
                lv12: R.Tensor((1, 4, 10, 10), "float32") = R.nn.softmax(lv11, axis=-1)
                lv13: R.Tensor((1, 4, 10, 64), "float32") = R.matmul(lv12, lv8)
                lv14: R.Tensor((1, 10, 4, 64), "float32") = R.permute_dims(lv13, [0, 2, 1, 3])
                lv15: R.Tensor((1, 10, 256), "float32") = R.reshape(lv14, [1, 10, 256])
                lv16: R.Tensor((1, 10, 128), "float32") = R.matmul(lv15, Wo)
                lv17: R.Tensor((1, 10, 128), "float32") = R.ccl.allreduce(lv16, "sum")
                R.output(lv17)
            return lv17

    dev = tvm.cpu(0)
    X = np.random.randn(1, 10, 128).astype("float32")
    Wq = np.random.randn(128, 512).astype("float32")
    Wk = np.random.randn(128, 512).astype("float32")
    Wv = np.random.randn(128, 512).astype("float32")
    Wo = np.random.randn(512, 128).astype("float32")
    Y_expected = VirtualMachine(relax_build(Attention), device=dev)["main"](
        tvm.runtime.tensor(X, device=dev),
        tvm.runtime.tensor(Wq, device=dev),
        tvm.runtime.tensor(Wk, device=dev),
        tvm.runtime.tensor(Wv, device=dev),
        tvm.runtime.tensor(Wo, device=dev),
    ).numpy()

    def run_test(sess):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tmpdir + "/test.so"
            relax_build(ShardedAttention).export_library(path)

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
            Y_result = tvm.runtime.empty((1, 10, 128), "float32", device=dev)
            sess.copy_from_worker_0(Y_result, d_Y)
            sess.sync_worker_0()
            return Y_result.numpy()

    Y_result = _run_with_ccl_session(session_kind, devices, run_test)
    # pylint: enable=invalid-name
    np.testing.assert_allclose(Y_result, Y_expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
