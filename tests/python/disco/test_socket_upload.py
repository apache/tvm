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

"""Test upload_vm_module over a multi-node SocketSession."""

import hashlib
import pathlib
import socket
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax as rx
from tvm.runtime import disco as di
from tvm.script import relax as R

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)

_NUM_NODES = 4
_NUM_WORKERS = 4
_REL_PATH = "./mod.so"


def _node_dir_name(node_id):
    return f"upload_node{node_id}"


def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _sha256(path):
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


@tvm.script.ir_module
class Mod:  # pylint: disable=too-few-public-methods
    @R.function
    def double(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
        R.func_attr({"global_symbol": "double"})
        with R.dataflow():
            y: R.Tensor((4,), "float32") = R.add(x, x)
            R.output(y)
        return y


class SocketSessionTester:
    def __init__(self, num_workers, node_cwds, num_groups=1, build_ring=True):
        self.sess = None
        self.remote_nodes = []
        num_nodes = len(node_cwds)
        assert num_workers % num_nodes == 0
        num_workers_per_node = num_workers // num_nodes
        server_host = "localhost"
        server_port = _get_free_port()

        cmd = "tvm.exec.disco_remote_socket_session"
        for cwd in node_cwds[1:]:  # node 0 is this process
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
                    cwd=str(cwd),
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )

        self.sess = di.SocketSession(
            num_nodes,
            num_workers_per_node,
            num_groups,
            server_host,
            server_port,
            build_ring,
        )

    # Bound at class creation: module globals may already be cleared when
    # __del__ runs during interpreter shutdown.
    _TIMEOUT_EXPIRED = subprocess.TimeoutExpired

    def close(self):
        try:
            # Shut down the session first so remote nodes can exit gracefully.
            if self.sess is not None:
                self.sess.shutdown()
                self.sess = None
        finally:
            for node in self.remote_nodes:
                try:
                    node.wait(timeout=10)
                except self._TIMEOUT_EXPIRED:
                    node.kill()
                    node.wait()
            self.remote_nodes = []

    def __del__(self):
        self.close()


def test_upload_vm_module(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="disco_upload_") as tmp_root:
        root = pathlib.Path(tmp_root)
        node_dirs = [root / _node_dir_name(i) for i in range(_NUM_NODES)]

        for node_dir in node_dirs:
            node_dir.mkdir()
        # Please add `-s` to print the directory for verification.
        print(f"\n[setup] created {_NUM_NODES} node directories under {root}")

        for node_dir in node_dirs:
            print(f"          {node_dir}")

        monkeypatch.chdir(node_dirs[0])

        target = tvm.target.Target("llvm")
        tvm.compile(rx.get_pipeline("zero")(Mod), target=target).export_library(_REL_PATH)
        expected_sha = _sha256(_REL_PATH)
        built = node_dirs[0] / _REL_PATH

        print(
            f"[setup] compiled {built} ({built.stat().st_size} bytes, sha256={expected_sha[:16]})"
        )

        # Precondition: only the controller has the artifact.
        for node_dir in node_dirs[1:]:
            assert not (node_dir / _REL_PATH).exists()

        print(f"[setup] upload_node1..{_NUM_NODES - 1} are empty, as expected")

        tester = SocketSessionTester(_NUM_WORKERS, node_dirs)
        try:
            sess = tester.sess
            sess._sync_all()
            sess.upload_vm_module(_REL_PATH)
            sess._sync_all()  # pylint: disable=protected-access

            print("[upload] per-node state after upload_vm_module:")

            for i, node_dir in enumerate(node_dirs):
                written = node_dir / _REL_PATH
                assert written.is_file(), f"{_node_dir_name(i)} never received the module"
                digest = _sha256(written)
                print(
                    f"          {_node_dir_name(i)}: {written.stat().st_size} bytes,"
                    f" sha256={digest[:16]}"
                )
                assert digest == expected_sha, f"{_node_dir_name(i)} got a corrupt copy"

            # The bytes arrived intact; now show they are loadable and runnable on every node.
            mod = sess.load_vm_module(_REL_PATH)
            print(f"[load] load_vm_module({_REL_PATH!r}) succeeded on all {_NUM_NODES} nodes")

            d_x = sess.empty((4,), "float32")

            for i in range(_NUM_WORKERS):
                d_x.debug_copy_from(i, np.full((4,), i + 1, dtype="float32"))

            d_y = mod["double"](d_x)
            for i in range(_NUM_WORKERS):
                got = d_y.debug_get_from_remote(i).numpy()
                np.testing.assert_equal(got, np.full((4,), 2 * (i + 1), dtype="float32"))
                print(f"          worker {i}: double({i + 1}) -> {got}")
        finally:
            tester.close()

    print(f"[cleanup] removed {tmp_root}")


if __name__ == "__main__":
    tvm.testing.main()
