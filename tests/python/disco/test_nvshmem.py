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
"""Basic tests for a Disco nvshmem support"""
# pylint: disable=missing-docstring
import tempfile

import numpy as np
import pytest
import subprocess
import threading
import sys
from multiprocessing import Process
from typing import Any, Callable, List

from tvm.script import tir as T


import tvm
import tvm.testing
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.exec import disco_worker as _  # pylint: disable=unused-import

_SOCKET_SESSION_TESTER = None


def get_free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SocketSessionTester:
    def __init__(self, num_workers):
        num_nodes = 2
        num_groups = 1
        assert num_workers % num_nodes == 0
        num_workers_per_node = num_workers // num_nodes
        server_host = "localhost"
        server_port = get_free_port()
        self.sess = None

        def start_server():
            self.sess = di.SocketSession(
                num_nodes, num_workers_per_node, num_groups, server_host, server_port
            )

        thread = threading.Thread(target=start_server)
        thread.start()

        cmd = "tvm.exec.disco_remote_socket_session"
        self.remote_nodes = []
        for _ in range(num_nodes - 1):
            self.remote_nodes.append(
                subprocess.Popen(
                    [
                        "python3",
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

    def __del__(self):
        if self.sess is not None:
            self.sess.shutdown()
            del self.sess


def create_socket_session(num_workers):
    global _SOCKET_SESSION_TESTER
    if _SOCKET_SESSION_TESTER is not None:
        del _SOCKET_SESSION_TESTER
    _SOCKET_SESSION_TESTER = SocketSessionTester(num_workers)
    assert _SOCKET_SESSION_TESTER.sess is not None
    return _SOCKET_SESSION_TESTER.sess


def test_nvshmem_init_finalize(session_kind: di.Session, num_workers: int):
    if tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", True) is None:
        return

    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


def test_nvshmem_empty(session_kind: di.Session, num_workers: int):
    if tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", True) is None:
        return

    device = tvm.cuda()
    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    empty_dfunc = sess.get_global_func("runtime.disco.nvshmem.empty")
    a = empty_dfunc(ShapeTuple((32, 64)), "float32", device)
    b = empty_dfunc(ShapeTuple((64, 32)), "float32", device)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


def test_nvshmem_compile():
    if tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", True) is None:
        return

    num_workers = 4
    sess = di.ProcessSession(num_workers=num_workers)

    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()

    @T.prim_func
    def main(A: T.Buffer((8, 16), "float32"), B: T.Buffer((16, 8), "float32")):
        for i in T.thread_binding(T.int64(8), thread="threadIdx.y"):
            for j in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(8), i)
                    v1 = T.axis.spatial(T.int64(16), j)
                    T.reads(A[v0, v1])
                    T.writes(B[v1, v0])
                    B[v1, v0] = A[v0, v1]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        A_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
        B_np = np.zeros((16, 8), dtype="float32")
        A_array = sess.empty(A_np.shape, "float32")
        B_array = sess.empty(B_np.shape, "float32")
        A_array.debug_copy_from(0, A_np)

        target = tvm.target.Target("cuda")
        tvm.compile(main, target=target).export_library(path)
        mod = sess.load_vm_module(path)
        mod["main"](A_array, B_array)

        B_res = B_array.debug_get_from_remote(0).numpy()
        np.testing.assert_equal(B_res, A_np.T)

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()

    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


if __name__ == "__main__":
    # After the first call to `nvshmem_init`, a subsequent call to `nvshmem_init`
    # or `nvshmem_init_thread` in the same program results in undefined behavior.
    # So we always create a new process to run the test. Then no repeated nvshmem
    # init happens in the same process, since the worker0 may share the same process.
    for session_kind in [create_socket_session, di.ProcessSession]:
        for num_workers in [2, 4]:
            for test_func in [test_nvshmem_init_finalize, test_nvshmem_empty]:
                p = Process(target=test_func, args=[session_kind, num_workers])
                p.start()
                p.join()

    # testing compilation flow
    p = Process(target=test_nvshmem_compile)
    p.start()
    p.join()
