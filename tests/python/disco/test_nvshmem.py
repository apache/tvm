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
