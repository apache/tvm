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
import numpy as np
import pytest

import shutil
import subprocess
import sys
import tempfile
import threading
import multiprocessing
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

    tmpdir = tempfile.mkdtemp()
    try:
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
    finally:
        sess.shutdown()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_nvshmem_compile_nvshmem_module(
    session_kind: di.Session, num_workers: int, compile_mode: str
):
    import tvm.contrib.nvcc

    if tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", True) is None:
        return

    # Check NVSHMEM availability
    try:
        nvshmem_include_path, nvshmem_lib_path = tvm.contrib.nvcc.find_nvshmem_paths()
    except RuntimeError:
        print("NVSHMEM not found, skipping compile test")
        return

    if compile_mode == "nvrtc":
        try:
            from cuda.bindings import nvrtc
        except ImportError:
            print("cuda-python not installed, skipping nvrtc test")
            return

    # Hook the compile callback to inject headers and set compiler
    current_callback = tvm.get_global_func("tvm_callback_cuda_compile")

    def compile_callback(code, target):
        # Inject headers
        code = "#include <nvshmem.h>\n#include <nvshmemx.h>\n" + code

        if compile_mode == "nvcc":
            return tvm.contrib.nvcc.compile_cuda(code, target_format="fatbin", compiler="nvcc")
        elif compile_mode == "nvrtc":
            options = ["-I" + nvshmem_include_path]
            return tvm.contrib.nvcc.compile_cuda(
                code, target_format="cubin", options=options, compiler="nvrtc"
            )
        else:
            raise ValueError(f"Unknown mode {compile_mode}")

    tvm.register_global_func("tvm_callback_cuda_compile", compile_callback, override=True)

    try:
        sess = session_kind(num_workers=num_workers)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, num_workers, 0)
        sess.sync_worker_0()

        @T.prim_func
        def main(out: T.Buffer((1,), "int32")):
            for bx in T.thread_binding(1, thread="blockIdx.x"):
                for tx in T.thread_binding(1, thread="threadIdx.x"):
                    out[0] = T.call_extern("int32", "nvshmem_my_pe")

        tmpdir = tempfile.mkdtemp()
        path = tmpdir + "/test.so"

        target = tvm.target.Target("cuda")
        mod = tvm.compile(main, target=target)
        mod.export_library(path)

        mod_remote = sess.load_vm_module(path)
        a_array = sess.empty((1,), "int32")

        mod_remote["main"](a_array)
        sess.sync_worker_0()

        for i in range(num_workers):
            res = a_array.debug_get_from_remote(i).numpy()
            assert res[0] == i, f"Worker {i} result {res[0]} != {i}"

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        sess._sync_all()

        finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
        finalize_dfunc()
        sess.sync_worker_0()

    except Exception as e:
        print(f"Test failed with {compile_mode}: {e}")
        raise e
    finally:
        tvm.register_global_func("tvm_callback_cuda_compile", current_callback, override=True)
        sess.shutdown()
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # After the first call to `nvshmem_init`, a subsequent call to `nvshmem_init`
    # or `nvshmem_init_thread` in the same program results in undefined behavior.
    # So we always create a new process to run the test. Then no repeated nvshmem
    # init happens in the same process, since the worker0 may share the same process.

    # Use 'spawn' start method to avoid inheriting CUDA state from parent process
    # 'fork' (default on Linux) can cause issues with CUDA contexts in child processes
    multiprocessing.set_start_method("spawn", force=True)

    for session_kind in [create_socket_session, di.ProcessSession]:
        for num_workers in [2, 4]:
            for test_func in [test_nvshmem_init_finalize, test_nvshmem_empty]:
                p = Process(target=test_func, args=[session_kind, num_workers])
                p.start()
                p.join()
                # Ensure the process finished successfully
                assert p.exitcode == 0, (
                    f"Test {test_func.__name__} failed with exit code {p.exitcode}"
                )

    # testing compilation flow
    p = Process(target=test_nvshmem_compile)
    p.start()
    p.join()
    assert p.exitcode == 0, f"Test test_nvshmem_compile failed with exit code {p.exitcode}"

    # testing compilation flow for nvshmem module
    for compile_mode in ["nvcc", "nvrtc"]:
        p = Process(
            target=test_nvshmem_compile_nvshmem_module, args=[di.ProcessSession, 4, compile_mode]
        )
        p.start()
        p.join()
        assert p.exitcode == 0, f"Test test_nvshmem_compile_integration failed with {compile_mode}"
