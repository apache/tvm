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
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest
from tvm_ffi import Shape

import tvm
import tvm.testing
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.testing import env

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not env.has_nvshmem(), reason="need nvshmem"),
]


_SOCKET_SESSION_TESTER = None


def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class SocketSessionTester:
    """Run a disco SocketSession with one local node and remote nodes.

    Each remote node is a `tvm.exec.disco_remote_socket_session` subprocess
    launched with the current Python interpreter.
    """

    def __init__(self, num_workers, num_nodes=2, num_groups=1):
        # Initialize the attributes used by __del__ first, so that teardown is
        # safe even when __init__ raises below.
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
                    num_nodes, num_workers_per_node, num_groups, server_host, server_port
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


def create_socket_session(num_workers):
    """Create a socket session backed by one local and one remote node.

    The tester is kept alive in a module-level global so that the session
    survives until the next call (or interpreter exit) replaces it.
    """
    global _SOCKET_SESSION_TESTER
    # Rebind (not `del`) so the global stays defined if the constructor raises.
    _SOCKET_SESSION_TESTER = None
    _SOCKET_SESSION_TESTER = SocketSessionTester(num_workers)
    assert _SOCKET_SESSION_TESTER.sess is not None
    return _SOCKET_SESSION_TESTER.sess


_all_session_kinds = [di.ProcessSession, create_socket_session]
_all_num_workers = [2, 4]

_SUBPROCESS_TIMEOUT_SEC = 600


def _run_in_fresh_process(target, *args):
    """Run a test body in a freshly spawned process.

    After the first call to `nvshmem_init`, a subsequent call to `nvshmem_init`
    or `nvshmem_init_thread` in the same program results in undefined behavior,
    and worker-0 of a Disco session lives in the calling process. So each test
    body must run in its own process. The 'spawn' start method avoids
    inheriting CUDA state from this process.
    """

    def run():
        proc = multiprocessing.get_context("spawn").Process(target=target, args=args)
        proc.start()
        proc.join(timeout=_SUBPROCESS_TIMEOUT_SEC)
        if proc.is_alive():
            proc.kill()
            proc.join()
            pytest.fail(
                f"{target.__name__}{args} timed out after {_SUBPROCESS_TIMEOUT_SEC} seconds"
            )
        assert proc.exitcode == 0, f"{target.__name__}{args} failed with exit code {proc.exitcode}"

    tvm.testing.run_with_gpu_lock(run)


def _require_cuda_devices(num_workers):
    # Each nvshmem worker binds its own CUDA device (cudaSetDevice(worker_id)).
    if not all(tvm.cuda(i).exist for i in range(num_workers)):
        pytest.skip(f"Requires {num_workers} CUDA devices")


def _init_finalize(session_kind, num_workers):
    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


def _empty(session_kind, num_workers):
    device = tvm.cuda()
    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    empty_dfunc = sess.get_global_func("runtime.disco.nvshmem.empty")
    _a = empty_dfunc(Shape((32, 64)), "float32", device)
    _b = empty_dfunc(Shape((64, 32)), "float32", device)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("num_workers", _all_num_workers)
def test_nvshmem_init_finalize(session_kind, num_workers: int):
    _require_cuda_devices(num_workers)
    _run_in_fresh_process(_init_finalize, session_kind, num_workers)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("num_workers", _all_num_workers)
def test_nvshmem_empty(session_kind, num_workers: int):
    _require_cuda_devices(num_workers)
    _run_in_fresh_process(_empty, session_kind, num_workers)


def _compile():
    num_workers = 2
    sess = di.ProcessSession(num_workers=num_workers)

    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()

    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((8, 16), "float32"), B: T.Buffer((16, 8), "float32")):
        for i in T.thread_binding(T.int64(8), thread="threadIdx.y"):
            for j in T.thread_binding(T.int64(16), thread="threadIdx.x"):
                with T.sblock("T_transpose"):
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


def test_nvshmem_compile():
    _require_cuda_devices(2)
    _run_in_fresh_process(_compile)


NVSHMEM_QUERY_KERNEL_SOURCE = """
#include <nvshmem.h>

extern "C" __global__ void nvshmem_query_kernel(int* my_pe_out, int* n_pes_out) {
    my_pe_out[0] = nvshmem_my_pe();
    n_pes_out[0] = nvshmem_n_pes();
}
"""


def _kernel_compile(compile_mode):
    """Compile and run a kernel that calls NVSHMEM functions.

    Runs in a fresh process, so setting the env var is safe.
    """
    os.environ["TVM_CUDA_COMPILE_MODE"] = compile_mode

    num_workers = 2
    sess = di.ProcessSession(num_workers=num_workers)

    f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()

    try:

        @I.ir_module(s_tir=True)
        class NvshmemQueryModule:
            @T.prim_func(s_tir=True)
            def query_pe(
                my_pe_out: T.Buffer((1,), "int32"),
                n_pes_out: T.Buffer((1,), "int32"),
            ):
                with T.sblock("root"):
                    T.reads()
                    T.writes(my_pe_out[0:1], n_pes_out[0:1])
                    T.call_kernel(
                        NVSHMEM_QUERY_KERNEL_SOURCE,
                        ((1,), (1,)),  # grid=(1,), block=(1,)
                        my_pe_out.data,
                        n_pes_out.data,
                        kernel_name="nvshmem_query_kernel",
                    )

            @R.function
            def main() -> R.Tuple(R.Tensor((1,), "int32"), R.Tensor((1,), "int32")):
                cls = NvshmemQueryModule
                with R.dataflow():
                    my_pe = R.call_tir(
                        cls.query_pe,
                        (),
                        out_ty=[
                            R.Tensor((1,), "int32"),
                            R.Tensor((1,), "int32"),
                        ],
                    )
                    R.output(my_pe)
                return my_pe

        tmpdir = tempfile.mkdtemp()
        try:
            path = tmpdir + "/test_nvshmem_kernel.so"

            target = tvm.target.Target("cuda")
            tvm.compile(NvshmemQueryModule, target=target).export_library(path)
            mod = sess.load_vm_module(path)
            result = mod["main"]()

            # Verify results from each worker
            for worker_id in range(num_workers):
                my_pe_result, n_pes_result = result.debug_get_from_remote(worker_id)
                my_pe_val = my_pe_result.numpy()[0]
                n_pes_val = n_pes_result.numpy()[0]
                assert my_pe_val == worker_id, (
                    f"Worker {worker_id} reported my_pe={my_pe_val}, expected {worker_id}"
                )
                assert n_pes_val == num_workers, (
                    f"Worker {worker_id} reported n_pes={n_pes_val}, expected {num_workers}"
                )

            # Sync all workers before cleanup
            sess._sync_all()

            finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
            finalize_dfunc()
            sess.sync_worker_0()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    finally:
        sess.shutdown()


def test_nvshmem_kernel_compile_nvcc():
    """Test NVSHMEM kernel compilation with nvcc."""
    _require_cuda_devices(2)
    _run_in_fresh_process(_kernel_compile, "nvcc")


def test_nvshmem_kernel_compile_nvrtc():
    """Test NVSHMEM kernel compilation with nvrtc."""
    _require_cuda_devices(2)
    try:
        from cuda.bindings import nvrtc  # noqa: F401
    except ImportError:
        pytest.skip("cuda-python not available, skipping nvrtc test")

    _run_in_fresh_process(_kernel_compile, "nvrtc")


if __name__ == "__main__":
    tvm.testing.main()
