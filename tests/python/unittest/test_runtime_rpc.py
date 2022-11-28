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
import tvm
from tvm import te
import tvm.testing
import multiprocessing
import os
import stat
import sys
import time

import pytest
import numpy as np
from tvm import rpc
from tvm.relay.backend import Runtime
from tvm.contrib import utils, cc
from tvm.rpc.tracker import Tracker
from tvm.rpc.proxy import Proxy


if __name__ == "__main__":
    # NOTE: must live here to avoid registering PackedFunc with libtvm.so twice.
    tvm.testing.main()


# tkonolige: The issue as I understand it is this: multiprocessing's spawn
# method launches a new process and then imports the relevant modules. This
# means that all registered functions must exist at the top level scope. In
# this file they are, so all is well when we run this file directly.
# However, when run under pytest, the functions aren't registered on the
# server. I believe this is because pytest is also using multiprocessing to
# run individual functions. Somewhere along the way, the imports are being
# lost, so the server ends up not registering the functions.
pytestmark = pytest.mark.skipif(
    # Windows does not support fork so we can enable Windows for testing
    sys.platform.startswith("win") == False and multiprocessing.get_start_method() != "fork",
    reason=(
        "pytest + multiprocessing spawn method causes tvm.register_func to "
        "not work on the rpc.Server."
    ),
)

# NOTE: When writing tests, wrap remote related checking in a sub-function
# to ensure all the remote resources destructs before the server terminates


@tvm.testing.requires_rpc
def test_bigendian_rpc():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        return

    def verify_rpc(remote, target, shape, dtype):
        A = te.placeholder(shape, dtype=dtype)
        B = te.compute(A.shape, lambda i: A[i] + tvm.tir.const(1, A.dtype))
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], target, name="myadd")

        dev = remote.cpu(0)
        a = tvm.nd.array(np.random.randint(0, 256, size=shape).astype(A.dtype), device=dev)
        b = tvm.nd.array(np.zeros(shape).astype(A.dtype), device=dev)
        temp = utils.tempdir()
        path_dso = temp.relpath("dev_lib.o")
        f.save(path_dso)
        remote.upload(path_dso)
        f = remote.load_module("dev_lib.o")
        f(a, b)
        tvm.testing.assert_allclose(a.numpy() + 1, b.numpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_rpc(remote, target, (10,), dtype)


@tvm.testing.requires_rpc
def test_rpc_simple():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        f1 = client.get_function("rpc.test.addone")
        assert f1(10) == 11
        f3 = client.get_function("rpc.test.except")

        with pytest.raises(tvm._ffi.base.TVMError):
            f3("abc")

        f2 = client.get_function("rpc.test.strcat")
        assert f2("abc", 11) == "abc:11"

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_simple_wlog():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1", enable_logging=True)

    def check_remote():
        f1 = client.get_function("rpc.test.addone")
        assert f1(10) == 11
        f3 = client.get_function("rpc.test.except")

        with pytest.raises(tvm._ffi.base.TVMError):
            f3("abc")

        f2 = client.get_function("rpc.test.strcat")
        assert f2("abc", 11) == "abc:11"

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_runtime_string():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        func = client.get_function("rpc.test.runtime_str_concat")
        x = tvm.runtime.container.String("abc")
        y = tvm.runtime.container.String("def")
        assert str(func(x, y)) == "abcdef"

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_array():
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        x = np.ones((3, 4))
        r_cpu = tvm.nd.array(x, remote.cpu(0))
        assert str(r_cpu.device).startswith("remote")
        np.testing.assert_equal(r_cpu.numpy(), x)
        fremote = remote.get_function("rpc.test.remote_array_func")
        fremote(r_cpu)

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_large_array():
    # testcase of large array creation
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        dev = remote.cpu(0)
        a_np = np.ones((5041, 720)).astype("float32")
        b_np = np.ones((720, 192)).astype("float32")
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        np.testing.assert_equal(a.numpy(), a_np)
        np.testing.assert_equal(b.numpy(), b_np)

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_echo():
    def check(remote):
        fecho = remote.get_function("testing.echo")
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"

        with pytest.raises(RuntimeError):
            raise_err = remote.get_function("testing.test_raise_error_callback")("RuntimeError")
            raise_err()

        remote.cpu().sync()
        with pytest.raises(AttributeError):
            f3 = remote.system_lib()["notexist"]

    temp = rpc.server._server_env([])
    server = rpc.Server()
    client = rpc.connect("127.0.0.1", server.port)
    check(rpc.LocalSession())

    check(client)

    def check_minrpc():
        if tvm.get_global_func("rpc.CreatePipeClient", allow_missing=True) is None:
            return
        # Test minrpc server.
        temp = utils.tempdir()
        minrpc_exec = temp.relpath("minrpc")
        tvm.rpc.with_minrpc(cc.create_executable)(minrpc_exec, [])
        check(rpc.PopenSession(minrpc_exec))
        # minrpc on the remote
        server = rpc.Server()
        client = rpc.connect(
            "127.0.0.1",
            server.port,
            session_constructor_args=["rpc.PopenSession", open(minrpc_exec, "rb").read()],
        )
        check(client)

    check_minrpc()


@tvm.testing.requires_rpc
def test_rpc_file_exchange():
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        blob = bytearray(np.random.randint(0, 10, size=(10)))
        remote.upload(blob, "dat.bin")
        rev = remote.download("dat.bin")
        assert rev == blob

    check_remote()


@tvm.testing.requires_rpc
@tvm.testing.requires_llvm
def test_rpc_remote_module():
    # graph
    n = tvm.runtime.convert(102)
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    server0 = rpc.Server(key="x0")
    server1 = rpc.Server(key="x1")

    client = rpc.connect(
        "127.0.0.1",
        server0.port,
        key="x0",
        session_constructor_args=["rpc.Connect", "127.0.0.1", server1.port, "x1", False],
    )

    def check_remote(remote):
        temp = utils.tempdir()
        dev = remote.cpu(0)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        path_dso = temp.relpath("dev_lib.so")
        f.export_library(path_dso)
        remote.upload(path_dso)
        f1 = remote.load_module("dev_lib.so")
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), dev)
        time_f = f1.time_evaluator(f1.entry_name, remote.cpu(0), number=10)
        cost = time_f(a, b).mean
        print("%g secs/op" % cost)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        # Download the file from the remote
        path_tar = temp.relpath("dev_lib.tar")
        f.export_library(path_tar)
        remote.upload(path_tar)
        local_download_path = temp.relpath("dev_lib.download.so")
        with open(local_download_path, "wb") as fo:
            fo.write(remote.download_linked_module("dev_lib.tar"))
        fupdated = tvm.runtime.load_module(local_download_path)
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), tvm.cpu(0))
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), tvm.cpu(0))
        fupdated(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    def check_minrpc():
        if tvm.get_global_func("rpc.CreatePipeClient", allow_missing=True) is None:
            return
        # export to minrpc
        temp = utils.tempdir()
        runtime = Runtime("cpp", {"system-lib": True})
        f = tvm.build(s, [A, B], "llvm", name="myadd", runtime=runtime)
        path_minrpc = temp.relpath("dev_lib.minrpc")
        f.export_library(path_minrpc, rpc.with_minrpc(cc.create_executable))

        with pytest.raises(RuntimeError):
            rpc.PopenSession("filenotexist")

        # statrt the minrpc session.
        remote = tvm.rpc.PopenSession(path_minrpc)
        dev = remote.cpu(0)
        f1 = remote.system_lib()

        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), dev)
        time_f = f1.time_evaluator("myadd", remote.cpu(0), number=1)
        cost = time_f(a, b).mean
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        # change to not executable
        os.chmod(path_minrpc, stat.S_IRUSR)
        with pytest.raises(RuntimeError):
            rpc.PopenSession(path_minrpc)

    def check_remote_link_cl(remote):
        """Test function to run remote code such as cl

        This is not enabled because there is forking issue
        of TVM runtime when server launches after OpenCL
        runtime initializes. We leave it as an example
        on how to do rpc when we want to do linking on remote.
        """
        if not tvm.testing.device_enabled("opencl"):
            print("Skip because opencl is not enabled")
            return
        temp = utils.tempdir()
        dev = remote.cl(0)
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=32)
        s[B].bind(xo, te.thread_axis("blockIdx.x"))
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        f = tvm.build(s, [A, B], "opencl --host=llvm", name="myadd")
        # Option 1: save modules separately and rely on remote compiler
        path_o = temp.relpath("myadd.o")
        path_cl = temp.relpath("myadd.cl")
        path_json = temp.relpath("myadd.tvm_meta.json")
        f.save(path_o)
        f.imported_modules[0].save(path_cl)
        remote.upload(path_o)
        remote.upload(path_cl)
        # upload meta data
        remote.upload(path_json)
        fhost = remote.load_module("myadd.o")
        fdev = remote.load_module("myadd.cl")
        fhost.import_module(fdev)
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), dev)
        fhost(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)
        # Option 2: export library as a tar ball then handled by remote compiler
        path_tar = temp.relpath("myadd.tar")
        f.export_library(path_tar)
        remote.upload(path_tar)
        fhost = remote.load_module("myadd.tar")
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), dev)
        fhost(a, b)
        np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    check_remote(rpc.LocalSession())
    check_remote(client)
    check_minrpc()


@tvm.testing.requires_rpc
def test_rpc_return_func():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        f1 = client.get_function("rpc.test.add_to_lhs")
        fadd = f1(10)
        assert fadd(12) == 22

    check_remote()


@tvm.testing.requires_rpc
def test_rpc_session_constructor_args():
    # start server
    server0 = rpc.Server(key="x0")
    server1 = rpc.Server(key="x1")

    def check_multi_hop():
        # use server0 as proxy to connect to server1
        client = rpc.connect(
            "127.0.0.1",
            server0.port,
            key="x0",
            session_constructor_args=["rpc.Connect", "127.0.0.1", server1.port, "x1", False],
        )

        fecho = client.get_function("testing.echo")
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"

        nd = tvm.nd.array([1, 2, 3], device=client.cpu(0))
        assert nd.numpy()[1] == 2

    def check_error_handling():
        with pytest.raises(tvm.error.RPCError):
            client = rpc.connect(
                "127.0.0.1",
                server0.port,
                key="x0",
                session_constructor_args=["rpc.NonExistingConstructor"],
            )

    check_multi_hop()
    check_error_handling()


@tvm.testing.requires_rpc
def test_rpc_return_ndarray():
    # start server
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    m = client.get_function("rpc.test.remote_return_nd")
    get_arr = m("get_arr")
    ref_count = m("ref_count")
    get_elem = m("get_elem")
    get_arr_elem = m("get_arr_elem")
    # array test
    def run_arr_test():
        arr = get_arr()
        assert get_elem(0) == 0.0
        assert get_arr_elem(arr, 0) == 0.0

    run_arr_test()


@tvm.testing.requires_rpc
def test_local_func():
    client = rpc.LocalSession()

    def check_remote():
        f1 = client.get_function("rpc.test.add_to_lhs")
        fadd = f1(10)
        assert fadd(12) == 22

        blob = bytearray(np.random.randint(0, 10, size=(10)))
        client.upload(blob, "dat.bin")
        rev = client.download("dat.bin")
        assert rev == blob

    check_remote()


@tvm.testing.requires_rpc
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_register(device_key):
    # test registration
    tracker = Tracker(port=9000, port_end=10000)
    server1 = rpc.Server(
        host="127.0.0.1",
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
    )
    server2 = rpc.Server(
        host="127.0.0.1",
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
        custom_addr="test_addr",  # this is a test address, which is unable to connect
    )
    time.sleep(1)
    client = rpc.connect_tracker("127.0.0.1", tracker.port)

    def exist_address(summary, key, host, port):
        server_info = summary["server_info"]
        for device in server_info:
            if device["key"] == "server:%s" % key:
                addr = device["addr"]
                if (host is None or host == addr[0]) and port == addr[1]:
                    return True
        return False

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 2
    assert exist_address(summary, device_key, "127.0.0.1", server1.port)
    assert exist_address(summary, device_key, "test_addr", server2.port)

    remote = client.request(device_key)
    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 1

    del remote
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 2

    server1.terminate()
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 1
    assert not exist_address(summary, device_key, "127.0.0.1", server1.port)
    assert exist_address(summary, device_key, "test_addr", server2.port)

    server2.terminate()
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 0
    assert not exist_address(summary, device_key, "test_addr", server2.port)

    tracker.terminate()


def _target(host, port, device_key, timeout):
    client = rpc.connect_tracker(host, port)
    remote = client.request(device_key, session_timeout=timeout)
    while True:
        pass
    remote.cpu()


@tvm.testing.requires_rpc
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_request(device_key):
    # test concurrent request
    tracker = Tracker(port=9000, port_end=10000)
    server = rpc.Server(
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
    )
    client = rpc.connect_tracker("127.0.0.1", tracker.port)

    proc1 = multiprocessing.Process(target=_target, args=("127.0.0.1", tracker.port, device_key, 4))
    proc2 = multiprocessing.Process(
        target=_target, args=("127.0.0.1", tracker.port, device_key, 200)
    )
    proc1.start()
    time.sleep(0.5)
    proc2.start()
    time.sleep(0.5)

    summary = client.summary()

    assert summary["queue_info"][device_key]["free"] == 0
    assert summary["queue_info"][device_key]["pending"] == 1

    proc1.terminate()
    proc1.join()
    time.sleep(0.5)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 0
    assert summary["queue_info"][device_key]["pending"] == 0

    proc2.terminate()
    proc2.join()
    server.terminate()
    tracker.terminate()


@tvm.testing.requires_rpc
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_via_proxy(device_key):
    """
         tracker
         /     \
    Host   --   Proxy -- RPC server
    """

    tracker_server = Tracker(port=9000, port_end=9100)
    proxy_server = Proxy(
        host=tracker_server.host,
        port=8888,
        port_end=8988,
        tracker_addr=(tracker_server.host, tracker_server.port),
    )

    server1 = rpc.Server(
        host=proxy_server.host,
        port=proxy_server.port,
        key=device_key,
        tracker_addr=(tracker_server.host, tracker_server.port),
        is_proxy=True,
    )
    server2 = rpc.Server(
        host=proxy_server.host,
        port=proxy_server.port,
        key=device_key,
        tracker_addr=(tracker_server.host, tracker_server.port),
        is_proxy=True,
    )

    client = rpc.connect_tracker(tracker_server.host, tracker_server.port)
    remote1 = client.request(device_key, session_timeout=30)  # pylint: disable=unused-variable
    remote2 = client.request(device_key, session_timeout=30)  # pylint: disable=unused-variable

    server2.terminate()
    server1.terminate()
    proxy_server.terminate()
    tracker_server.terminate()
