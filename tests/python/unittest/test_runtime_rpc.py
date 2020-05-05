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
import os
import stat
import logging
import time
import multiprocessing

import pytest
import numpy as np
from tvm import rpc
from tvm.contrib import util
from tvm.rpc.tracker import Tracker


def test_bigendian_rpc():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        return
    def verify_rpc(remote, target, shape, dtype):
        A = te.placeholder(shape, dtype=dtype)
        B = te.compute(A.shape, lambda i: A[i]+tvm.tir.const(1, A.dtype))
        s = te.create_schedule(B.op)
        f = tvm.build(s, [A, B], target, name="myadd")

        ctx = remote.cpu(0)
        a = tvm.nd.array(np.random.randint(0, 256, size=shape).astype(A.dtype), ctx=ctx)
        b = tvm.nd.array(np.zeros(shape).astype(A.dtype), ctx=ctx)
        temp = util.tempdir()
        path_dso = temp.relpath("dev_lib.o")
        f.save(path_dso)
        remote.upload(path_dso)
        f = remote.load_module("dev_lib.o")
        f(a, b)
        tvm.testing.assert_allclose(a.asnumpy() + 1, b.asnumpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_rpc(remote, target, (10,), dtype)


def test_rpc_simple():
    if not tvm.runtime.enabled("rpc"):
        return
    @tvm.register_func("rpc.test.addone")
    def addone(x):
        return x + 1
    @tvm.register_func("rpc.test.strcat")
    def strcat(name, x):
        return "%s:%d" % (name, x)

    @tvm.register_func("rpc.test.except")
    def remotethrow(name):
        raise ValueError("%s" % name)

    server = rpc.Server("localhost", key="x1")
    client = rpc.connect(server.host, server.port, key="x1")
    f1 = client.get_function("rpc.test.addone")
    assert f1(10) == 11
    f3 = client.get_function("rpc.test.except")

    with pytest.raises(tvm.error.RPCError):
        f3("abc")

    f2 = client.get_function("rpc.test.strcat")
    assert f2("abc", 11) == "abc:11"

def test_rpc_array():
    if not tvm.runtime.enabled("rpc"):
        return
    x = np.random.randint(0, 10, size=(3, 4))
    @tvm.register_func("rpc.test.remote_array_func")
    def remote_array_func(y):
        np.testing.assert_equal(y.asnumpy(), x)
    server = rpc.Server("localhost")
    remote = rpc.connect(server.host, server.port)
    r_cpu = tvm.nd.array(x, remote.cpu(0))
    assert str(r_cpu.context).startswith("remote")
    np.testing.assert_equal(r_cpu.asnumpy(), x)
    fremote = remote.get_function("rpc.test.remote_array_func")
    fremote(r_cpu)


def test_rpc_large_array():
    # testcase of large array creation
    server = rpc.Server("localhost")
    remote = rpc.connect(server.host, server.port)
    ctx = remote.cpu(0)
    a_np = np.ones((5041, 720)).astype('float32')
    b_np = np.ones((720, 192)).astype('float32')
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    np.testing.assert_equal(a.asnumpy(), a_np)
    np.testing.assert_equal(b.asnumpy(), b_np)


def test_rpc_echo():
    def check(remote):
        fecho = remote.get_function("testing.echo")
        assert(fecho(1, 2, 3) == 1)
        assert(fecho(100, 2, 3) == 100)
        assert(fecho("xyz") == "xyz")
        assert(bytes(fecho(bytearray(b"123"))) == b"123")

        with pytest.raises(RuntimeError):
            raise_err = remote.get_function(
                "testing.test_raise_error_callback")("RuntimeError")
            raise_err()

    temp = rpc.server._server_env([])
    server = rpc.Server("localhost")
    client = rpc.connect(server.host, server.port)
    check(rpc.LocalSession())

    check(client)
    # Test minrpc server.
    temp = util.tempdir()
    minrpc_exec = temp.relpath("minrpc")
    tvm.rpc.with_minrpc("g++")(minrpc_exec, [])
    check(rpc.PopenSession(minrpc_exec))
    # minrpc on the remote
    server = rpc.Server("localhost")
    client = rpc.connect(
        server.host, server.port,
        session_constructor_args=["rpc.PopenSession",
                             open(minrpc_exec, "rb").read()])
    check(client)


def test_rpc_file_exchange():
    if not tvm.runtime.enabled("rpc"):
        return
    server = rpc.Server("localhost")
    remote = rpc.connect(server.host, server.port)
    blob = bytearray(np.random.randint(0, 10, size=(10)))
    remote.upload(blob, "dat.bin")
    rev = remote.download("dat.bin")
    assert(rev == blob)

def test_rpc_remote_module():
    if not tvm.runtime.enabled("rpc"):
        return
    # graph
    n = tvm.runtime.convert(102)
    A = te.placeholder((n,), name='A')
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = te.create_schedule(B.op)

    server = rpc.Server("localhost")
    client = rpc.connect(server.host, server.port)

    def check_remote(remote):
        if not tvm.runtime.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        temp = util.tempdir()
        ctx = remote.cpu(0)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        path_dso = temp.relpath("dev_lib.so")
        f.export_library(path_dso)
        remote.upload(path_dso)
        f1 = remote.load_module("dev_lib.so")
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), ctx)
        time_f = f1.time_evaluator(f1.entry_name, remote.cpu(0), number=10)
        cost = time_f(a, b).mean
        print('%g secs/op' % cost)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    def check_minrpc():
        if not tvm.runtime.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        if tvm.get_global_func("rpc.PopenSession", allow_missing=True) is None:
            return
        # export to minrpc
        temp = util.tempdir()
        f = tvm.build(s, [A, B], "llvm --system-lib", name="myadd")
        path_minrpc = temp.relpath("dev_lib.minrpc")
        f.export_library(path_minrpc, rpc.with_minrpc("g++"))

        with pytest.raises(RuntimeError):
            rpc.PopenSession("filenotexist")

        # statrt the minrpc session.
        remote = tvm.rpc.PopenSession(path_minrpc)
        ctx = remote.cpu(0)
        f1 = remote.system_lib()
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), ctx)
        time_f = f1.time_evaluator("myadd", remote.cpu(0), number=1)
        cost = time_f(a, b).mean
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

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
        if not tvm.runtime.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        if not tvm.runtime.enabled("opencl"):
            print("Skip because opencl is not enabled")
            return
        temp = util.tempdir()
        ctx = remote.cl(0)
        s = te.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=32)
        s[B].bind(xo, te.thread_axis("blockIdx.x"))
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        f = tvm.build(s, [A, B], "opencl", target_host="llvm", name="myadd")
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
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), ctx)
        fhost(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        # Option 2: export library as a tar ball then handled by remote compiler
        path_tar = temp.relpath("myadd.tar")
        f.export_library(path_tar)
        remote.upload(path_tar)
        fhost = remote.load_module("myadd.tar")
        a = tvm.nd.array(np.random.uniform(size=102).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(102, dtype=A.dtype), ctx)
        fhost(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    check_remote(rpc.LocalSession())
    check_remote(client)
    check_minrpc()



def test_rpc_return_func():
    @tvm.register_func("rpc.test.remote_func")
    def addone(x):
        return lambda y: x+y

    server = rpc.Server("localhost", key="x1")
    client = rpc.connect(server.host, server.port, key="x1")
    f1 = client.get_function("rpc.test.remote_func")
    fadd = f1(10)
    assert fadd(12) == 22


def test_rpc_session_constructor_args():
    # start server
    server0 = rpc.Server("localhost", key="x0")
    server1 = rpc.Server("localhost", key="x1")

    def check_multi_hop():
        # use server0 as proxy to connect to server1
        client = rpc.connect(
            server0.host, server0.port, key="x0",
            session_constructor_args=[
                "rpc.Connect", server1.host, server1.port, "x1"])

        fecho = client.get_function("testing.echo")
        assert(fecho(1, 2, 3) == 1)
        assert(fecho(100, 2, 3) == 100)
        assert(fecho("xyz") == "xyz")
        assert(bytes(fecho(bytearray(b"123"))) == b"123")

        nd = tvm.nd.array([1,2,3], ctx=client.cpu(0))
        assert(nd.asnumpy()[1] == 2)

    def check_error_handling():
        with pytest.raises(tvm.error.RPCError):
            client = rpc.connect(
                server0.host, server0.port, key="x0",
                session_constructor_args=["rpc.NonExistingConstructor"])

    check_multi_hop()
    check_error_handling()


def test_rpc_return_ndarray():
    # Use closure to check the ref counter correctness
    nd = tvm.nd.array(np.zeros(10).astype("float32"))
    @tvm.register_func("rpc.test.remote_return_nd")
    def my_module(name):
        if name == "get_arr":
            return lambda : nd
        elif name == "ref_count":
            return lambda : tvm.testing.object_use_count(nd)
        elif name == "get_elem":
            return lambda idx: nd.asnumpy()[idx]
        elif name == "get_arr_elem":
            return lambda arr, idx: arr.asnumpy()[idx]

    # start server
    server = rpc.Server("localhost", key="x1")
    client = rpc.connect(server.host, server.port, key="x1")

    m = client.get_function("rpc.test.remote_return_nd")
    get_arr = m("get_arr")
    ref_count = m("ref_count")
    get_elem = m("get_elem")
    get_arr_elem = m("get_arr_elem")
    # array test
    def run_arr_test():
        arr = get_arr()
        assert ref_count() == 2
        arr2 = get_arr()
        assert ref_count() == 3
        assert arr.context == client.cpu(0)
        arr.copyfrom(np.ones(10).astype(arr.dtype))
        assert arr2.asnumpy()[0] == 1.0
        assert get_elem(0) == 1.0
        assert get_arr_elem(arr2, 0) == 1.0

    assert ref_count() == 1
    run_arr_test()
    # check recycle correctness
    assert ref_count() == 1


def test_local_func():
    @tvm.register_func("rpc.test.remote_func2")
    def addone(x):
        return lambda y: x+y
    client = rpc.LocalSession()
    f1 = client.get_function("rpc.test.remote_func2")
    fadd = f1(10)
    assert fadd(12) == 22

    blob = bytearray(np.random.randint(0, 10, size=(10)))
    client.upload(blob, "dat.bin")
    rev = client.download("dat.bin")
    assert rev == blob

def test_rpc_tracker_register():
    # test registration
    tracker = Tracker('localhost', port=9000, port_end=10000)
    device_key = 'test_device'
    server = rpc.Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        tracker_addr=(tracker.host, tracker.port))
    time.sleep(1)
    client = rpc.connect_tracker(tracker.host, tracker.port)

    summary = client.summary()
    assert summary['queue_info'][device_key]['free'] == 1

    remote = client.request(device_key)
    summary = client.summary()
    assert summary['queue_info'][device_key]['free'] == 0

    del remote
    time.sleep(1)

    summary = client.summary()
    assert summary['queue_info'][device_key]['free'] == 1

    server.terminate()
    time.sleep(1)

    summary = client.summary()
    assert summary['queue_info'][device_key]['free'] == 0

    tracker.terminate()

def test_rpc_tracker_request():
    # test concurrent request
    tracker = Tracker('localhost', port=9000, port_end=10000)
    device_key = 'test_device'
    server = rpc.Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        tracker_addr=(tracker.host, tracker.port))
    client = rpc.connect_tracker(tracker.host, tracker.port)

    def target(host, port, device_key, timeout):
        client = rpc.connect_tracker(host, port)
        remote = client.request(device_key, session_timeout=timeout)
        while True:
            pass
        remote.cpu()

    proc1 = multiprocessing.Process(target=target,
                                    args=(tracker.host, tracker.port, device_key, 4))
    proc2 = multiprocessing.Process(target=target,
                                    args=(tracker.host, tracker.port, device_key, 200))
    proc1.start()
    time.sleep(0.5)
    proc2.start()
    time.sleep(0.5)

    summary = client.summary()

    assert summary['queue_info'][device_key]['free'] == 0
    assert summary['queue_info'][device_key]['pending'] == 1

    proc1.terminate()
    proc1.join()
    time.sleep(0.5)

    summary = client.summary()
    assert summary['queue_info'][device_key]['free'] == 0
    assert summary['queue_info'][device_key]['pending'] == 0

    proc2.terminate()
    proc2.join()
    server.terminate()
    tracker.terminate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_rpc_echo()
    test_rpc_session_constructor_args()
    test_rpc_return_ndarray()
    test_rpc_return_func()
    test_bigendian_rpc()
    test_rpc_remote_module()
    test_rpc_file_exchange()
    test_rpc_array()
    test_rpc_simple()
    test_local_func()
    test_rpc_tracker_register()
    test_rpc_tracker_request()
    test_rpc_large_array()
