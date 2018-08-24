import tvm
import os
import logging
import time
import multiprocessing

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
        A = tvm.placeholder(shape, dtype=dtype)
        B = tvm.compute(A.shape, lambda i: A[i]+tvm.const(1, A.dtype))
        s = tvm.create_schedule(B.op)
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
        np.testing.assert_allclose(a.asnumpy() + 1, b.asnumpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_rpc(remote, target, (10,), dtype)


def test_rpc_simple():
    if not tvm.module.enabled("rpc"):
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
    try:
        f3("abc")
        assert False
    except tvm.TVMError as e:
        assert "abc" in str(e)

    f2 = client.get_function("rpc.test.strcat")
    assert f2("abc", 11) == "abc:11"

def test_rpc_array():
    if not tvm.module.enabled("rpc"):
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

def test_rpc_file_exchange():
    if not tvm.module.enabled("rpc"):
        return
    server = rpc.Server("localhost")
    remote = rpc.connect(server.host, server.port)
    blob = bytearray(np.random.randint(0, 10, size=(10)))
    remote.upload(blob, "dat.bin")
    rev = remote.download("dat.bin")
    assert(rev == blob)

def test_rpc_remote_module():
    if not tvm.module.enabled("rpc"):
        return
    server = rpc.Server("localhost")
    client = rpc.connect(server.host, server.port)
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    def check_remote(remote):
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        temp = util.tempdir()
        ctx = remote.cpu(0)
        f = tvm.build(s, [A, B], "llvm", name="myadd")
        path_dso = temp.relpath("dev_lib.so")
        f.export_library(path_dso)
        remote.upload(path_dso)
        f1 = remote.load_module("dev_lib.so")
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        time_f = f1.time_evaluator(f1.entry_name, remote.cpu(0), number=10)
        cost = time_f(a, b).mean
        print('%g secs/op' % cost)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    def check_remote_link_cl(remote):
        """Test function to run remote code such as cl

        This is not enabled because there is forking issue
        of TVM runtime when server launches after OpenCL
        runtime initializes. We leave it as an example
        on how to do rpc when we want to do linking on remote.
        """
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        if not tvm.module.enabled("opencl"):
            print("Skip because opencl is not enabled")
            return
        temp = util.tempdir()
        ctx = remote.cl(0)
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=32)
        s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
        s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
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
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        fhost(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)
        # Option 2: export library as a tar ball then handled by remote compiler
        path_tar = temp.relpath("myadd.tar")
        f.export_library(path_tar)
        remote.upload(path_tar)
        fhost = remote.load_module("myadd.tar")
        a = tvm.nd.array(np.random.uniform(size=1024).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(1024, dtype=A.dtype), ctx)
        fhost(a, b)
        np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    check_remote(client)
    check_remote(rpc.LocalSession())


def test_rpc_return_func():
    @tvm.register_func("rpc.test.remote_func")
    def addone(x):
        return lambda y: x+y

    server = rpc.Server("localhost", key="x1")
    client = rpc.connect(server.host, server.port, key="x1")
    f1 = client.get_function("rpc.test.remote_func")
    fadd = f1(10)
    assert fadd(12) == 22


def test_rpc_return_ndarray():
    # Use closure to check the ref counter correctness
    nd = tvm.nd.array(np.zeros(10).astype("float32"))
    @tvm.register_func("rpc.test.remote_return_nd")
    def my_module(name):
        if name == "get_arr":
            return lambda : nd
        elif name == "ref_count":
            return lambda : tvm._api_internal._ndarray_use_count(nd)
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
