import tvm
import logging
import numpy as np
import time
from tvm.contrib import rpc, util

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

    server = rpc.Server("localhost")
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
    print("second connect")
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
    remote = rpc.connect(server.host, server.port)
    # graph
    n = tvm.convert(1024)
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    def check_remote():
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

    def check_remote_link_cl():
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

    check_remote()

def test_rpc_return_func():
    @tvm.register_func("rpc.test.remote_func")
    def addone(x):
        return lambda y: x+y
    server = rpc.Server("localhost")
    client = rpc.connect(server.host, server.port, key="x1")
    f1 = client.get_function("rpc.test.remote_func")
    fadd = f1(10)
    assert fadd(12) == 22


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_rpc_remote_module()
    test_rpc_return_func()
    test_rpc_file_exchange()
    test_rpc_array()
    test_rpc_simple()
