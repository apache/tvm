import tvm
import os
import logging
import time
import multiprocessing
import json

import numpy as np
from tvm import rpc
from tvm.contrib import util, graph_runtime
from tvm.rpc.tracker import Tracker

def test_rpc_file_exchange():
    if not tvm.module.enabled("rpc"):
        return
    server = rpc.Server("localhost", port=9191, use_popen=True, cpp_server=True)
    remote = rpc.connect(server.host, server.port)
    blob = bytearray(np.random.randint(0, 10, size=(10)))
    remote.upload(blob, "dat.bin")
    rev = remote.download("dat.bin")
    assert(rev == blob)

def test_rpc_remote_module():
    if not tvm.module.enabled("rpc"):
        return
    server = rpc.Server("localhost", port=9193, use_popen=True, cpp_server=True)
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
    server.terminate()

def test_rpc_tracker_register():
    # test registration
    tracker = Tracker('localhost', port=9000, port_end=10000)
    device_key = 'test_device'
    server = rpc.Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        tracker_addr=(tracker.host, tracker.port),
                 cpp_server=True)
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

def test_rpc_tracker_request(cpp_server=True):
    # test concurrent request
    tracker = Tracker('localhost', port=9000, port_end=10000)
    device_key = 'test_device'
    server = rpc.Server('localhost', port=9000, port_end=10000,
                        key=device_key,
                        tracker_addr=(tracker.host, tracker.port), cpp_server=cpp_server)
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

def test_graph_simple():
    n = 4
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {"op": "tvm_op", "name": "add",
             "inputs": [[0, 0, 0]],
             "attrs": {"func_name": "myadd",
                       "flatten_data": "1",
                       "num_inputs" : "1",
                    "num_outputs" : "1"}}
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape" : ["list_shape", [shape, shape]],
        "dltype" : ["list_str", ["float32", "float32"]],
        "storage_id" : ["list_int", [0, 1]],
    }
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": outputs,
             "attrs": attrs}
    graph = json.dumps(graph)

    def check_verify():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=a)
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.asnumpy(), a + 1)

    def check_remote():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        server = rpc.Server("localhost", port=9195, use_popen=True, cpp_server=True)
        remote = rpc.connect(server.host, server.port)
        temp = util.tempdir()
        ctx = remote.cpu(0)
        path_dso = temp.relpath("dev_lib.so")
        mlib.export_library(path_dso)
        remote.upload(path_dso)
        mlib = remote.load_module("dev_lib.so")
        mod = graph_runtime.create(graph, mlib, remote.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.run(x=tvm.nd.array(a, ctx))
        out = tvm.nd.empty((n,), ctx=ctx)
        out = mod.get_output(0, out)
        np.testing.assert_equal(out.asnumpy(), a + 1)
        time.sleep(1)
        server.terminate()

    check_verify()
    check_remote()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_graph_simple()
    test_rpc_remote_module()
    test_rpc_file_exchange()
    test_rpc_tracker_register()
    test_rpc_tracker_request()
