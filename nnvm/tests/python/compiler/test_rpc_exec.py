import tvm
from tvm.contrib import util, rpc, graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
import numpy as np

def test_rpc_executor():
    host = "localhost"
    port = 9100
    server = rpc.Server(host, port, use_popen=True)

    x = sym.Variable("x")
    y = sym.Variable("y")
    z = sym.exp(y + x)
    shape = (10, 128)
    dtype = tvm.float32
    shape_dict = {"x": shape, "y": shape}
    tmp = util.tempdir()
    lib_name  = tmp.relpath("net.o")

    graph, lib, _ = nnvm.compiler.build(z, "llvm", shape_dict)
    # save module
    lib.save(lib_name)
    remote = rpc.connect(host, port)
    remote.upload(lib_name)
    ctx = remote.cpu(0)
    # load remote
    rlib = remote.load_module("net.o")

    # Create remotemodule
    m = graph_runtime.create(graph, rlib, remote.cpu(0))
    # get member functions
    set_input, run, get_output = m["set_input"], m["run"], m["get_output"]
    na = tvm.nd.array(np.ones(shape).astype(dtype), ctx)
    nb = tvm.nd.array(np.ones(shape).astype(dtype), ctx)
    # set inputs
    set_input("x", na)
    set_input("y", nb)
    # execute
    run()
    # get outputs
    out = tvm.nd.empty(shape, dtype, ctx)
    get_output(0, out)
    np.testing.assert_allclose(
        out.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))
    server.terminate()

if __name__ == "__main__":
    test_rpc_executor()
