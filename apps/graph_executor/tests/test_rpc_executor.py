import tvm
from tvm.contrib import util, rpc
import tvm_graph as tg
import numpy as np
import os

def test_rpc_executor():
    host = 'localhost'
    port = 9091
    server = rpc.Server(host, port)

    tmp = util.tempdir()
    sym_fname    = tmp.relpath('net.json')
    lib_fname    = tmp.relpath('net.o')
    param_fname  = tmp.relpath('net.param')

    x = tg.Variable('x')
    y = tg.Variable('y')
    sym = tg.exp(y + x)

    shape = (10, 128)
    dtype = tvm.float32
    na = tvm.nd.array(np.ones(shape).astype(dtype))
    nb = tvm.nd.array(np.ones(shape).astype(dtype))
    tg.save_params(param_fname, {'x': na, 'y': nb})

    remote = rpc.connect(host, port)
    ctx = remote.cpu(0)

    target = "llvm"
    shapes = {'x': shape, 'y': shape}

    sym_json = tg.compile_graph(lib_fname, sym, target, shapes)
    remote.upload(lib_fname)
    param_blob = bytearray(open(param_fname, "rb").read())

    rm = tg.remote_load_exec(remote,
                             sym_json,
                             os.path.basename(lib_fname),
                             param_blob,
                             ctx)

    run, get_output = rm['run'], rm['get_output']

    nc = tvm.nd.array(np.zeros(shape, dtype=dtype), ctx)
    run()
    get_output(0, nc)

    np.testing.assert_allclose(
        nc.asnumpy(), np.exp(na.asnumpy() + nb.asnumpy()))
    server.terminate()

if __name__ == "__main__":
    test_rpc_executor()
