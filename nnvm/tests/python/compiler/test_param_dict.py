import os
import numpy as np
import nnvm.compiler
import tvm
from tvm.contrib import rpc, util, graph_runtime


def test_save_load():
    x = np.random.uniform(size=(10, 2)).astype("float32")
    y = np.random.uniform(size=(1, 2, 3)).astype("float32")
    x[:] = 1
    y[:] = 1
    params = {"x": x, "y": y}
    param_bytes = nnvm.compiler.save_param_dict(params)
    assert isinstance(param_bytes, bytearray)
    param2 = nnvm.compiler.load_param_dict(param_bytes)
    assert len(param2) == 2
    np.testing.assert_equal(param2["x"].asnumpy(), x)
    np.testing.assert_equal(param2["y"].asnumpy(), y)


def test_bigendian_rpc_param():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        return

    def verify_nnvm(remote, target, shape, dtype):
        x = nnvm.sym.Variable("x")
        y = x + 1
        graph, lib, _ = nnvm.compiler.build(
            y, target,
            shape={"x": shape},
        dtype={"x": dtype})

        temp = util.tempdir()
        path_dso = temp.relpath("dev_lib.o")
        lib.save(path_dso)
        remote.upload(path_dso)
        lib = remote.load_module("dev_lib.o")
        a = np.random.randint(0, 256, size=shape).astype(dtype)
        a[:] = 1
        params = {"x" : a}
        ctx = remote.cpu(0)
        m = graph_runtime.create(graph, lib, ctx)
        # uses save param_dict
        m.load_params(nnvm.compiler.save_param_dict(params))
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype, ctx=ctx))
        np.testing.assert_allclose(a + 1, out.asnumpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_nnvm(remote, target, (10,), dtype)



if __name__ == "__main__":
    test_save_load()
    test_bigendian_rpc_param()
