import os
import numpy as np
import tvm
import json
import base64
from tvm._ffi.base import py_str
from tvm import relay
from tvm import rpc
from tvm.contrib import util, graph_runtime


def test_save_load():
    x = np.ones((10, 2)).astype("float32")
    y = np.ones((1, 2, 3)).astype("float32")
    params = {"x": x, "y": y}
    param_bytes = tvm.relay.save_param_dict(params)
    assert isinstance(param_bytes, bytearray)
    param2 = tvm.relay.load_param_dict(param_bytes)
    assert len(param2) == 2
    np.testing.assert_equal(param2["x"].asnumpy(), x)
    np.testing.assert_equal(param2["y"].asnumpy(), y)


def test_ndarray_reflection():
    # Make a param dict where both keys point to the same array.
    x = np.random.uniform(size=(10, 2)).astype("float32")
    xx = tvm.nd.array(x)
    param_dict = {'xx': xx, 'x2': xx}
    assert param_dict['xx'].same_as(param_dict['x2'])
    # Serialize the param dict.
    param_dict_bytes = relay.save_param_dict(param_dict)
    # Deserialize (using `json.loads`), then verify the base64 `NDArray`
    # encoding wasn't corrupted.
    deser_param_dict = json.loads(param_dict_bytes.decode('utf-8'))
    b64_str = deser_param_dict["b64ndarrays"][0]
    decoded = py_str(base64.b64encode(base64.b64decode(b64_str)))
    assert b64_str == decoded
    # Deserialize (using `load_param_dict`), and verify the data wasn't corrupted.
    deser_param_dict = relay.load_param_dict(param_dict_bytes)
    np.testing.assert_equal(deser_param_dict['xx'].asnumpy(), xx.asnumpy())
    assert deser_param_dict['xx'] == deser_param_dict['x2']


def test_bigendian_rpc_param():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        return

    def verify_nnvm(remote, target, shape, dtype):
        x = nnvm.sym.Variable("x")
        y = x + 1
        graph, lib, _ = relay.build(
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
        m.load_params(relay.save_param_dict(params))
        m.run()
        out = m.get_output(0, tvm.nd.empty(shape, dtype=dtype, ctx=ctx))
        tvm.testing.assert_allclose(a + 1, out.asnumpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_nnvm(remote, target, (10,), dtype)


if __name__ == "__main__":
    test_save_load()
    test_ndarray_reflection()
    test_bigendian_rpc_param()
