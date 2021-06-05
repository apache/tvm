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
import os
import numpy as np
import tvm
from tvm import te, runtime
import json
import base64
from tvm._ffi.base import py_str
from tvm.relay.op import add
from tvm import relay
from tvm import rpc
from tvm.contrib import utils, graph_executor


def test_save_load():
    x = np.ones((10, 2)).astype("float32")
    y = np.ones((1, 2, 3)).astype("float32")
    params = {"x": x, "y": y}
    param_bytes = runtime.save_param_dict(params)
    assert isinstance(param_bytes, bytearray)
    param2 = relay.load_param_dict(param_bytes)
    assert len(param2) == 2
    np.testing.assert_equal(param2["x"].numpy(), x)
    np.testing.assert_equal(param2["y"].numpy(), y)


def test_ndarray_reflection():
    # Make two `NDArrayWrapper`s that point to the same underlying array.
    np_array = np.random.uniform(size=(10, 2)).astype("float32")
    tvm_array = tvm.nd.array(np_array)
    param_dict = {"x": tvm_array, "y": tvm_array}
    assert param_dict["x"].same_as(param_dict["y"])
    # Serialize then deserialize `param_dict`.
    deser_param_dict = relay.load_param_dict(runtime.save_param_dict(param_dict))
    # Make sure the data matches the original data and `x` and `y` contain the same data.
    np.testing.assert_equal(deser_param_dict["x"].numpy(), tvm_array.numpy())
    # Make sure `x` and `y` contain the same data.
    np.testing.assert_equal(deser_param_dict["x"].numpy(), deser_param_dict["y"].numpy())


def test_bigendian_rpc_param():
    """Test big endian rpc when there is a PowerPC RPC server available"""
    host = os.environ.get("TVM_POWERPC_TEST_HOST", None)
    port = os.environ.get("TVM_POWERPC_TEST_PORT", 9090)
    if host is None:
        return

    def verify_graph_executor(remote, target, shape, dtype):
        x = relay.var("x")
        y = relay.const(1)
        z = relay.add(x, y)
        func = relay.Function([x], z)

        x_in = np.ones(shape).astype(dtype)
        params = {"x": x_in}
        graph, lib, params = relay.build(func, target=target, params=params)

        temp = utils.tempdir()
        path_dso = temp.relpath("dev_lib.o")
        lib.save(path_dso)
        remote.upload(path_dso)
        lib = remote.load_module("dev_lib.o")
        dev = remote.cpu(0)
        mod = graph_executor.create(graph, lib, dev)
        mod.load_params(runtime.save_param_dict(params))
        mod.run()
        out = mod.get_output(0, tvm.nd.empty(shape, dtype=dtype, device=dev))
        tvm.testing.assert_allclose(x_in + 1, out.numpy())

    print("Test RPC connection to PowerPC...")
    remote = rpc.connect(host, port)
    target = "llvm -mtriple=powerpc-linux-gnu"
    for dtype in ["float32", "float64", "int32", "int8"]:
        verify_graph_executor(remote, target, (10,), dtype)


if __name__ == "__main__":
    test_save_load()
    test_ndarray_reflection()
    test_bigendian_rpc_param()
