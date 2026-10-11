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
"""Ownership and address-space boundaries of explicit RPC values."""

import gc

import numpy as np
import pytest

import tvm
from tvm import rpc

pytestmark = pytest.mark.skipif(
    tvm.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)


@pytest.fixture(params=["local", "socket"])
def sessions(request):
    servers = []
    if request.param == "local":
        clients = [rpc.LocalSession(), rpc.LocalSession()]
    else:
        servers = [rpc.Server(), rpc.Server()]
        clients = [rpc.connect("127.0.0.1", server.port) for server in servers]
    try:
        yield clients
    finally:
        for client in clients:
            client.close()
        for server in servers:
            server.terminate()


def test_transfer_and_view_lifetime(sessions):
    client, _ = sessions
    device = client.cpu()
    source = np.arange(16, dtype="int32")
    tensor = device.tensor(source)
    assert isinstance(tensor, rpc.RemoteTensor)
    assert not isinstance(tensor, tvm.runtime.Tensor)
    assert tensor.device.endpoint_device == tvm.cpu()
    np.testing.assert_array_equal(tensor.numpy(), source)
    clone = tensor.copyto(device)
    with pytest.raises(ValueError, match="convert dtype"):
        device.tensor(tensor, dtype="float32")
    with pytest.raises(ValueError, match="full-slice"):
        tensor[::-1] = source
    assert tensor.copyto(tvm.cpu(3)).device == tvm.cpu(3)
    view = tensor._create_view((4,), relative_byte_offset=16)
    nested = view._create_view((2,), relative_byte_offset=4)
    del tensor, view
    gc.collect()
    np.testing.assert_array_equal(nested.numpy(), source[5:7])
    nested.copyfrom(np.array([51, 61], dtype="int32"))
    np.testing.assert_array_equal(nested.numpy(), [51, 61])
    np.testing.assert_array_equal(clone.numpy(), source)
    local = clone.copyto(tvm.cpu())
    assert isinstance(local, tvm.runtime.Tensor)
    np.testing.assert_array_equal(local.numpy(), source)
    with pytest.raises(ValueError, match="bounds"):
        nested._create_view((3,))
    with pytest.raises(ValueError, match="bounds"):
        nested._create_view((1,), relative_byte_offset=-1)
    with pytest.raises(ValueError, match="Negative"):
        device.empty((-1,))
    with pytest.raises(ValueError, match="overflow"):
        device.empty((2**62, 8))


def test_foreign_values_rejected_before_handle_use(sessions):
    first, second = sessions
    a, b = first.cpu(), second.cpu()
    source, destination = a.tensor(np.ones(4)), b.empty((4,), "float64")
    echo = second.get_function("testing.echo")
    foreign_object = first.get_function("ffi.Shape")(4, 5)
    for value in [a, source, foreign_object, first._sess]:
        with pytest.raises(ValueError, match="different remote session"):
            echo(value)
    with pytest.raises(ValueError, match="different remote sessions"):
        source.copyto(destination)
    with pytest.raises(ValueError, match="local tensor"):
        echo(tvm.runtime.tensor(np.ones(4)))
    with pytest.raises((TypeError, ValueError)):
        tvm.runtime.empty((4,), device=a)
    with pytest.raises((TypeError, ValueError)):
        tvm.runtime.from_dlpack(source)
    with pytest.raises(TypeError, match="DLPack"):
        source.__dlpack__()
    stream = a.create_stream()
    with pytest.raises(ValueError, match="different remote device or session"):
        b.sync(stream)
    a.set_stream(stream)
    alias = a.current_stream()
    a.sync(alias)
    a.free_stream(stream)
    with pytest.raises(ValueError, match="closed"):
        a.sync(stream)
    with pytest.raises(ValueError, match="closed"):
        a.sync(alias)
    a.set_stream(a.create_stream())
    gc.collect()
    current = a.current_stream()
    a.sync(current)
    a.free_stream(current)


def test_returned_alias_and_close(sessions):
    client, _ = sessions
    factory = client.get_function("rpc.test.remote_return_nd")
    get_array = factory("get_arr")
    tensor = get_array()
    alias = get_array()
    tensor.copyfrom(np.arange(10, dtype="float32"))
    np.testing.assert_array_equal(alias.numpy(), np.arange(10, dtype="float32"))
    view = alias._create_view((2,), relative_byte_offset=4)
    del tensor, alias, factory, get_array
    gc.collect()
    np.testing.assert_array_equal(view.numpy(), [1, 2])
    device = view.device
    function = client.get_function("rpc.test.addone")
    client.close()
    for operation in [lambda: function(1), view.numpy, device.sync, lambda: device.empty((1,))]:
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    del view
    gc.collect()


def test_devices_do_not_consume_global_session_slots():
    clients = [rpc.LocalSession() for _ in range(40)]
    devices = [client.cpu() for client in clients]
    del clients
    gc.collect()
    assert all(device.exist for device in devices)
    assert all(device.endpoint_device == tvm.cpu() for device in devices)
