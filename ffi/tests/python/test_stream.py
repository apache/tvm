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

import pytest
import tvm_ffi
import tvm_ffi.cpp


ffi_mod = tvm_ffi.cpp.load_inline(
    name="check_stream",
    cpp_sources="""
        void check_stream(int device_type, int device_id, uint64_t stream);
    """,
    cuda_sources=r"""
        void check_stream(int device_type, int device_id, uint64_t stream) {
            uint64_t cur_stream = reinterpret_cast<uint64_t>(TVMFFIEnvGetStream(device_type, device_id));
            TVM_FFI_ICHECK_EQ(cur_stream, stream);
        }
    """,
    functions=["check_stream"],
)


def test_raw_stream():
    device = tvm_ffi.device("cuda:0")
    stream_1 = 123456789
    stream_2 = 987654321
    with tvm_ffi.use_raw_stream(device, stream_1):
        ffi_mod.check_stream(device.dlpack_device_type(), device.index, stream_1)

        with tvm_ffi.use_raw_stream(device, stream_2):
            ffi_mod.check_stream(device.dlpack_device_type(), device.index, stream_2)

        ffi_mod.check_stream(device.dlpack_device_type(), device.index, stream_1)


def test_torch_stream():
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")

    device_id = torch.cuda.current_device()
    device = tvm_ffi.device("cuda", device_id)
    device_type = device.dlpack_device_type()
    stream_1 = torch.cuda.Stream(device_id)
    stream_2 = torch.cuda.Stream(device_id)
    with tvm_ffi.use_torch_stream(torch.cuda.stream(stream_1)):
        assert torch.cuda.current_stream() == stream_1
        ffi_mod.check_stream(device_type, device_id, stream_1.cuda_stream)

        with tvm_ffi.use_torch_stream(torch.cuda.stream(stream_2)):
            assert torch.cuda.current_stream() == stream_2
            ffi_mod.check_stream(device_type, device_id, stream_2.cuda_stream)

        assert torch.cuda.current_stream() == stream_1
        ffi_mod.check_stream(device_type, device_id, stream_1.cuda_stream)


def test_torch_current_stream():

    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")

    device_id = torch.cuda.current_device()
    device = tvm_ffi.device("cuda", device_id)
    device_type = device.dlpack_device_type()
    stream_1 = torch.cuda.Stream(device_id)
    stream_2 = torch.cuda.Stream(device_id)
    with torch.cuda.stream(stream_1):
        assert torch.cuda.current_stream() == stream_1
        with tvm_ffi.use_torch_stream():
            ffi_mod.check_stream(device_type, device_id, stream_1.cuda_stream)

        with torch.cuda.stream(stream_2):
            assert torch.cuda.current_stream() == stream_2
            with tvm_ffi.use_torch_stream():
                ffi_mod.check_stream(device_type, device_id, stream_2.cuda_stream)

        assert torch.cuda.current_stream() == stream_1
        with tvm_ffi.use_torch_stream():
            ffi_mod.check_stream(device_type, device_id, stream_1.cuda_stream)


def test_torch_graph():
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False")

    device_id = torch.cuda.current_device()
    device = tvm_ffi.device("cuda", device_id)
    device_type = device.dlpack_device_type()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device_id)
    with tvm_ffi.use_torch_stream(torch.cuda.graph(graph, stream=stream)):
        assert torch.cuda.current_stream() == stream
        ffi_mod.check_stream(device_type, device_id, stream.cuda_stream)
