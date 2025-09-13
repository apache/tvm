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


import torch
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


def test_torch_stream():
    env_stream = torch.cuda.current_stream()
    env_dev = env_stream.device
    new_stream = torch.cuda.Stream(env_dev)
    with tvm_ffi.DeviceStream(new_stream):
        assert torch.cuda.current_stream() == new_stream
        dev = tvm_ffi.device(str(env_dev))
        ffi_mod.check_stream(dev.dlpack_device_type(), dev.index, new_stream.cuda_stream)
    assert torch.cuda.current_stream() == env_stream
    dev = tvm_ffi.device(str(env_dev))
    ffi_mod.check_stream(dev.dlpack_device_type(), dev.index, env_stream.cuda_stream)


test_torch_stream()
