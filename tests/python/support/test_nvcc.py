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

from tvm.support import nvcc


def test_tirx_reqntid_rewrites_only_the_named_ptx_entry(monkeypatch):
    source = """
// tirx.reqntid compressor_fwd_kernel 64 1 1
extern "C" __global__ void compressor_fwd_kernel() {}
"""
    ptx = b"""
.visible .entry compressor_fwd_kernel()
.maxntid 64
{
  ret;
}
.visible .entry unrelated_kernel()
.maxntid 32
{
  ret;
}
"""
    calls = []

    def compile_stub(code, *, target_format, compiler):
        calls.append((code, target_format, compiler))
        return ptx

    monkeypatch.delenv("TVM_CUDA_COMPILE_MODE", raising=False)
    monkeypatch.setattr(nvcc, "compile_cuda", compile_stub)

    result = bytes(nvcc.tvm_callback_cuda_compile(source))

    assert calls == [(source, "ptx", "nvrtc")]
    assert b".reqntid 64, 1, 1" in result
    assert b".maxntid 64" not in result
    assert b".maxntid 32" in result


def test_tirx_reqntid_rejects_nonflat_thread_extent(monkeypatch):
    monkeypatch.setattr(
        nvcc,
        "compile_cuda",
        lambda *_args, **_kwargs: pytest.fail("invalid metadata must fail before compilation"),
    )

    with pytest.raises(ValueError, match="flat CUDA thread extent"):
        nvcc.tvm_callback_cuda_compile("// tirx.reqntid kernel 64 2 1")
