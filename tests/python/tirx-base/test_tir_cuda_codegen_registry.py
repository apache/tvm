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

import tvm.tirx.operator.intrinsics.cuda as cuda_intrinsics


def _ops(manifest, namespace):
    return {entry["op"]: entry for entry in manifest[namespace]}


def test_cuda_codegen_registry_manifest_groups_user_visible_ops():
    manifest = cuda_intrinsics.list_registered_codegen()

    assert "internal" not in manifest
    assert {"cuda", "nvshmem", "ptx"}.issubset(manifest)

    assert "tirx.cuda.copy_bytes" in _ops(manifest, "cuda")
    assert "tirx.nvshmem.signal_op" in _ops(manifest, "nvshmem")
    assert "tirx.ptx.cp_async" in _ops(manifest, "ptx")
    assert "tirx.ptx.ldmatrix" in _ops(manifest, "ptx")
    assert "tirx.ptx.mma" in _ops(manifest, "ptx")


def test_cuda_codegen_registry_manifest_tracks_aliases():
    manifest = cuda_intrinsics.list_registered_codegen()
    ptx_ops = _ops(manifest, "ptx")

    assert "tirx.ptx_mma" in ptx_ops["tirx.ptx.mma"]["aliases"]
    assert cuda_intrinsics.get_codegen("tirx.ptx_mma") is not None
    assert cuda_intrinsics.get_codegen("tirx.ptx.mma") is not None


def test_cuda_codegen_registry_manifest_can_include_internal_ops():
    manifest = cuda_intrinsics.list_registered_codegen(include_internal=True)

    assert "internal" in manifest
    assert any(entry["op"].startswith("tirx._") for entry in manifest["internal"])
