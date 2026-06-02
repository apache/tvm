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
# pylint: disable=missing-function-docstring
"""Tests for cp.async.bulk.shared::cluster.shared::cta PTX instruction codegen."""

import tvm
import tvm.testing
from tvm.script import tirx as Tx


def _get_source(func: tvm.tirx.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src


def test_ptx_cp_async_bulk_s2c_codegen():
    """Test that Tx.ptx.cp_async.bulk.s2c emits the correct PTX instruction."""

    # fmt: off
    @Tx.prim_func
    def main(A: Tx.Buffer((128,), "float16")):
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        tid = Tx.thread_id([1])
        A_smem = Tx.alloc_shared([128], "float16")
        for i in Tx.serial(128):
            A_smem[i] = A[i]
                # Use the raw PTX instruction directly
        dst_ptr = Tx.ptx.map_shared_rank(A_smem.ptr_to([0]), Tx.int32(1))
        mbar_ptr = Tx.ptx.map_shared_rank(A_smem.ptr_to([0]), Tx.int32(1))
        Tx.ptx.cp_async.bulk.s2c(
            dst_ptr,
            A_smem.ptr_to([0]),
            Tx.int32(256),  # 128 elements * 2 bytes
            mbar_ptr,
        )
        # fmt: on

    src = _get_source(main)
    assert "tvm_builtin_ptx_cp_async_bulk_s2s_cluster" in src
    assert "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes" in src


def test_ptx_cp_async_bulk_s2c_codegen_address_conversion():
    """Test that the codegen correctly converts addresses to shared space."""

    # fmt: off
    @Tx.prim_func
    def main(A: Tx.Buffer((64,), "float32")):
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        tid = Tx.thread_id([1])
        A_smem = Tx.alloc_shared([64], "float32")
        for i in Tx.serial(64):
            A_smem[i] = A[i]
        dst_ptr = Tx.ptx.map_shared_rank(A_smem.ptr_to([0]), Tx.int32(0))
        mbar_ptr = Tx.ptx.map_shared_rank(A_smem.ptr_to([0]), Tx.int32(0))
        Tx.ptx.cp_async.bulk.s2c(
            dst_ptr,
            A_smem.ptr_to([0]),
            Tx.int32(256),  # 64 * 4 bytes
            mbar_ptr,
        )
        # fmt: on

    src = _get_source(main)
    # Verify address conversion to shared space
    assert "__cvta_generic_to_shared" in src
    assert "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes" in src


if __name__ == "__main__":
    test_ptx_cp_async_bulk_s2c_codegen()
    test_ptx_cp_async_bulk_s2c_codegen_address_conversion()
    print("All codegen tests passed!")
