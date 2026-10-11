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

import tvm_ffi

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType, assert_structural_equal
from tvm.script import tirx as T


def _is_buffer_binding(node, *op_names):
    return (
        isinstance(node, tvm.ir.Bind)
        and isinstance(node.value, tvm.ir.Call)
        and isinstance(node.value.op, tvm.ir.Op)
        and node.value.op.name in op_names
    )


def _get_source(func: tvm.tirx.Function) -> str:
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src


def test_ptx_cp_async_bulk_s2c_codegen():
    """Test that the ptx cp.async.bulk s2c chain emits the correct PTX instruction."""

    # fmt: off
    @T.function
    def main(A: T.Tensor((128,), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        cta_id = T.cuda.block_idx('x')
        tid = T.cuda.thread_idx('x')
        A_smem = T.alloc_shared([128], "float16")
        for i in T.serial(128):
            A_smem[i] = A[i]
                # Use the raw PTX instruction directly
        mapped = T.alloc_local([2], "uint64")
        T.ptx.mapa.u64(mapped[0], A_smem.ptr_to([0]), T.uint32(1))
        T.ptx.mapa.u64(mapped[1], A_smem.ptr_to([0]), T.uint32(1))
        dst_ptr = mapped[0]
        mbar_ptr = mapped[1]
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            T.cast(dst_ptr, "uint32"),
            A_smem.ptr_to([0]),
            T.uint32(256),  # 128 elements * 2 bytes
            T.cast(mbar_ptr, "uint32"),
        )
        # fmt: on

    src = _get_source(main)
    assert "tvm_builtin_ptx_cp_async_bulk_s2c" in src
    assert "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes" in src


def test_ptx_cp_async_bulk_s2c_codegen_address_conversion():
    """Test that the codegen correctly converts addresses to shared space."""

    # fmt: off
    @T.function
    def main(A: T.Tensor((64,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        cta_id = T.cuda.block_idx('x')
        tid = T.cuda.thread_idx('x')
        A_smem = T.alloc_shared([64], "float32")
        for i in T.serial(64):
            A_smem[i] = A[i]
        mapped = T.alloc_local([2], "uint64")
        T.ptx.mapa.u64(mapped[0], A_smem.ptr_to([0]), T.uint32(0))
        T.ptx.mapa.u64(mapped[1], A_smem.ptr_to([0]), T.uint32(0))
        dst_ptr = mapped[0]
        mbar_ptr = mapped[1]
        T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
            T.cast(dst_ptr, "uint32"),
            A_smem.ptr_to([0]),
            T.uint32(256),  # 64 * 4 bytes
            T.cast(mbar_ptr, "uint32"),
        )
        # fmt: on

    src = _get_source(main)
    # Verify address conversion to shared space
    assert "__cvta_generic_to_shared" in src
    assert "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes" in src


def test_mapa_pointer_bind_codegen():
    ptr_ty = PointerType(PrimType("uint64"), "shared")

    # fmt: off
    @T.function
    def main(A: T.Tensor((1,), "uint64")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(1,)))
        cta_id = T.cuda.block_idx('x')
        tid = T.cuda.thread_idx('x')
        mbar = T.alloc_shared([2], "uint64")
        mapped = T.alloc_local([1], "uint64")
        T.ptx.mapa.u64(mapped[0], mbar.ptr_to([0]), T.uint32(0))
        remote_ptr = T.reinterpret( mapped[0], ty=ptr_ty)
        remote_mbar = T.decl_tensor([1], "uint64", data=remote_ptr, scope="shared")
        A[0] = remote_mbar[0]
        # fmt: on

    binds = []
    decl_buffers = []
    loads = []

    def collect(node):
        if _is_buffer_binding(node, "tirx.decl_tensor"):
            decl_buffers.append(node)
        elif isinstance(node, tvm.ir.Bind) and isinstance(node.var.ty, PointerType):
            binds.append(node)
        elif isinstance(node, tvm.ir.TensorLoad):
            loads.append(node)

    tvm_ffi.structural_walk(main.body, collect)
    assert len(binds) == 1
    assert isinstance(binds[0].var.ty, PointerType)
    assert binds[0].var.ty.storage_scope == "shared"
    assert binds[0].value.ty.storage_scope == "shared"
    assert_structural_equal(binds[0].var.ty, binds[0].value.ty)
    assert len(decl_buffers) == 1
    assert decl_buffers[0].value.args[0].same_as(binds[0].var)
    assert any(load.source.same_as(decl_buffers[0].var) for load in loads)

    assert_structural_equal(
        main,
        tvm.script.from_source(
            main.script(), extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
        ),
    )
    src = _get_source(main)
    assert "uint64_t* remote_ptr" in src
    assert "A_ptr[0] = remote_ptr[0]" in src
    assert "tvm_builtin_ptx_mapa_u64" in src


if __name__ == "__main__":
    test_ptx_cp_async_bulk_s2c_codegen()
    test_ptx_cp_async_bulk_s2c_codegen_address_conversion()
    test_mapa_pointer_bind_codegen()
    print("All codegen tests passed!")
