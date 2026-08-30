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

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm._kernel_common import (
    _get_prefill_kernel_config,
    _get_prefill_shared_memory_usage,
)
from tvm.relax.frontend.nn.llm._prefill_kernels import (
    _attention_prefill_mla,
    _attention_prefill_ragged,
)
from tvm.relax.frontend.nn.llm.tree_attn import tree_attn, tree_attn_with_paged_kv_cache


def _get_allocated_shared_memory(func):
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    allocated_bytes = tvm.s_tir.analysis.calculate_allocated_bytes(mod)
    (function_allocations,) = allocated_bytes.values()
    return function_allocations["shared"]


def test_wide_head_prefill_fits_metal_shared_memory():
    target = tvm.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=target,
    )

    _, _, _, _, num_warps, tile_x, _, tile_z = config
    assert num_warps == 2
    assert tile_z == 8
    assert _get_prefill_shared_memory_usage(tile_x, tile_z, 512, "float16") == 24_928
    assert 24_928 <= int(target.attrs["max_shared_memory_per_block"])


def test_normal_head_prefill_keeps_existing_metal_config():
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=256,
        dtype="float16",
        target=tvm.target.Target("metal"),
    )

    assert config == (16, 4, 8, 32, 4, 16, 256, 16)


def test_wide_head_prefill_keeps_existing_webgpu_config():
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=tvm.target.Target("webgpu"),
    )

    assert config == (16, 4, 8, 32, 2, 8, 512, 8)


def test_wide_head_prefill_uses_target_shared_memory_limit():
    target = tvm.target.Target({"kind": "metal", "max_shared_memory_per_block": 65_536})
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=target,
    )

    assert config == (16, 4, 8, 32, 4, 8, 512, 16)
    assert _get_prefill_shared_memory_usage(8, 16, 512, "float16") == 41_568
    assert 41_568 <= int(target.attrs["max_shared_memory_per_block"])


def test_wide_head_prefill_rejects_unachievable_shared_memory_limit():
    target = tvm.target.Target({"kind": "metal", "max_shared_memory_per_block": 1_024})

    with pytest.raises(ValueError, match="target allows 1024 bytes"):
        _get_prefill_kernel_config(
            h_kv=1,
            h_q=8,
            d=512,
            dtype="float16",
            target=target,
        )


def test_ragged_prefill_accounts_for_wider_value_dimension():
    target = tvm.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=256,
        dtype="float16",
        target=target,
        d_v=512,
    )
    func = _attention_prefill_ragged(1, 8, 256, 512, "float16", {}, target)

    assert config == (16, 4, 8, 32, 4, 16, 256, 8)
    assert _get_prefill_shared_memory_usage(16, 8, 256, "float16", d_v=512) == 21_184
    assert _get_allocated_shared_memory(func) == 21_184
    assert 21_184 <= int(target.attrs["max_shared_memory_per_block"])


def test_mla_prefill_accounts_for_merged_kv_buffer():
    target = tvm.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=576,
        dtype="float16",
        target=target,
        d_v=512,
        merged_kv=True,
    )
    func = _attention_prefill_mla(8, 512, 64, "float16", False, target)

    assert config == (16, 4, 8, 32, 4, 8, 576, 16)
    assert (
        _get_prefill_shared_memory_usage(8, 16, 576, "float16", d_v=512, merged_kv=True) == 28_256
    )
    assert _get_allocated_shared_memory(func) == 28_256
    assert 28_256 <= int(target.attrs["max_shared_memory_per_block"])


@pytest.mark.parametrize("kernel", [tree_attn, tree_attn_with_paged_kv_cache])
def test_wide_head_tree_attention_has_legal_metal_schedule(kernel):
    target = tvm.target.Target("metal")
    func = kernel(1, 8, 512, "float16", {}, target)

    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == 24_928
    assert 24_928 <= int(target.attrs["max_shared_memory_per_block"])


if __name__ == "__main__":
    tvm.testing.main()
