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
import json

import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm
import tvm.testing
from tvm.error import InternalError
from tvm.relax.frontend.nn.llm.kv_cache import AttnKind, RopeMode

reserved_nseq = 4
maximum_total_seq_length = 128
prefill_chunk_size = 64
page_size = 16
num_layers = 4
num_qo_heads = 32
num_kv_heads = 4
head_dim = 64
rope_scale = 1.0
rope_theta = 1e4
device = tvm.cpu()


def _nop(*args):
    return None


def create_kv_cache(
    *,
    dtype="float16",
    head_dim_value=head_dim,
    page_size_value=page_size,
    num_layers_value=num_layers,
    rope_mode=RopeMode.NORMAL,
    attn_kind=AttnKind.MHA,
    support_sliding_window=False,
):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    dummy_func = tvm.runtime.convert(_nop)
    return fcreate(
        tvm_ffi.Shape(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size_value,
                int(support_sliding_window),
            ]
        ),
        tvm_ffi.Shape([0, num_layers_value]),
        num_qo_heads,
        num_kv_heads,
        head_dim_value,
        head_dim_value,
        tvm_ffi.Shape([int(attn_kind) for _ in range(num_layers_value)]),
        False,  # enable_kv_transfer
        int(rope_mode),
        rope_scale,
        rope_theta,
        None,  # rope_ext_factors
        tvm.runtime.empty((), dtype, device=device),
        dummy_func,  # f_transpose_append_mha
        None,  # f_transpose_append_mla
        [],  # f_attention_prefill_ragged
        [],  # f_attention_prefill
        [],  # f_attention_decode
        [],  # f_attention_prefill_sliding_window
        [],  # f_attention_decode_sliding_window
        [],  # f_attention_prefill_with_tree_mask_paged_kv
        [],  # f_attention_prefill_with_tree_mask
        [],  # f_mla_prefill
        [dummy_func],  # f_merge_inplace
        dummy_func,  # f_split_rotary
        dummy_func,  # f_copy_single_page
        dummy_func,  # f_debug_get_kv
        dummy_func,  # f_compact_copy
    )


def append_tokens(kv_cache, seq_id=0, append_length=page_size + 1):
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fadd_sequence(kv_cache, seq_id)
    fbegin_forward(kv_cache, Shape([seq_id]), Shape([append_length]), None)
    fend_forward(kv_cache)


def test_checkpoint_metadata_reports_layout_pages_and_groups():
    fget_checkpoint_metadata = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fget_layout_hash = tvm.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")
    fexport_page_group = tvm.get_global_func("vm.builtin.attention_kv_cache_export_page_group")
    fprepare_import = tvm.get_global_func("vm.builtin.attention_kv_cache_prepare_import")
    fimport_page_group = tvm.get_global_func("vm.builtin.attention_kv_cache_import_page_group")
    fget_sequence_length = tvm.get_global_func("vm.builtin.attention_kv_cache_get_sequence_length")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)

    assert metadata["cacheType"] == "relax.vm.PagedAttentionKVCache"
    assert metadata["layoutHash"] == fget_layout_hash(kv_cache)
    assert metadata["seqId"] == 0
    assert metadata["seqLength"] == page_size + 1
    assert metadata["pageSize"] == page_size
    assert metadata["dtype"] == "float16"
    assert metadata["layerBegin"] == 0
    assert metadata["layerEnd"] == num_layers
    assert metadata["numKVHeads"] == num_kv_heads
    assert metadata["qkHeadDim"] == head_dim
    assert metadata["vHeadDim"] == head_dim
    assert metadata["attnKinds"] == ["mha"] * num_layers
    assert metadata["pageTensorLayout"] == "num_total_pages,2,num_kv_heads,page_size,qk_head_dim"
    assert metadata["pageTensorShape"] == [
        metadata["numTotalPages"],
        2,
        num_kv_heads,
        page_size,
        head_dim,
    ]
    assert len(metadata["blocks"]) == 1
    assert metadata["blocks"][0]["seqLength"] == page_size + 1
    assert len(metadata["logicalPages"]) == 2
    assert metadata["logicalPages"][0]["startPos"] == 0
    assert metadata["logicalPages"][0]["length"] == page_size
    assert metadata["logicalPages"][1]["startPos"] == page_size
    assert metadata["logicalPages"][1]["length"] == 1
    assert len(metadata["groups"]) == num_layers
    assert metadata["groups"][0]["layerBegin"] == 0
    assert metadata["groups"][0]["layerEnd"] == 1
    assert metadata["groups"][0]["numLogicalPages"] == 2
    assert metadata["groups"][0]["dtype"] == "float16"
    assert metadata["groups"][0]["shape"] == [1, 2, 2, num_kv_heads, page_size, head_dim]

    group = tvm.runtime.empty(tuple(metadata["groups"][0]["shape"]), "float16", device=device)
    fexport_page_group(kv_cache, 0, 0, group)
    import_cache = create_kv_cache()
    fprepare_import(import_cache, 0, metadata_json)
    assert fget_sequence_length(import_cache, 0) == page_size + 1
    fimport_page_group(import_cache, 0, 0, group)


def test_checkpoint_layout_hash_is_stable_and_layout_sensitive():
    fget_layout_hash = tvm.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")

    kv_cache = create_kv_cache()
    same_layout = create_kv_cache()
    different_page_size = create_kv_cache(page_size_value=page_size * 2)
    different_num_layers = create_kv_cache(num_layers_value=2)
    different_head_dim = create_kv_cache(head_dim_value=128)
    different_dtype = create_kv_cache(dtype="float32")
    different_rope = create_kv_cache(rope_mode=RopeMode.NONE)

    layout_hash = fget_layout_hash(kv_cache)
    assert layout_hash == fget_layout_hash(kv_cache)
    assert layout_hash == fget_layout_hash(same_layout)
    assert layout_hash != fget_layout_hash(different_page_size)
    assert layout_hash != fget_layout_hash(different_num_layers)
    assert layout_hash != fget_layout_hash(different_head_dim)
    assert layout_hash != fget_layout_hash(different_dtype)
    assert layout_hash != fget_layout_hash(different_rope)


def test_checkpoint_metadata_rejects_unsupported_layouts_and_sequence_ids():
    fget_checkpoint_metadata = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fget_layout_hash = tvm.get_global_func("vm.builtin.attention_kv_cache_get_layout_hash")
    fexport_page_group = tvm.get_global_func("vm.builtin.attention_kv_cache_export_page_group")

    sliding_cache = create_kv_cache(support_sliding_window=True)
    append_tokens(sliding_cache)
    mla_cache = create_kv_cache(attn_kind=AttnKind.MLA)
    dst = tvm.runtime.empty((1, 1, 2, num_kv_heads, page_size, head_dim), "float16", device=device)

    with pytest.raises(InternalError, match="sliding-window"):
        fget_layout_hash(sliding_cache)
    with pytest.raises(InternalError, match="sliding-window"):
        fget_checkpoint_metadata(sliding_cache, 0)
    with pytest.raises(InternalError, match="sliding-window"):
        fexport_page_group(sliding_cache, 0, 0, dst)
    with pytest.raises(InternalError, match="sequence id 0"):
        fget_checkpoint_metadata(create_kv_cache(), 1)
    with pytest.raises(InternalError, match="full-context MHA/GQA"):
        fget_layout_hash(mla_cache)
    with pytest.raises(InternalError, match="full-context MHA/GQA"):
        fexport_page_group(mla_cache, 0, 0, dst)

    tree_cache = create_kv_cache()
    tvm.get_global_func("vm.builtin.kv_state_add_sequence")(tree_cache, 0)
    tvm.get_global_func("vm.builtin.kv_state_begin_forward")(
        tree_cache, Shape([0]), Shape([2]), Shape([-1, 0])
    )
    with pytest.raises(InternalError, match="committed token-chain state"):
        fexport_page_group(tree_cache, 0, 0, dst)


def test_checkpoint_export_page_group_validates_group_shape():
    fget_checkpoint_metadata = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fexport_page_group = tvm.get_global_func("vm.builtin.attention_kv_cache_export_page_group")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata = json.loads(fget_checkpoint_metadata(kv_cache, 0))
    shape = metadata["groups"][0]["shape"]

    with pytest.raises(InternalError, match="group id"):
        fexport_page_group(
            kv_cache,
            0,
            num_layers,
            tvm.runtime.empty(tuple(shape), "float16", device=device),
        )

    bad_shape = shape.copy()
    bad_shape[-1] += 1
    with pytest.raises(InternalError, match="ExportPageGroup expects"):
        fexport_page_group(
            kv_cache,
            0,
            0,
            tvm.runtime.empty(tuple(bad_shape), "float16", device=device),
        )

    with pytest.raises(InternalError, match="dtype mismatches"):
        fexport_page_group(
            kv_cache,
            0,
            0,
            tvm.runtime.empty(tuple(shape), "float32", device=device),
        )


def test_checkpoint_prepare_import_validates_metadata():
    fget_checkpoint_metadata = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fprepare_import = tvm.get_global_func("vm.builtin.attention_kv_cache_prepare_import")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)

    with pytest.raises(InternalError, match="sequence id 0"):
        fprepare_import(create_kv_cache(), 1, metadata_json)

    with pytest.raises(InternalError, match="dtype"):
        fprepare_import(create_kv_cache(dtype="float32"), 0, metadata_json)

    bad_length = json.loads(metadata_json)
    bad_length["seqLength"] = page_size * 2 + 1
    with pytest.raises(InternalError, match="sequence length"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_length))

    bad_group = json.loads(metadata_json)
    bad_group["groups"][0]["shape"][-1] += 1
    with pytest.raises(InternalError, match="shape"):
        fprepare_import(create_kv_cache(), 0, json.dumps(bad_group))

    metadata["layoutHash"] = "bad-layout-hash"
    with pytest.raises(InternalError, match="layout hash mismatch"):
        fprepare_import(create_kv_cache(), 0, json.dumps(metadata))


def test_checkpoint_import_page_group_validates_group_shape():
    fget_checkpoint_metadata = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_get_checkpoint_metadata"
    )
    fprepare_import = tvm.get_global_func("vm.builtin.attention_kv_cache_prepare_import")
    fimport_page_group = tvm.get_global_func("vm.builtin.attention_kv_cache_import_page_group")

    kv_cache = create_kv_cache()
    append_tokens(kv_cache)
    metadata_json = fget_checkpoint_metadata(kv_cache, 0)
    metadata = json.loads(metadata_json)
    shape = metadata["groups"][0]["shape"]

    import_cache = create_kv_cache()
    fprepare_import(import_cache, 0, metadata_json)

    with pytest.raises(InternalError, match="group id"):
        fimport_page_group(
            import_cache,
            0,
            num_layers,
            tvm.runtime.empty(tuple(shape), "float16", device=device),
        )

    bad_shape = shape.copy()
    bad_shape[-1] += 1
    with pytest.raises(InternalError, match="ImportPageGroup expects"):
        fimport_page_group(
            import_cache,
            0,
            0,
            tvm.runtime.empty(tuple(bad_shape), "float16", device=device),
        )

    with pytest.raises(InternalError, match="dtype mismatches"):
        fimport_page_group(
            import_cache,
            0,
            0,
            tvm.runtime.empty(tuple(shape), "float32", device=device),
        )


if __name__ == "__main__":
    tvm.testing.main()
