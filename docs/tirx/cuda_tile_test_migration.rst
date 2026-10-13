..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

CUDA tile test migration
========================

Baseline: ``701a4b07eb85dd5b1f1355a5b910cdfd7050b803``. The inventory below
tracks each original test function, not the number of newly collected tests.
Parameter products, storage layouts, regions and numerical references remain
part of each migration. ``Pending`` means the original coverage has not yet
been restored and must not be counted as complete.

Migration rules
--------------

* Instruction selection belongs to each call site. Test helpers may spell out
  a fixed view recipe, but may not search layouts or choose an opcode.
* ``copy/test_ld_stmatrix.py`` preserves all 75 parameter combinations. The
  multi-iteration cases also assert exactly eight / ten x4 instructions.
  Assertions about the old temporary-variable names are replaced by physical
  output checks and instruction counts.
* ``copy_async/test_tcgen05_ldst.py`` preserves the original allocations and
  references. The M=64 tests on identity-layout TMEM historically accessed
  each warp's lower 16-row slab despite spelling ``tmem[0:64, ...]``. The new
  caller explicitly selects those slabs through a zero-copy view. This is
  recorded as an old implicit physical selection, not as equivalence with a
  literal first-64-rows slice.
* The old 16-bit round-trip comments claimed native pack/unpack; the emitter
  actually used ordinary b32 instructions with dense payloads. Those payload
  tests remain ordinary b32. Native pack/unpack is tested independently by
  staging and reading back ordinary b32 cells in ``test_pattern_transfers.py``.
* ``test_layout_B_rejects_partial_column_copy`` becomes the positive
  ``test_layout_B_partial_column_copy``: two explicitly selected warps can
  access the first column half. Invalid hardware ownership and out-of-bounds
  negative tests remain negative.
* Native TMEM tests compare each load/store direction separately with a host
  reconstruction. Store tests initialize all cells and check untouched rows
  and columns. The upper halves written by unpack are outside its defined
  16-bit payload and are not used as numerical evidence.

* ``copy/test_reg.py`` preserves all 69 original parameter cases and adds a
  second explicit-width variant. The former solver-layout checks now compare
  every physical register/shared address after the caller's view, including
  the warpgroup slice ``[:, 8:40]`` and the tcgen05 epilogue layout. The former
  hoisted-base/loop-variable checks assert eight 128-bit stores and sixteen
  64-bit epilogue stores, respectively. Shared readback and scalar-region mov
  cases now also run their numerical checks on the GPU.
* ``copy/test_gmem_smem.py`` retains the original synchronous copy loops and
  all 88 parameter cases. Former width-selection helper tests now run the
  explicitly selected legal width and reject a larger, misaligned width.
  The K-tiled shared regression additionally reads every value back with
  scalar indexing, independently of the vector stores.
* The old negative test for non-thread forced copy scope now rejects a
  missing group axis. A correctly shaped vector copy is legal at warp,
  warpgroup and CTA scope; the round-trip cases cover these scopes. Blocks
  smaller than 32 threads now have one partially active warp instead of zero
  warps. Their scalar/vector memory instructions work, while full-warp
  matrix instructions still reject them.

* ``copy_async/test_tcgen05_cp.py`` accounts for all 118 original parameter
  cases. The old decompression rejection becomes 96 native GPU checks:
  six shape/multicast combinations, four swizzles, two formats, and both
  CTA groups. Padding bytes are randomized, and unused TMEM lanes are zeroed
  and checked. These cases do not use software decompression on the device.
* The former shape-inference checks now pin the explicitly requested shape
  and its instruction count on the same original arrays. Descriptor helper
  name/temporary-buffer checks become native call-count and address-field
  checks; GPU references independently check the data placement.
* Two old cp negative cases (SW128 row ``[4:36]`` and SW64 columns ``[32:48]``)
  are legal with explicit descriptor fields. They now run bit-exact GPU
  checks using random input. The original unswizzled 64-byte row-stride
  case still fails because it disagrees with the instruction's address walk.
* The lane-tiled ``4x256b`` cases retain the original allocation and partial
  window, including checking untouched lanes. Their descriptor starts use
  the inverse swizzle to avoid applying the origin's phase twice.

* ``elementwise/test_fma.py`` preserves all eight original cases, adds actual
  ``layout=None`` allocation coverage, and verifies the ``rm`` rounding result
  exactly. A caller view states the compact local layout before applying a
  native packed instruction. Constants are explicitly encoded as two f32
  fields in a b64 operand; runtime scalar broadcasts retain their bit patterns.
* ``elementwise/test_binary.py`` preserves all 123 parameter cases, including
  the original shared offset/broadcast regions, nonzero warp/warpgroup
  selection, per-thread broadcast arrays, and split tcgen05 register layout.
  Shared staging and synchronization are caller code. Half division is a
  caller-supplied scalar composite; native half and packed f32 arithmetic
  specify their instruction and raw register representation explicitly.
* The old const-left subtraction rejection becomes a positive GPU check.
  Six shared const-left sub/div combinations that previously returned without
  executing now validate their ordered numerical results. Former packed
  auto-selection tests retain their arrays and code-generation assertions with
  an explicit opcode. The tcgen05 multiplication also gains a GPU reference.
* CUDA index calls are related to launch variables for bounds proofs, without
  changing their emitted code. Tests check all nine thread/block index forms
  at both valid boundaries and one element beyond each boundary.

Original case inventory
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 55 15

   * - Original file (relative to ``tests/python/tirx/tile/cuda``)
     - Original test
     - Migration
   * - ``copy/test_gmem_smem.py``
     - ``test_gmem_smem_roundtrip``
     - Restored
   * - ``copy/test_gmem_smem.py``
     - ``test_copy_g2s_s2g``
     - Restored
   * - ``copy/test_gmem_smem.py``
     - ``test_swizzled_smem_vec_len_must_fit_chunk``
     - Replaced by ``test_swizzled_smem_requires_width_within_chunk``
   * - ``copy/test_gmem_smem.py``
     - ``test_unaligned_strides_must_clamp_vec_len``
     - Replaced by ``test_strided_rows_reject_unaligned_vector``
   * - ``copy/test_gmem_smem.py``
     - ``test_unaligned_region_offset_must_clamp_vec_len``
     - Replaced by ``test_region_offset_requires_scalar_instruction``
   * - ``copy/test_gmem_smem.py``
     - ``test_swizzled_smem_emit_must_be_swizzle_aware``
     - Restored
   * - ``copy/test_gmem_smem.py``
     - ``test_layout_permute_copy_preserves_smem_strides``
     - Restored
   * - ``copy/test_ld_stmatrix.py``
     - ``test_ldstmatrix``
     - Restored
   * - ``copy/test_ld_stmatrix.py``
     - ``test_ldstmatrix_swizzle``
     - Restored
   * - ``copy/test_ld_stmatrix.py``
     - ``test_ldstmatrix_swizzle_multi_iter_pow2``
     - Restored
   * - ``copy/test_ld_stmatrix.py``
     - ``test_ldstmatrix_tcgen05_warpgroup_atom_emits_ldmatrix``
     - Restored
   * - ``copy/test_ld_stmatrix.py``
     - ``test_ldstmatrix_swizzle_multi_iter_linear``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_reg_roundtrip``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_reg_roundtrip_gapped_permuted_storage``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_copy_g2l_l2g_vec_load``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_vec_auto_reg_honors_cache_nc``
     - Replaced by ``test_reg_explicit_cache_nc``
   * - ``copy/test_reg.py``
     - ``test_reg_copy_linear_shared_hoists_thread_base``
     - Replaced by ``test_reg_copy_linear_shared_explicit_repeats``
   * - ``copy/test_reg.py``
     - ``test_reg_copy_wg_local_to_swizzled_shared_uses_structured_compose_apply``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_ptx_st_from_src_f32_vector_preserves_values``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_mov_handles_scalar_regions``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_copy_forced_vec_width_codegen``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_copy_forced_vec_dynamic_swizzled_shared_uses_vector_ptx``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_copy_explicit_vec_auto_uses_auto_family``
     - Replaced by ``test_copy_explicit_vector_instructions``
   * - ``copy/test_reg.py``
     - ``test_copy_forced_vec_rejects_size_mismatch``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_copy_forced_vec_rejects_non_thread_scope``
     - Replaced by ``test_copy_vector_rejects_missing_thread_axis``
   * - ``copy/test_reg.py``
     - ``test_reg_swizzle_chunk_caps_vector_width``
     - Replaced by ``test_reg_explicit_vector_width``
   * - ``copy/test_reg.py``
     - ``test_reg_synthetic_tile_matches_thread_base_plus_outer_delta``
     - Replaced by ``test_reg_view_preserves_shared_and_register_addresses``
   * - ``copy/test_reg.py``
     - ``test_reg_copy_tcgen05_d_epilogue_deposit_layout_pairing``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_reg_copy_tcgen05_d_epilogue_deposit_codegen``
     - Restored
   * - ``copy/test_reg.py``
     - ``test_reg_copy_tcgen05_d_epilogue_deposit_gpu``
     - Restored
   * - ``copy/test_vec_forced_cache.py``
     - ``test_copy_vec_256b_plain``
     - Restored
   * - ``copy/test_vec_forced_cache.py``
     - ``test_copy_vec_128b_nc_with_hints``
     - Restored
   * - ``copy/test_vec_forced_cache.py``
     - ``test_copy_vec_256b_nc_with_hints``
     - Restored
   * - ``copy/test_vec_forced_cache.py``
     - ``test_copy_vec_128b_nc_global_to_shared``
     - Restored
   * - ``copy/test_vec_forced_cache.py``
     - ``test_copy_vec_nc_rejects_non_global_src``
     - Restored
   * - ``copy_async/test_dsmem.py``
     - ``test_dsmem``
     - Restored
   * - ``copy_async/test_dsmem.py``
     - ``test_dsmem_dispatch_missing_config``
     - Restored
   * - ``copy_async/test_ldgsts.py``
     - ``test_copy_g2s_s2g_cta_vec_load``
     - Restored
   * - ``copy_async/test_ldgsts.py``
     - ``test_copy_ldgsts_predicate_zero_fill_codegen``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_shape_roundtrip_swizzled``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_shape_roundtrip_nonswizzled``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_shape_roundtrip_offsets``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_4x256b_compile_emits_shape_and_count``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_shape_config_routes_to_generic_planner``
     - Replaced by ``test_cp_explicit_shape_and_descriptor``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_shape_inferred_from_layouts_matches_explicit``
     - Replaced by ``test_cp_requested_shape_is_preserved``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_128x128b_layout_infers_wider_256b_atom``
     - Replaced by ``test_cp_128x128b_layout_accepts_explicit_256b_atom``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_4x256b_lane_tiled_partial_window``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_cta_group_2_128x256b_pair_collective``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_cta_group_2_64x128b_warpx2_pair_collective``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_4x256b_lane_tiled_layout_f_scatter``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_default_32x128b_instruction_sequence_unchanged``
     - Replaced by ``test_cp_explicit_32x128b_instruction_sequence``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_wrong_replica_for_multicast``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_illegal_shape_multicast_combo``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_missing_multicast_for_64x128b``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_non_128b_tcol_offset``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_unknown_shape``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_nonzero_tmem_lane_offset``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_decompress_on_generic_path``
     - Replaced by ``test_cp_native_decompression``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_non_16b_aligned_row_group_stride``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_non_canonical_swizzle_family``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_cp_rejects_flipped_swizzle_inner``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_single_cp``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_multi_cp_sw0_4tiles``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_align_middle_2_to_1_nvfp4_sfb``
     - Restored
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_dispatch_rejects_bad_inputs``
     - Replaced by ``test_cp_subregion_descriptor_contract``
   * - ``copy_async/test_tcgen05_cp.py``
     - ``test_multi_cp_encodes_descriptor_once_and_patches_addr``
     - Replaced by ``test_multi_cp_reuses_explicit_descriptor_fields``
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_ld_16xnb_load_fp32``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_ld_16xnb_load_fp32_large_rep``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_16xnb_roundtrip_16b``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_16xnb_roundtrip_16b_M128``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_16xnb_roundtrip_16b_layout_F``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tmem_datapath_layout_F_row_to_lane_mapping``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tmem_datapath_layout_rejects_invalid_sub_slab``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_atom_layout_apply_matches_decompose_fp32``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tmem_datapath_layout_D_row_to_lane_mapping``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tmem_datapath_layout_B_col_split_mapping``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_atom_layout_32x32b_datapath_B_mapping``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_datapath_B_layout_factories_reject_invalid_inputs``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_16xnb_sub_slab_view_read``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_layout_F_rejects_incompatible_atoms``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_layout_B_rejects_16xnb_fragment``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_layout_B_rejects_partial_column_copy``
     - Positive case
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_datapath_B_codegen``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_datapath_B_ld_st_roundtrip``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_st_16xnb_store``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_alloc_tcgen05_frag_wrapper_compiles``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_32x32b_float32_keeps_typed_register_operands``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_ldst_constant_tmem_address_is_uint32``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_tcgen05_ld_16x256b_sliced_matches_full_M128``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_copy_tmem2reg_async``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_copy_tmem2reg``
     - Restored
   * - ``copy_async/test_tcgen05_ldst.py``
     - ``test_copy_tmem2reg_sliced_local``
     - Restored
   * - ``copy_async/test_tma.py``
     - ``test_tma_case_matrix_is_complete``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_tma_codegen``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_canonicalization_does_not_create_an_oversized_box``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_sw128_gt_grouping_and_cuda_order``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_retains_coordinate_only_global_dimension``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_retains_unit_copy_dimensions_with_coordinates_and_strides``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_canonicalizes_dynamic_stage_slice_before_global_grouping``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_shared_pointer_counts_each_offset_once``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_defers_dynamic_global_dimension_bounds_to_runtime``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_dispatch_propagates_flat_bind_to_auto_coordinate_proof``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_recovers_stride_from_split_coordinate_only_dimension``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_preserves_coordinate_stride_for_interleaved_tensor_dimensions``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_recovers_leading_unit_dimension_stride``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_maximum_prefix_and_mixed_radix_issue_pointer``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_copy_tma_host_init_dtype_is_attribute``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_requires_complete_chain_and_exact_division``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_canonicalization_repairs_rank_and_unaligned_stride``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_promotion_preserves_payload_and_shared_pointer``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_promotes_before_crossing_box_blocked_inner_chain_boundary``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_promotion_rejects_unsafe_modes``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_odd_partial_and_unrepairable_box_fail_loudly``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_descriptor_cache_key_includes_promoted_dtype``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_direct_rank_1_through_5``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_retains_symbolic_outer_stride_for_runtime_encode``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_rejects_unknown_or_nonunit_implicit_inner_stride``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_view_base_offset_is_encoded_without_coordinate_rewrite``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_oob_reduce_and_tf32_configs``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_never_repairs_rank_or_descriptor_units``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_requires_box_linear_shared_slice``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_gather_uses_extracted_swizzled_slice_pointer``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_shared_pointer_counts_each_offset_once``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_allows_different_operand_ranks_with_equal_payload_bytes``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_gather_selector_encodes_each_map_selects_address_and_issues_once``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_selector_uses_first_true_and_requires_static_compatibility``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_gather_validates_target_source_and_destination_layouts``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_nested_selector_parser_printer_roundtrip_and_lowering_remap``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_prefetch_only_main_selector_descriptor``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_sparse_decode_runtime_stride_q_tail_o_counts_and_descriptor_dedup``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_sparse_decode_nope_rope_descriptor_dtypes_and_swizzles``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_sparse_decode_kv_descriptor_uses_tma_row_stride``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_sparse_decode_absent_extra_map_has_no_dummy_descriptor_or_select``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_shared_validator_static_boundary_matrix``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_shared_validator_unknown_policy_differs_for_auto_and_explicit``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_auto_defers_only_unknown_global_dimension_bounds_to_runtime``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_shared_validator_packed_and_interleave_rules``
     - Pending
   * - ``copy_async/test_tma.py``
     - ``test_explicit_gather_selector_gpu_roundtrip``
     - Pending
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_shared``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_non_commutative_const_lhs_rejected``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_shared_subcta_scope``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_local_subcta_trivial``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_vectorized``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_packed_f32x2_auto_dispatch``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_warpgroup_wg_local_layout``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_op_warpgroup_wg_local_emits_packed_f32x2``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_fma_warpgroup_wg_local_emits_packed_f32x2``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_add_f32_sm100_packed_f32x2_dispatch``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_maximum_reg``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_binary_add_f16_scalar_fallback_dispatch``
     - Migrated
   * - ``elementwise/test_binary.py``
     - ``test_mul_tcgen05_16x256b_atom_warpgroup_dispatch``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_fma_scalar_scalar``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_fma_tensor_scale_scalar_bias``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_mul_scalar_broadcast``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_add_rounding_mode``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_fma_no_layout``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_sub_tensor_tensor_rounding``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_fma_warpgroup_wg_local_layout``
     - Migrated
   * - ``elementwise/test_fma.py``
     - ``test_fma_f32_sm100_packed_f32x2_dispatch``
     - Migrated
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_shared``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_shared_subcta_scope``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_shared_with_bias_scale``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_local``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_local_with_bias_scale``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_vectorized``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_op_local_thread_wise``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_thread_local``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_warpgroup_local_view``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_warpgroup_src_layout_to_flat_uses_vec2_intrinsic``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_cta_local_view``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_local_view_sliced``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_layout_partition_and_validation``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_mixed_axes_and_subregion``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_joint_decomposition_extents_order``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_validate_extent_mismatch_rejected``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_unary_exp_f16_shared_scalar_fallback_dispatch``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_vec2_packed_dispatch``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_wg_rejects_thread_local_view``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_cta_rejects_thread_local_view``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_wg_rejects_partial_thread_coverage``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_wg_accepts_wg_level_layout``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_thread_accepts_local_view``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_tcgen05_atom_warpgroup_reg_dispatch_compiles``
     - Pending
   * - ``elementwise/test_unary.py``
     - ``test_cast_tcgen05_atom_warpgroup_gpu``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_mma_sync_is_a_registered_instruction``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_lowers_to_mma_sync``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_accumulates_c_when_beta_one``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_rejects_nonunit_alpha``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_rejects_fractional_beta``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_numerical``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_numerical_tiled``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_numerical_transpose``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_lowers_tiled``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_codegen_issue_count``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_lowers_transpose``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_codegen_transpose``
     - Pending
   * - ``gemm/test_gemm_mma_m16n8k_.py``
     - ``test_cuda_gemm_mma_rejects_unsupported_dtype``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_1``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_1_layout_f_m64``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_2``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_2_layout_b``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_2_datapath_b_readback``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_block_scaled_fp8_cta_group_1``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_block_scaled_fp8_cta_group_2``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_block_scaled_nvfp4_cta_group_1``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_block_scaled_nvfp4_cta_group_2``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_block_scaled_fp8_sf_id``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_arbitrary_tiles``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_no_swizzle_smem_descriptor_codegen``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_2_accepts_replicated_tmem_a_codegen``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta_group_2_rejects_flat_tmem_a_codegen``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_no_swizzle_col_major_a_ws_local_idesc``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_contiguous_kslice_partial_k``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_dense_fp8``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tf32_with_tfloat32_tma``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_explicit_mma_tile_changes_physical_instructions``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_explicit_mma_m_is_cluster_total_for_cta_group_2``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_block_scaled_explicit_mma_tile_splits_n``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_explicit_mma_tile_rejects_invalid_config``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_smem_desc_modes_codegen``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_weight_stationary_codegen``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_dense_descI_rejected``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta1_m64_accepts_packed_c_layout_ws``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_explicit_mma_tile_preserves_inferred_ws_datapath``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta1_m64_ws_requires_layout_e_c``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta1_m64_ws_requires_batched_a``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_batched_a_rejects_unproven_datapath``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta1_m64_accepts_batched_c_layout_ws``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_cta1_m64_packed_c_infers_weight_stationary``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_preserves_explicit_tmem_lane_bases``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_preserves_block_scale_tmem_lane_bases``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_hoisted_descriptor_uniformization``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_no_swizzle_col_major_rejects_non_128B_contiguous``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_no_swizzle_packed_16b_rejects_non_16bit_dtype``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_instr_desc_fold_mirrors_runtime_shape_rules``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_rejects_tf32_mn_major``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_rejects_non_uniform_atom_grid``
     - Pending
   * - ``gemm_async/test_gemm_async.py``
     - ``test_gemm_tcgen05_dense_descI_rejected_at_dispatch``
     - Pending
   * - ``test_explicit_patterns.py``
     - ``test_roundtrip``
     - Retained unchanged
   * - ``test_explicit_patterns.py``
     - ``test_instruction_name``
     - Retained unchanged
   * - ``test_explicit_patterns.py``
     - ``test_contract_rejections``
     - Contract updated
   * - ``test_explicit_patterns.py``
     - ``test_matrix_contract_rejections``
     - Contract updated
   * - ``test_explicit_patterns.py``
     - ``test_global_gpu``
     - Retained unchanged
   * - ``test_explicit_patterns.py``
     - ``test_matrix_gpu``
     - Retained unchanged
   * - ``test_explicit_patterns.py``
     - ``test_partial_warp_rejected_in_pipeline``
     - Retained unchanged
   * - ``test_explicit_patterns.py``
     - ``test_region_starts_and_element_offsets_gpu``
     - Retained unchanged
