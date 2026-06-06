/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/tir/target_builtin/cuda.h
 * \brief TIR builtin intrinsics specific to CUDA target.
 */
#ifndef TVM_TIRX_TARGET_BUILTIN_CUDA_H_
#define TVM_TIRX_TARGET_BUILTIN_CUDA_H_

#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {
namespace builtin {

// TODO(tvm-team) TensorCore specific intrinsics should be directly registered under
//                cuda. namespace and used through op.
/*!
 * \brief tvm intrinsic for tensor core load operators.
 *
 *  void tvm_load_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                            Expr index, Expr buffer_ptr, Expr stride,
 *                            StringImm layout) {
 *    // m, n, k are the shape of wmma fragment.
 *    // Determine fragment layout(column-major or row major) by layout.
 *    // fragments must be in 'wmma.matrix_a' or 'wmma.matrix_b' scope.
 *    nvcuda::wmma::load_matrix_sync(fragment[index], buffer_ptr, stride);
 *  }
 */
TVM_DLL const Op& tvm_load_matrix_sync();

/*!
 * \brief tvm intrinsic for tensor core mma_sync operators.
 *
 *  void tvm_mma_sync(Var fragment_d, Expr index_d,
 *                    Var fragment_a, Expr index_a,
 *                    Var fragment_b, Expr index_b,
 *                    Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::mma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
TVM_DLL const Op& tvm_mma_sync();

/*!
 * \brief tvm intrinsic for tensor core bmma_sync operators.
 *
 *  void tvm_bmma_sync(Var fragment_d, Expr index_d,
 *                     Var fragment_a, Expr index_a,
 *                     Var fragment_b, Expr index_b,
 *                     Var fragment_c, Expr index_c) {
 *    nvcuda::wmma::bmma_sync(fragment_d[index_d], fragment_a[index_a],
 *                           fragment_b[index_b], fragment_c[index_c]);
 *  }
 */
TVM_DLL const Op& tvm_bmma_sync();

/*!
 * \brief tvm intrinsic for tensor core fill_fragment operators.
 *
 *  void tvm_fill_fragment(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                         Expr index, Expr value) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::fill_fragment(fragment[index], value);
 *  }
 */
TVM_DLL const Op& tvm_fill_fragment();

/*!
 * \brief tvm intrinsic for tensor core store operators.
 *
 *  void tvm_store_matrix_sync(Var fragment, UIntImm m, UIntImm, n, UIntImm k,
 *                             Expr index, Expr buffer_ptr, Expr stride,
 *                             StringImm layout) {
 *    // m, n, k are the shape of wmma fragment
 *    // fragments must be in 'wmma.accumulator' scope.
 *    nvcuda::wmma::store_matrix_sync(fragment[index], buffer_ptr, stride, layout);
 *  }
 */
TVM_DLL const Op& tvm_store_matrix_sync();

/*!
 * \brief tvm intrinsic for ptx tensor core mma instructions.
 *
 *  void ptx_mma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index, bool saturate);
 */
TVM_DLL const Op& ptx_mma();

/*!
 * \brief ptx mma / ldmatrix / mma_store / mma_fill variants that take
 *  ``(ptr_var, offset)`` pairs (not a folded access_ptr Call). Codegen
 *  emits ``ptr + offset`` C pointer arithmetic; ``lower_warp_memory``
 *  rewrites the offset's group component to its thread-local index.
 */
TVM_DLL const Op& ptx_mma_legacy();
TVM_DLL const Op& ptx_ldmatrix_legacy();
TVM_DLL const Op& mma_store_legacy();
TVM_DLL const Op& mma_fill_legacy();

/*!
 * \brief tvm intrinsic for ptx predicate load with 32-bit data type.
 *
 */
TVM_DLL const Op& ptx_ldg32();

/*!
 * \brief tvm intrinsic for sparse tensor core ptx instructions.
 *
 * void ptx_mma_sp(StringImm shape, StringImm A_layout, StringImm B_layout,
 *                 StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *                 Var multiplicand_a, Expr a_index,
 *                 Var multiplicand_b, Expr b_index,
 *                 Var accumulator, Expr c_index,
 *                 Var metadata, Expr meta_index,
 *                 Var sparse_selector, bool saturate);
 */
TVM_DLL const Op& ptx_mma_sp();

/*!
 * \brief tvm intrinsic for ptx load matrix from shared memory.
 *
 * void ptx_ldmatrix(Bool trans, IntImm num, StringImm type,
 *                   Var local_ptr, Expr local_offset,
 *                   Var smem_ptr, Expr smem_offset);
 */
TVM_DLL const Op& ptx_ldmatrix();

/*!
 * \brief tvm intrinsics for ptx async copy from global to shared memory using cp.async
 *
 * void ptx_cp_async(Var shared_ptr,
 *                   Expr shared_offset,
 *                   Var global_ptr,
 *                   Expr global_offset,
 *                   size_t bytes);
 */
TVM_DLL const Op& ptx_cp_async();

/*!
 * \brief tvm intrinsics for ptx async copy from global to shared memory using cp.async.bulk
 *
 * void ptx_cp_async_bulk(Var shared_ptr,
 *                   Expr shared_offset,
 *                   Var global_ptr,
 *                   Expr global_offset,
 *                   size_t bytes,
 *                   int barrier_arr_id,
 *                   int barrier_id);
 */
TVM_DLL const Op& ptx_cp_async_bulk();

/*!
 * \brief tvm intrinsics for ptx async bulk copy from shared::cta to shared::cluster
 *
 * void ptx_cp_async_bulk_shared_to_cluster(Expr dst_ptr,
 *                                          Expr src_ptr,
 *                                          Expr size,
 *                                          Expr mbar);
 */
TVM_DLL const Op& ptx_cp_async_bulk_shared_to_cluster();

/*!
 * \brief tvm intrinsics for ptx async copy commit and wait.
 *
 * void ptx_cp_async_commit_group();
 * void ptx_cp_async_wait_group(int num);
 *
 */
TVM_DLL const Op& ptx_cp_async_commit_group();
TVM_DLL const Op& ptx_cp_async_wait_group();

/*!
 * \brief tvm intrinsics for ptx async copy barrier using cp.async.mbarrier.arrive
 *
 * ptx_cp_async_mbarrier_arrive(int barrier_arr_id, int barrier_id)
 *
 */
TVM_DLL const Op& ptx_cp_async_mbarrier_arrive();

/*!
 * \brief PTX fence instruction: fence.{sem}.{scope}
 *
 * ptx_fence(StringImm sem, StringImm scope)
 */
TVM_DLL const Op& ptx_fence();

/*!
 * \brief PTX fence.proxy.async instruction: fence.proxy.async[.{space}]
 *
 * ptx_fence_proxy_async(StringImm space)
 */
TVM_DLL const Op& ptx_fence_proxy_async();

/*!
 * \brief tvm instrinsics to call mbarrier.init.shared::cta.b64
 *
 * ptx_mbarrier_init(uint64_t* bar_ptr, int thread_count)
 */
TVM_DLL const Op& ptx_mbarrier_init();

/*!
 * \brief tvm instrinsics to call
 *             mbarrier.arrive.shared::cta.b64
 * or
 *             @p mapa.shared::cluster.u32
 *             @p mbarrier.arrive.shared::cluster.b64
 */
TVM_DLL const Op& ptx_mbarrier_arrive();

/*!
 * \brief tvm instrinsics to call
 *              mbarrier.arrive.expect_tx.shared.b64
 * or
 *             @p mapa.shared::cluster.u32
 *             @p mbarrier.arrive.expect_tx.shared.b64
 *
 * ptx_mbarrier_arrive_expect_tx(uint64_t* bar_ptr, int byte_count)
 */
TVM_DLL const Op& ptx_mbarrier_arrive_expect_tx();

/*!
 * \brief tvm instrinsics to call mbarrier.try_wait.parity repeatedly until it returns true
 *
 * ptx_mbarrier_try_wait(uint64_t* bar_ptr, int phase)
 */
TVM_DLL const Op& ptx_mbarrier_try_wait();

/*!
 * \brief tvm instrinsics to call bar.arrive a, b
 *
 * bar_arrive(int name_bar_id, int thread_count)
 */
TVM_DLL const Op& ptx_bar_arrive();

/*!
 * \brief tvm instrinsics to call bar.sync a, {b}
 *
 * bar_sync(int name_bar_id, int thread_count)
 */
TVM_DLL const Op& ptx_bar_sync();

/*!
 * \brief tvm instrinsics to call
 * cp.async.bulk.tensor.dim.shared::cluster.global.tile.mbarrier::complete_tx::bytes
 *
 * TMA alignment requirement:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
 *
 * ptx_cp_async_bulk_tensor_global_to_cluster(int dim, PrimExpr dst_ptr, PrimExpr bar_ptr,
 * PrimExpr tensormap_addr, int...coords, int cta_mask, int cta_group, string cache_hint)
 */
TVM_DLL const Op& ptx_cp_async_bulk_tensor_global_to_cluster();

/*!
 * \brief tvm intrinsic to call
 * cp.async.bulk.tensor.dim.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes
 *
 * TMA alignment requirement:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
 *
 * ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(int dim, PrimExpr dst_ptr, PrimExpr
 * bar_ptr, PrimExpr tensormap_addr, int...coords, int cta_mask, int cta_group, string cache_hint)
 */
TVM_DLL const Op& ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster();

/*!
 * \brief tvm instrinsics to call
 * cp.async.bulk.tensor.dim.global.shared::cta.tile。bulk_group
 *
 * TMA alignment requirement:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
 *
 * ptx_cp_async_bulk_tensor_shared_to_global(int dim, PrimExpr src_ptr, PrimExpr tensormap_addr,
 * int...coords, string cache_hint)
 */
TVM_DLL const Op& ptx_cp_async_bulk_tensor_shared_to_global();

/*!
 * \brief tvm instrinsics to call
 * cp.async.bulk.prefetch.tensor.dim.L2.global.tile
 *
 * TMA alignment requirement:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#table-alignment-multi-dim-tma
 *
 * ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(int dim, PrimExpr tensormap_addr,
 * int...coords, string cache_hint)
 */
TVM_DLL const Op& ptx_cp_async_bulk_tensor_global_to_cluster_prefetch();

/*!
 * \brief tvm instrinsics to call
 * cp.reduce.async.bulk.tensor.dim.dst.src.redOp
 *
 * ptx_cp_async_bulk_tensor_shared_to_global_reduce(int dim, PrimExpr src_ptr, PrimExpr
 * tensormap_addr, int...coords, string cache_hint)
 */
TVM_DLL const Op& ptx_cp_async_bulk_tensor_shared_to_global_reduce();

/*!
 * \brief tvm instrinsics to call cp.async.bulk.commit_group
 *
 * ptx_cp_async_bulk_commit_group()
 */
TVM_DLL const Op& ptx_cp_async_bulk_commit_group();

/*!
 * \brief tvm instrinsics to call cp.async.bulk.wait_group{.read} N
 *
 * ptx_cp_async_bulk_wait_group(int N, bool read)
 */
TVM_DLL const Op& ptx_cp_async_bulk_wait_group();

/*!
 * \brief tvm instrinsics to call barrier.cluster.arrive{.sem}{.aligned}
 *
 * ptx_barrier_cluster_arrive(string sem, bool aligned)
 */
TVM_DLL const Op& ptx_barrier_cluster_arrive();

/*!
 * \brief tvm instrinsics to call barrier.cluster.wait.{acquire}{.aligned}
 *
 * ptx_barrier_cluster_wait(bool acquire, bool aligned)
 */
TVM_DLL const Op& ptx_barrier_cluster_wait();

/*!
 * \brief tvm instrinsics to call elect.sync _|p, membermask and return the predicate
 *
 * elect_sync(membermask)
 */
TVM_DLL const Op& ptx_elect_sync();

/*!
 * \brief PTX fence.mbarrier_init.release.cluster instruction
 *
 * ptx_fence_mbarrier_init()
 */
TVM_DLL const Op& ptx_fence_mbarrier_init();

/*!
 * \brief tvm instrinsics to fetch PTX pre-defined registers
 *
 * ptx_fetch_register(int bits, string reg_name)
 */
TVM_DLL const Op& ptx_fetch_register();

/*!
 * \brief PTX programmatic dependent launch synchronization.
 */
TVM_DLL const Op& ptx_griddepcontrol_wait();
TVM_DLL const Op& ptx_griddepcontrol_launch_dependents();

/*!
 * \brief tvm intrinsic for storing the result of PTX MMA into a destination pointer.
 *        For example, if each thread in a warp of size 32 has 4 elements from the result of
 *        m16xn8xk16 MMA in its registers, this intrinsic can be used to store the result in a
 *        16x8 region in shared or global memory.
 *
 *        There is no real PTX instruction that does that, but we want to hide details of
 *        complex index manipulation behind this intrinsic to simplify TIR lowering passes (e.g.
 *        LowerWarpMemory).
 *
 * void mma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr src_offset, Var dst_stride);
 */
TVM_DLL const Op& mma_store();

/*!
 * \brief tvm intrinsic for zero-initializing an MMA accumulation register.
 *        For example, if each thread in a warp of size 32 has 8 elements from the A matrix in
 *        m16xn8xk16 MMA in its registers, this intrinsic can be used to zero-initialize its
 *        4 accumulation registers.
 *
 *        There is no real PTX instruction that does that, but we introduce this intrinsic for the
 *        same reason as mma_store above.
 *
 * void mma_fill(IntImm local_size, Var local_ptr, Expr offset);
 */
TVM_DLL const Op& mma_fill();

/*!
 * \brief tvm intrinsic to encode matrix descriptor for wgmma instructions.
 *
 * ptx_wgmma_encode_matrix_descriptor(PrimExpr ptr, PrimExpr ldo, PrimExpr sdo, int swizzle)
 */
TVM_DLL const Op& ptx_wgmma_encode_matrix_descriptor();

/*!
 * \brief tvm intrinsic to call "" : "+r"(reg) :: "memory"
 *
 * ptx_wgmma_noop_barrier()
 */
TVM_DLL const Op& ptx_wgmma_noop_barrier();

/*!
 * \brief tvm intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
 * where both A and B are in shared memory.
 *
 * ptx_wgmma_mma_async_ss()
 */
TVM_DLL const Op& ptx_wgmma_mma_async_ss();

/*!
 * \brief tvm intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
 * where A is in register and B is in shared memory.
 *
 * ptx_wgmma_mma_async_rs()
 */
TVM_DLL const Op& ptx_wgmma_mma_async_rs();

/*!
 * \brief tvm intrinsic to call wgmma.fence.sync.aligned;
 *
 * ptx_wgmma_fence()
 */
TVM_DLL const Op& ptx_wgmma_fence();

/*!
 * \brief tvm intrinsic to call wgmma.commit_group.sync.aligned;
 *
 * ptx_wgmma_commit_group()
 */
TVM_DLL const Op& ptx_wgmma_commit_group();

/*!
 * \brief tvm intrinsic to call wgmma.wait_group.sync.aligned;
 *
 * ptx_wgmma_wait_group(int N)
 */
TVM_DLL const Op& ptx_wgmma_wait_group();

/*!
 * \brief tvm intrinsic to call stmatrix.sync.aligned.m8n8.num{.trans}.shared.b16 [p], r;
 *
 * ptx_stmatrix(int num, bool trans, PrimExpr ptr, PrimExpr... vars)
 */
TVM_DLL const Op& ptx_stmatrix();

/*!
 * \brief tvm intrinsic to call setmaxnreg.action.sync.aligned.u32 imm-reg-count
 */
TVM_DLL const Op& ptx_setmaxnreg();

/*!
 * \brief tvm intrinsic to call ld.global.acquire.gpu.b32
 *
 * ptx_ld_global_acquire()
 */
TVM_DLL const Op& ptx_ld_global_acquire();

/*!
 * \brief tvm instrinsics to call tcgen05.alloc.cta_group.sync.aligned;
 *
 * ptx_tcgen05_alloc(Var dst_ptr, int n_cols, int cta_group)
 */
TVM_DLL const Op& ptx_tcgen05_alloc();

/*!
 * \brief tvm instrinsics to call tcgen05.dealloc.cta_group.sync.aligned;
 *
 * ptx_tcgen05_dealloc(uint32_t taddr, int n_cols, int cta_group)
 */
TVM_DLL const Op& ptx_tcgen05_dealloc();

/*!
 * \brief tvm instrinsics to call tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
 *
 * ptx_tcgen05_relinquish_alloc_permit(int cta_group)
 */
TVM_DLL const Op& ptx_tcgen05_relinquish_alloc_permit();

/*!
 * \brief tvm instrinsics to call tcgen05.fence::before_thread_sync;
 *
 * ptx_tcgen05_fence_before_thread_sync()
 */
TVM_DLL const Op& ptx_tcgen05_fence_before_thread_sync();

/*!
 * \brief tvm instrinsics to call tcgen05.fence::after_thread_sync;
 *
 * ptx_tcgen05_fence_after_thread_sync()
 */
TVM_DLL const Op& ptx_tcgen05_fence_after_thread_sync();

/*!
 * \brief tvm instrinsics to call tcgen05.ld.sync.aligned;
 *
 * ptx_tcgen05_ld()
 */
TVM_DLL const Op& ptx_tcgen05_ld();

/*!
 * \brief tvm instrinsics to call tcgen05.st.sync.aligned;
 *
 * ptx_tcgen05_st()
 */
TVM_DLL const Op& ptx_tcgen05_st();

/*!
 * \brief tvm instrinsics to call tcgen05.wait::ld.sync.aligned;
 *
 * ptx_tcgen05_wait_ld()
 */
TVM_DLL const Op& ptx_tcgen05_wait_ld();

/*!
 * \brief tvm instrinsics to call tcgen05.wait::st.sync.aligned;
 *
 * ptx_tcgen05_wait_st()
 */
TVM_DLL const Op& ptx_tcgen05_wait_st();

/*!
 * \brief tvm intrinsic to encode matrix descriptor for tcgen05 instructions.
 *
 * ptx_tcgen05_encode_matrix_descriptor(PrimExpr ptr, PrimExpr ldo, PrimExpr sdo, int swizzle)
 */
TVM_DLL const Op& ptx_tcgen05_encode_matrix_descriptor();

/*!
 * \brief tvm intrinsic to encode instruction descriptor for tcgen05 MMA.
 *
 * ptx_tcgen05_encode_instr_descriptor(PrimExpr desc, string d_dtype, string a_dtype, string
 * b_dtype, int M, int N, int K, bool trans_a, bool trans_b, int n_cta_groups, bool neg_a, bool
 * neg_b, bool sat_d, bool is_sparse)
 */
TVM_DLL const Op& ptx_tcgen05_encode_instr_descriptor();

/*!
 * \brief tvm intrinsic to encode instruction descriptor for tcgen05 MMA block scaled.
 *
 * ptx_tcgen05_encode_instr_descriptor_block_scaled(PrimExpr desc, string d_dtype,
 * string a_dtype, string b_dtype, string sfa_dtype, string stb_dtype,
 * int M, int N, int K, bool trans_a, bool trans_b,
 * int n_cta_groups, bool neg_a, bool neg_b, bool is_sparse)
 */
TVM_DLL const Op& ptx_tcgen05_encode_instr_descriptor_block_scaled();

/*!
 * \brief tvm intrinsic to call tcgen05.mma.cta_group.kind without block scaling.
 *
 * ptx_tcgen05_mma()
 */
TVM_DLL const Op& ptx_tcgen05_mma();

/*!
 * \brief tvm intrinsic to call tcgen05.mma.cta_group.kind.block_scale{.scale_vec_size}
 *
 * ptx_tcgen05_mma_block_scale()
 */
TVM_DLL const Op& ptx_tcgen05_mma_block_scale();

/*!
 * \brief tvm intrinsic to call tcgen05.mma.sp.cta_group.kind without block scaling.
 *
 * ptx_tcgen05_mma_sp()
 */
TVM_DLL const Op& ptx_tcgen05_mma_sp();

/*!
 * \brief tvm intrinsic to call tcgen05.mma.sp.cta_group.kind.block_scale{.scale_vec_size}
 *
 * ptx_tcgen05_mma_sp_block_scale()
 */
TVM_DLL const Op& ptx_tcgen05_mma_sp_block_scale();

/*!
 * \brief tvm instrinsics to call tcgen05.commit.cta_group
 *
 * ptx_tcgen05_commit()
 */
TVM_DLL const Op& ptx_tcgen05_commit();

/*!
 * \brief tvm instrinsics to call tcgen05.cp.cta_group
 *
 * ptx_tcgen05_cp()
 */
TVM_DLL const Op& ptx_tcgen05_cp();

/*!
 * \brief tvm instrinsics to call tcgen05.shift.cta_group.down
 *
 * ptx_tcgen05_shift()
 */
TVM_DLL const Op& ptx_tcgen05_shift();

/*!
 * \brief tvm instrinsics to call map_shared_rank
 *
 * ptx_map_shared_rank(PrimExpr ptr, int rank)
 */
TVM_DLL const Op& ptx_map_shared_rank();

/*!
 * \brief tvm instrinsics to call a CUDA function. Source code is provided as a string.
 *
 * cuda_func_call(String func_name, PrimExpr... args, String source_code)
 */
TVM_DLL const Op& cuda_func_call();

/*!
 * \brief nvshmem intrinsics for nvshmem_my_pe() operation.
 *
 * int nvshmem_my_pe()
 */
TVM_DLL const Op& nvshmem_my_pe();

/*!
 * \brief nvshmem intrinsics for nvshmem_n_pes() operation.
 *
 * int nvshmem_n_pes()
 */
TVM_DLL const Op& nvshmem_n_pes();

/*!
 * \brief nvshmem intrinsics for nvshmem_getmem_nbi() operation.
 *
 * void nvshmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_getmem_nbi();

/*!
 * \brief nvshmem intrinsics for nvshmem_putmem_nbi() operation.
 *
 * void nvshmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_nbi();

/*!
 * \brief nvshmem intrinsics for nvshmemx_getmem_nbi_warp() operation.
 *
 * void nvshmemx_getmem_nbi_warp(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_getmem_nbi_warp();

/*!
 * \brief nvshmem intrinsics for nvshmemx_putmem_nbi_warp() operation.
 *
 * void nvshmemx_putmem_nbi_warp(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_nbi_warp();

/*!
 * \brief nvshmem intrinsics for nvshmemx_getmem_nbi_block() operation.
 *
 * void nvshmemx_getmem_nbi_block(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_getmem_nbi_block();

/*!
 * \brief nvshmem intrinsics for nvshmemx_putmem_nbi_block() operation.
 *
 * void nvshmemx_putmem_nbi_block(void *dest, const void *source, size_t nelems, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_nbi_block();

/*!
 * \brief nvshmem intrinsics for nvshmemx_signal_op() operation.
 *
 * void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
 */
TVM_DLL const Op& nvshmem_signal_op();

/*!
 * \brief nvshmem intrinsics for nvshmem_FuncParam{TYPENAME}_wait_until() operation.
 *
 * void nvshmem_FuncParam{TYPENAME}_wait_until(TYPE *ivar, int cmp, TYPE cmp_value)
 */
TVM_DLL const Op& nvshmem_wait_until();

/*!
 * \brief nvshmem intrinsics for nvshmem_quiet() operation.
 *
 * void nvshmem_quiet()
 */
TVM_DLL const Op& nvshmem_quiet();

/*!
 * \brief nvshmem intrinsics for nvshmemx_putmem_signal_nbi() operation.
 *
 * void nvshmemx_putmem_signal_nbi(void *dest, const void *source, size_t nelems, uint64_t
 * *sig_addr, uint64_t signal, int sig_op, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_signal_nbi();

/*!
 * \brief nvshmem intrinsics for nvshmemx_putmem_signal_nbi_warp() operation.
 *
 * void nvshmemx_putmem_signal_nbi_warp(void *dest, const void *source, size_t nelems, uint64_t
 * *sig_addr, uint64_t signal, int sig_op, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_signal_nbi_warp();

/*!
 * \brief nvshmem intrinsics for nvshmemx_putmem_signal_nbi_block() operation.
 *
 * void nvshmemx_putmem_signal_nbi_block(void *dest, const void *source, size_t nelems,
 * uint64_t *sig_addr, uint64_t signal, int sig_op, int pe)
 */
TVM_DLL const Op& nvshmem_putmem_signal_nbi_block();

/*!
 * \brief nvshmem intrinsics for nvshmem_fence() operation.
 *
 * void nvshmem_fence()
 */
TVM_DLL const Op& nvshmem_fence();

/*!
 * \brief nvshmem intrinsics for nvshmem_barrier_all() operation.
 *
 * void nvshmem_barrier_all()
 */
TVM_DLL const Op& nvshmem_barrier_all();

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TARGET_BUILTIN_CUDA_H_
