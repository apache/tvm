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
 * \file ptx.h
 * \brief Code generation with inlined PTX code.
 */
#ifndef TVM_TARGET_SOURCE_PTX_H_
#define TVM_TARGET_SOURCE_PTX_H_

#include <tvm/runtime/logging.h>

#include <string>
#include <tuple>

namespace tvm {
namespace codegen {

/*!
 * \brief Print MMA assembly string given parameters.
 * \param shape The shape string mMnNkK
 * \param A_layout The layout of multiplicand A, can be either "row" or "col".
 * \param B_layout The layout of multiplicand B, can be either "row" or "col".
 * \param A_dtype The data type of multiplicand A.
 * \param B_dtype The data type of multiplicand B.
 * \param C_dtype The data type of multiplicand C.
 * \param a_ptr Pointer to buffer A.
 * \param a_offset The offset of element in A.
 * \param b_ptr Pointer to buffer B.
 * \param b_offset The offset of element in B.
 * \param c_ptr Pointer to buffer C.
 * \param c_offset The offset of element in C.
 * \param metadata Pointer to metadata buffer (only used for sparse mma).
 * \param metadata_offset The offset of element in metadata.
 * \param sparsity_selector The sparsity selector in sparse mma.
 * \param bit_op The bit operator used in 1-bit mma, can be either "xor" or "and".
 * \param sparse Whether it's sparse mma or not.
 * \param saturate Whether saturate output or not.
 */
std::string PrintMMAAssembly(const std::string& shape, const std::string& A_layout,
                             const std::string& B_layout, const std::string& A_dtype,
                             const std::string& B_dtype, const std::string& C_dtype,
                             const std::string& a_ptr, const std::string& a_offset,
                             const std::string& b_ptr, const std::string& b_offset,
                             const std::string& c_ptr, const std::string& c_offset,
                             const std::string& metadata, const std::string& metadata_offset,
                             const std::string& sparsity_selector, const std::string& bit_op,
                             bool sparse, bool saturate);

/*!
 * \brief Print ldmatrix assembly string given parameters.
 * \param trans: whether the matrix is loaded in column major format or not.
 * \param num: number of matrices to load.
 * \param type: The data type in the matrix, .b16 is the only accepted data type.
 * \param local_ptr: pointer to local buffer.
 * \param local_elem_offset: The offset of the element to store in the local buffer.
 * \param smem_ptr: pointer to the shared memory buffer to load.
 * \param smem_elem_offset: The offset of the start element of the row to load in shared memory.
 */
std::string PrintLoadMatrixAssembly(bool trans, int num, const std::string& type,
                                    const std::string& local_ptr,
                                    const std::string& local_elem_offset,
                                    const std::string& smem_ptr,
                                    const std::string& smem_elem_offset);

/*!
 * \brief Print ptx cp.async assembly string given parameters.
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy, valid values are 4, 8, and 16.
 */
std::string PrintCpAsyncAssembly(const std::string& shared_ptr,
                                 const std::string& shared_elem_offset,
                                 const std::string& global_ptr,
                                 const std::string& global_elem_offset, const std::string& bytes);

/*!
 * \brief Print predicated ptx cp.async assembly string given parameters.
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy, valid values are 4, 8, and 16.
 * \param predicate_value: The value of predicate `@p`.
 */
std::string PrintPredicatedCpAsyncAssembly(const std::string& shared_ptr,
                                           const std::string& shared_elem_offset,
                                           const std::string& global_ptr,
                                           const std::string& global_elem_offset,
                                           const std::string& bytes,
                                           const std::string& predicate_value);

/*!
 * \brief Print ptx async copy from global to shared memory using cp.async.bulk
 * \param shared_ptr: The pointer to the destination shared memory.
 * \param shared_elem_offset: The offset into the shared memory.
 * \param global_ptr: The pointer to the global memory.
 * \param global_elem_offset: The offset into the global memory.
 * \param bytes: The number of bytes to copy.
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintCpAsyncBulkAsm(const std::string& shared_ptr,
                                const std::string& shared_elem_offset,
                                const std::string& global_ptr,
                                const std::string& global_elem_offset, const std::string& bytes,
                                const std::string& barrier);

/*!
 * \brief Print ptx async copy barrier using cp.async.mbarrier.arrive
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintCpAsyncBarrierAsm(const std::string& barrier);

/*!
 * \brief Print ptx barrier initialization of thread count using mbarrier.init
 * \param barrier: The name of the barrier in shared memory.
 * \param thread_count: The number of threads expected to arrive at the barrier.
 */
std::string PrintInitBarrierThreadCountAsm(const std::string& barrier,
                                           const std::string& thread_count);

/*!
 * \brief Print ptx barrier arrival using mbarrier.arrive
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintArriveBarrierAsm(const std::string& barrier);

/*!
 * \brief Print ptx barrier arrival with expect tx operation using mbarrier.arrive.expect_tx
 * \param barrier: The name of the barrier in shared memory.
 * \param byte_count: Increases the tx count of the mbarrier object to track completion of
 * addtional async transactions.
 */
std::string PrintArriveBarrierExpectTxAsm(const std::string& barrier,
                                          const std::string& byte_count);

/*!
 * \brief Print ptx barrier wait using mbarrier.try_wait
 * \param barrier: The name of the barrier in shared memory.
 */
std::string PrintWaitBarrierAsm(const std::string& barrier);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_PTX_H_
