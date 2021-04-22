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
 * \file runtime.h
 * \brief VTA runtime library.
 */

#ifndef VTA_RUNTIME_RUNTIME_H_
#define VTA_RUNTIME_RUNTIME_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <tvm/runtime/c_runtime_api.h>
#include <vta/driver.h>

#define VTA_MEMCPY_H2D 1
#define VTA_MEMCPY_D2H 2
#define VTA_MEMCPY_D2D 3

#define VTA_DEBUG_DUMP_INSN (1 << 1)
#define VTA_DEBUG_DUMP_UOP (1 << 2)
#define VTA_DEBUG_SKIP_READ_BARRIER (1 << 3)
#define VTA_DEBUG_SKIP_WRITE_BARRIER (1 << 4)
#define VTA_DEBUG_FORCE_SERIAL (1 << 5)

#define ALLOC_ALIGNMENT 64

/*!
 * \brief Allocate data buffer.
 * \param size Buffer size.
 * \return A pointer to the allocated buffer.
 */
TVM_DLL void* VTABufferAlloc(size_t size);

/*!
 * \brief Free data buffer.
 * \param buffer The data buffer to be freed.
 */
TVM_DLL void VTABufferFree(void* buffer);

/*!
 * \brief Copy data buffer from one location to another.
 * \param from The source buffer base address.
 * \param from_offset The offset of the source buffer.
 * \param to The target buffer base address.
 * \param to_offset The offset of the target buffer.
 * \param size Size of copy.
 * \param kind_mask The memory copy kind.
 */
TVM_DLL void VTABufferCopy(const void* from, size_t from_offset, void* to, size_t to_offset,
                           size_t size, int kind_mask);

/*! \brief VTA command handle */
typedef void* VTACommandHandle;

/*! \brief Shutdown hook of VTA to cleanup resources */
TVM_DLL void VTARuntimeShutdown();

/*!
 * \brief Get thread local command handle.
 * \return A thread local command handle.
 */
TVM_DLL VTACommandHandle VTATLSCommandHandle();

/*!
 * \brief Get the buffer access pointer on CPU.
 * \param cmd The VTA command handle.
 * \param buffer The data buffer.
 * \return The pointer that can be accessed by the CPU.
 */
TVM_DLL void* VTABufferCPUPtr(VTACommandHandle cmd, void* buffer);

/*!
 * \brief Perform a write barrier to make a memory region visible to the CPU.
 * \param cmd The VTA command handle.
 * \param buffer The head buffer pointer.
 * \param elem_bits The size in bits of each element.
 * \param start The start of the region (in elements).
 * \param extent The end of the region (in elements).
 */
TVM_DLL void VTAWriteBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                             uint32_t extent);

/*!
 * \brief Perform a read barrier to a memory region visible to VTA.
 * \param cmd The VTA command handle.
 * \param buffer The head buffer pointer.
 * \param elem_bits The unit bits of each elements.
 * \param start The start of the region (in elements).
 * \param extent The end of the region (in elements).
 */
TVM_DLL void VTAReadBarrier(VTACommandHandle cmd, void* buffer, uint32_t elem_bits, uint32_t start,
                            uint32_t extent);

/*!
 * \brief Set debug mode on the command handle.
 * \param cmd The VTA command handle.
 * \param debug_flag The debug flag.
 */
TVM_DLL void VTASetDebugMode(VTACommandHandle cmd, int debug_flag);

/*!
 * \brief Perform a 2D data load from DRAM.
 *  Sizes are measured in units of vector elements.
 * \param cmd The VTA command handle.
 * \param src_dram_addr Source DRAM address.
 * \param src_elem_offset The source DRAM offset in number of unit elements.
 * \param x_size The lowest dimension (x axis) size in number of unit elements.
 * \param y_size The number of rows (y axis).
 * \param x_stride The x axis stride.
 * \param x_pad_before The start padding on x axis.
 * \param y_pad_before The start padding on y axis.
 * \param x_pad_after The end padding on x axis.
 * \param y_pad_after The end padding of y axis.
 * \param dst_sram_index Destination SRAM index.
 * \param dst_memory_type Destination memory type.
 */
TVM_DLL void VTALoadBuffer2D(VTACommandHandle cmd, void* src_dram_addr, uint32_t src_elem_offset,
                             uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                             uint32_t x_pad_before, uint32_t y_pad_before, uint32_t x_pad_after,
                             uint32_t y_pad_after, uint32_t dst_sram_index,
                             uint32_t dst_memory_type);

/*!
 * \brief Perform a 2D data store into DRAM
 *  Sizes are measured in units of vector elements.
 * \param cmd The VTA command handle.
 * \param src_sram_index Source SRAM index.
 * \param src_memory_type Source memory type.
 * \param dst_dram_addr Destination DRAM address.
 * \param dst_elem_offset The destination DRAM offset in number of unit elements.
 * \param x_size The lowest dimension (x axis) size in number of unit elements.
 * \param y_size The number of rows.
 * \param x_stride The x axis stride.
 */
TVM_DLL void VTAStoreBuffer2D(VTACommandHandle cmd, uint32_t src_sram_index,
                              uint32_t src_memory_type, void* dst_dram_addr,
                              uint32_t dst_elem_offset, uint32_t x_size, uint32_t y_size,
                              uint32_t x_stride);

/*!
 * \brief Push uop into kernel buffer.
 * In GEMM mode, do a blocked GEMM with 2d access pattern.
 * In ALU mode, do a vectorized ALU operation with 2d access pattern.
 *
 *  \code
 *
 *   DType accum[INP_BUFF_DEPTH][l][n];
 *   DType weight[WGT_BUFF_DEPTH][n][m];
 *   DType input[INP_BUFF_DEPTH][l][m];
 *   if reset_out == 1
 *    accum[dst_index] = 0
 *   elif mode == 0
 *    accum[dst_index] += GEMM(input[src_index], weight[wgt_index]);
 *   else
 *    if (use_imm)
 *      accum[dst_index] = opcode(accum[dst_index], imm_val);
 *    else
 *      accum[dst_index] = opcode(accum[dst_index], accum[src_index]);
 *
 *  \endcode
 *
 * \param mode Set to GEMM mode if set to 0, ALU mode is set to 1.
 * \param reset_out Resets the accum to 0.
 * \param dst_index The accum memory index.
 * \param src_index The input memory (gemm) / accum memory (alu) index.
 * \param wgt_index The weight memory index.
 * \param opcode The ALU opcode.
 * \param use_imm Use immediate in ALU mode if set to true.
 * \param imm_val Immediate value in ALU mode.
 */
TVM_DLL void VTAUopPush(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
                        uint32_t wgt_index, uint32_t opcode, uint32_t use_imm, int32_t imm_val);

/*!
 * \brief Mark start of a micro op loop.
 * \param extent The extent of the loop.
 * \param dst_factor The accum factor.
 * \param src_factor The input factor.
 * \param wgt_factor The weight factor.
 */
TVM_DLL void VTAUopLoopBegin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                             uint32_t wgt_factor);

/*!
 * \brief Mark end of a micro op loop.
 */
TVM_DLL void VTAUopLoopEnd();

/*!
 * \brief Push GEMM uop kernel into the command handle.
 * \param uop_handle The uop cache handle.
 * \param finit The initalization function to initialize uop.
 * \param signature The closure arguments of the finit.
 * \param nbytes Number of bytes to in the closure arguments.
 * \return 0 if success.
 */
TVM_DLL int VTAPushGEMMOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes);

/*!
 * \brief Push ALU uop kernel into the command handle.
 * \param uop_handle The uop cache handle.
 * \param finit The initalization function to initialize uop.
 * \param signature The closure arguments of the finit.
 * \param nbytes Number of bytes to in the closure arguments.
 * \return 0 if success.
 */
TVM_DLL int VTAPushALUOp(void** uop_handle, int (*finit)(void*), void* signature, int nbytes);

/*!
 * \brief Push dependence token.
 * \param cmd The VTA command handle.
 * \param from_qid The source queue.
 * \param to_qid The destination queue.
 * \return 0 if success.
 */
TVM_DLL int VTADepPush(VTACommandHandle cmd, int from_qid, int to_qid);

/*!
 * \brief Pop dependence signal.
 * \param cmd The VTA command handle.
 * \param from_qid The source queue.
 * \param to_qid The destination queue.
 * \return 0 if success.
 */
TVM_DLL int VTADepPop(VTACommandHandle cmd, int from_qid, int to_qid);

/*!
 * \brief Synchronize the command handle.
 *  Commit all the instructions to VTA and wait until
 *  the accelerator finishes its job.
 *  Perform all of the out-of-order DRAM stores.
 * \param cmd The VTA command handle.
 * \param wait_cycles The limit of poll cycles.
 *
 */
TVM_DLL void VTASynchronize(VTACommandHandle cmd, uint32_t wait_cycles);

#ifdef __cplusplus
}
#endif
#endif  // VTA_RUNTIME_RUNTIME_H_
