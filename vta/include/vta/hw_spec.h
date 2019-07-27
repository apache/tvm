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
 * \file hw_spec.h
 * \brief Preprocessor definitions for VTA HLS design and runtime.
 */

#ifndef VTA_HW_SPEC_H_
#define VTA_HW_SPEC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*! Memory bus width */
#define VTA_BUS_WIDTH (1 << VTA_LOG_BUS_WIDTH)

/*! log2 of instruction data type width */
#define VTA_LOG_INS_WIDTH 7
/*! Instruction data type width */
#define VTA_INS_WIDTH (1 << VTA_LOG_INS_WIDTH)
/*! log2 of micro op data type width */
#define VTA_LOG_UOP_WIDTH 5
/*! Micro Op data type width */
#define VTA_UOP_WIDTH (1 << VTA_LOG_UOP_WIDTH)
/*! Weight data type width */
#define VTA_WGT_WIDTH (1 << VTA_LOG_WGT_WIDTH)
/*! Input data type width */
#define VTA_INP_WIDTH (1 << VTA_LOG_INP_WIDTH)
/*! Output data type width */
#define VTA_OUT_WIDTH (1 << VTA_LOG_OUT_WIDTH)
/*! Accumulator data type width */
#define VTA_ACC_WIDTH (1 << VTA_LOG_ACC_WIDTH)

/*! Batch size (corresponds to A in (A,B)x(B,C) mat mult)*/
#define VTA_BATCH (1 << VTA_LOG_BATCH)
/*! Blocking factor of inner most loop (corresponds to B in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_IN (1 << VTA_LOG_BLOCK_IN)
/*! Blocking factor of the outer loop (corresponds to C in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_OUT (1 << VTA_LOG_BLOCK_OUT)

/*! On-chip micro-op buffer size in B */
#define VTA_UOP_BUFF_SIZE (1 << VTA_LOG_UOP_BUFF_SIZE)
/*! On-chip weight buffer size in B */
#define VTA_WGT_BUFF_SIZE (1 << VTA_LOG_WGT_BUFF_SIZE)
/*! On-chip activation buffer size in B */
#define VTA_INP_BUFF_SIZE (1 << VTA_LOG_INP_BUFF_SIZE)
/*! On-chip accumulator buffer size in B */
#define VTA_ACC_BUFF_SIZE (1 << VTA_LOG_ACC_BUFF_SIZE)

/*! Input vector size in bits */
#define VTA_INP_MATRIX_WIDTH (VTA_INP_WIDTH * VTA_BATCH * VTA_BLOCK_IN)
/*! Weight vector size in bits */
#define VTA_WGT_MATRIX_WIDTH (VTA_WGT_WIDTH * VTA_BLOCK_OUT * VTA_BLOCK_IN)
/*! Accumulator vector size in bits */
#define VTA_ACC_MATRIX_WIDTH (VTA_ACC_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)
/*! Output vector size in bits */
#define VTA_OUT_MATRIX_WIDTH (VTA_OUT_WIDTH * VTA_BATCH * VTA_BLOCK_OUT)

/*! Ratio between input matrix size and axi width */
#define INP_MAT_AXI_RATIO (VTA_INP_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between weight matrix size and axi width */
#define WGT_MAT_AXI_RATIO (VTA_WGT_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between accumulator matrix size and axi width */
#define ACC_MAT_AXI_RATIO (VTA_ACC_MATRIX_WIDTH / VTA_BUS_WIDTH)
/*! Ratio between output matrix size and axi width */
#define OUT_MAT_AXI_RATIO (VTA_OUT_MATRIX_WIDTH / VTA_BUS_WIDTH)

/*! Size of instruction buffer element in B */
#define VTA_INS_ELEM_BYTES (VTA_INS_WIDTH / 8)
/*! Size of uop buffer element in B*/
#define VTA_UOP_ELEM_BYTES (VTA_UOP_WIDTH / 8)
/*! Size of activation buffer element in B*/
#define VTA_INP_ELEM_BYTES (VTA_INP_MATRIX_WIDTH / 8)
/*! Size of weight buffer element in B*/
#define VTA_WGT_ELEM_BYTES (VTA_WGT_MATRIX_WIDTH / 8)
/*! Size of accumulator buffer element in B*/
#define VTA_ACC_ELEM_BYTES (VTA_ACC_MATRIX_WIDTH / 8)
/*! Size of output buffer element in B*/
#define VTA_OUT_ELEM_BYTES (VTA_OUT_MATRIX_WIDTH / 8)

/*! On-chip micro-op buffer depth */
#define VTA_UOP_BUFF_DEPTH (VTA_UOP_BUFF_SIZE / VTA_UOP_ELEM_BYTES)
/*! log2 of on-chip micro-op buffer depth */
#define VTA_LOG_UOP_BUFF_DEPTH (VTA_LOG_UOP_BUFF_SIZE - VTA_LOG_UOP_WIDTH + 3)
// ! \brief On-chip weight buffer depth
#define VTA_WGT_BUFF_DEPTH (VTA_WGT_BUFF_SIZE / VTA_WGT_ELEM_BYTES)
/*! log2 of weight micro-op buffer depth */
#define VTA_LOG_WGT_BUFF_DEPTH \
    (VTA_LOG_WGT_BUFF_SIZE - VTA_LOG_BLOCK_OUT - VTA_LOG_BLOCK_IN - VTA_LOG_WGT_WIDTH + 3)
/*! On-chip activation buffer depth */
#define VTA_INP_BUFF_DEPTH (VTA_INP_BUFF_SIZE / VTA_INP_ELEM_BYTES)
/*! log2 of activation micro-op buffer depth */
#define VTA_LOG_INP_BUFF_DEPTH \
    (VTA_LOG_INP_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_IN - VTA_LOG_INP_WIDTH + 3)
/*! On-chip accumulator buffer depth */
#define VTA_ACC_BUFF_DEPTH (VTA_ACC_BUFF_SIZE / VTA_ACC_ELEM_BYTES)
/*! log2 of on-chip accumulator buffer depth */
#define VTA_LOG_ACC_BUFF_DEPTH \
    (VTA_LOG_ACC_BUFF_SIZE - VTA_LOG_BATCH - VTA_LOG_BLOCK_OUT - VTA_LOG_ACC_WIDTH + 3)

/*! Instruction opcode field bitwidth */
#define VTA_OPCODE_BIT_WIDTH 3
/*! ALU opcode field bitwidth */
#define VTA_ALU_OPCODE_BIT_WIDTH 2

/*! Opcode: load encoding */
#define VTA_OPCODE_LOAD 0
/*! Opcode: store encoding */
#define VTA_OPCODE_STORE 1
/*! Opcode: GEMM encoding */
#define VTA_OPCODE_GEMM 2
/*! Opcode: finish encoding */
#define VTA_OPCODE_FINISH 3
/*! Opcode: ALU encoding */
#define VTA_OPCODE_ALU 4

/*! ALU opcode: unary min op */
#define VTA_ALU_OPCODE_MIN 0
/*! ALU opcode: unary max op */
#define VTA_ALU_OPCODE_MAX 1
/*! ALU opcode: binary add op */
#define VTA_ALU_OPCODE_ADD 2
/*! ALU opcode: shift right by immediate op */
#define VTA_ALU_OPCODE_SHR 3

/*! Memory type field bitwidth */
#define VTA_MEMOP_ID_BIT_WIDTH 2
/*! Load/Store Instruction: DRAM address width*/
#define VTA_MEMOP_SRAM_ADDR_BIT_WIDTH 16
/*! Load/Store Instruction: DRAM address width*/
#define VTA_MEMOP_DRAM_ADDR_BIT_WIDTH 32
/*! Load/Store Instruction: transfer size width*/
#define VTA_MEMOP_SIZE_BIT_WIDTH 16
/*! Load/Store Instruction: stride size width*/
#define VTA_MEMOP_STRIDE_BIT_WIDTH 16
/*! Load/Store Instruction: padding width*/
#define VTA_MEMOP_PAD_BIT_WIDTH 4
/*! Load/Store Instruction: padding value encoding width*/
#define VTA_MEMOP_PAD_VAL_BIT_WIDTH 2
/*! GEMM/ALU Instruction: loop max iter bits */
#define VTA_LOOP_ITER_WIDTH 14
/*! ALU Instruction: immediate bitwidth*/
#define VTA_ALUOP_IMM_BIT_WIDTH 16
/*! ALU Instruction: shift arg bitwidth*/
#define VTA_SHR_ARG_BIT_WIDTH (VTA_LOG_ACC_WIDTH)
/*! ALU Instruction: multiply arg bitwidth*/
#define VTA_MUL_ARG_BIT_WIDTH 8

/*! Mem ID constant: uop memory */
#define VTA_MEM_ID_UOP 0
/*! Mem ID constant: weight memory */
#define VTA_MEM_ID_WGT 1
/*! Mem ID constant: input memory */
#define VTA_MEM_ID_INP 2
/*! Mem ID constant: accumulator/bias memory */
#define VTA_MEM_ID_ACC 3
/*! Mem ID constant: output store buffer */
#define VTA_MEM_ID_OUT 4

/*! GEMM Micro-op start position of the acc_idx field */
#define VTA_UOP_GEM_0_0 0
/*! GEMM Micro-op end position of the acc_idx field */
#define VTA_UOP_GEM_0_1 (VTA_UOP_GEM_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
/*! GEMM Micro-op start position of the inp_idx field */
#define VTA_UOP_GEM_1_0 (VTA_UOP_GEM_0_1 + 1)
/*! GEMM Micro-op end position of the inp_idx field */
#define VTA_UOP_GEM_1_1 (VTA_UOP_GEM_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)
/*! GEMM Micro-op start position of the wgt_idx field */
#define VTA_UOP_GEM_2_0 (VTA_UOP_GEM_1_1 + 1)
/*! GEMM Micro-op end position of the wgt_idx field */
#define VTA_UOP_GEM_2_1 (VTA_UOP_GEM_2_0 + VTA_LOG_WGT_BUFF_DEPTH - 1)

/*! GEMM Micro-op start position of the acc_idx field */
#define VTA_UOP_ALU_0_0 0
/*! GEMM Micro-op end position of the acc_idx field */
#define VTA_UOP_ALU_0_1 (VTA_UOP_ALU_0_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
/*! GEMM Micro-op start position of the inp_idx field */
#define VTA_UOP_ALU_1_0 (VTA_UOP_ALU_0_1 + 1)
/*! GEMM Micro-op end position of the inp_idx field */
#define VTA_UOP_ALU_1_1 (VTA_UOP_ALU_1_0 + VTA_LOG_INP_BUFF_DEPTH - 1)

/*! \brief VTA generic instruction */
typedef struct {
  /*! \brief The instruction opcode */
  uint64_t opcode         : VTA_OPCODE_BIT_WIDTH;
  /*! \brief Unused in this instruction */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from GEMM stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Unused in this instruction */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to GEMM stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Padding */
  uint64_t pad_0          : 64 - VTA_OPCODE_BIT_WIDTH - 4;
  /*! \brief Padding */
  uint64_t pad_1          : 64;
} VTAGenericInsn;

/*! \brief VTA load/store instruction
*   Load/store instruction can describe a 2D strided access pattern
*   with padding, which can be useful to perform spatial padding
*   on the fly on a tensor on which to perform 2D convolution.
*   For instance if we try to load a 4x4 spatial tile from a 16x16
*   matrix with padding of size 1 on all dimensions:
*   y_size = 4, x_size = 4, x_stride = 16, y_pad_0 = 1, y_pad_1 = 1,
*   x_pad_0 = 1, x_pad_1 = 1.
*/
typedef struct {
  /*! \brief The instruction opcode */
  uint64_t opcode         : VTA_OPCODE_BIT_WIDTH;
  /*! \brief Unused in this instruction */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from GEMM stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Unused in this instruction */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to GEMM stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Source/destination SRAM for store/load instruction */
  uint64_t memory_type    : VTA_MEMOP_ID_BIT_WIDTH;
  /*! \brief SRAM base address (pointer to memory elem type) */
  uint64_t sram_base      : VTA_MEMOP_SRAM_ADDR_BIT_WIDTH;
  /*! \brief DRAM base address (pointer to memory elem type) */
  uint64_t dram_base      : VTA_MEMOP_DRAM_ADDR_BIT_WIDTH;
  /*! \brief 2D access pattern: y-size */
  uint64_t y_size         : VTA_MEMOP_SIZE_BIT_WIDTH;
  /*! \brief 2D access pattern: x-size (in terms of memory elements) */
  uint64_t x_size         : VTA_MEMOP_SIZE_BIT_WIDTH;
  /*! \brief 2D access pattern: x-stride (in terms of memory elements) */
  uint64_t x_stride       : VTA_MEMOP_STRIDE_BIT_WIDTH;
  /*! \brief 2D access pattern: start padding along y dimension */
  uint64_t y_pad_0        : VTA_MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: end padding along y dimension */
  uint64_t y_pad_1        : VTA_MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: start padding along x dimension */
  uint64_t x_pad_0        : VTA_MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: end padding along x dimension */
  uint64_t x_pad_1        : VTA_MEMOP_PAD_BIT_WIDTH;
} VTAMemInsn;

/*! \brief VTA GEMM instruction
*   GEMM instruction is implemented by executing a sequence of micro-operations
*   that is read in the local micro-op memory, delimited by \a uop_bgn and
*   \a uop_end. For improved storage-efficiency, the micro-operations can be
*   executed in a 2-level nested loop as follows:
*   \code{.cpp}
*     for (i = 0; i < iter_out; i++) {
*       for (j = 0; j < iter_in; j++) {
*         for (k = uop_bgn; k < uop_end; k++) {
*           // Read micro op
*           uop_T uop = uop_mem[k];
*           // Read in memory indices
*           acc_idx_T acc_idx = uop.dst_idx;
*           inp_idx_T inp_idx = uop.inp_idx;
*           wgt_idx_T wgt_idx = uop.wgt_idx;
*           // Update those indices with the following affine functions
*           acc_idx += iter_in * dst_factor_in + iter_out * dst_factor_out;
*           inp_idx += iter_in * src_factor_in + iter_out * src_factor_out;
*           wgt_idx += iter_in * wgt_factor_in + iter_out * wgt_factor_out;
*           // Perform GEMM operation
*           acc_mem[acc_idx] += dot(inp_mem[inp_idx], wgt[wgt_idx]);
*         }
*       }
*     }
*   \endcode
*
*/
typedef struct {
  /*! \brief The instruction opcode */
  uint64_t opcode         : VTA_OPCODE_BIT_WIDTH;
  /*! \brief Pop dependence token from load stage */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from store stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Push dependence token to load stage */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to store stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Reset register */
  uint64_t reset_reg      : 1;
  /*! \brief Micro-op begin address */
  uint64_t uop_bgn        : VTA_LOG_UOP_BUFF_DEPTH;
  /*! \brief Micro-op end address */
  uint64_t uop_end        : VTA_LOG_UOP_BUFF_DEPTH + 1;
  /*! \brief Iterations in the outer uop execution loop */
  uint64_t iter_out       : VTA_LOOP_ITER_WIDTH;
  /*! \brief Iterations in the inner uop execution loop */
  uint64_t iter_in        : VTA_LOOP_ITER_WIDTH;
  /*! \brief Outer loop accumulator memory index factor */
  uint64_t dst_factor_out : VTA_LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory index factor */
  uint64_t dst_factor_in  : VTA_LOG_ACC_BUFF_DEPTH;
  /*! \brief Outer loop input memory index factor */
  uint64_t src_factor_out : VTA_LOG_INP_BUFF_DEPTH;
  /*! \brief Inner loop input memory index factor */
  uint64_t src_factor_in  : VTA_LOG_INP_BUFF_DEPTH;
  /*! \brief Outer loop weight memory index factor */
  uint64_t wgt_factor_out : VTA_LOG_WGT_BUFF_DEPTH;
  /*! \brief Inner loop weight memory index factor */
  uint64_t wgt_factor_in  : VTA_LOG_WGT_BUFF_DEPTH;
} VTAGemInsn;

/*! \brief VTA ALU instruction
*   ALU instruction is implemented by executing a sequence of micro-operations
*   that is read in the local micro-op memory, delimited by \a uop_bgn and
*   \a uop_end. For improved storage-efficiency, the micro-operations can be
*   executed in a 2-level nested loop as follows:
*   \code{.cpp}
*     for (i = 0; i < iter_out; i++) {
*       for (j = 0; j < iter_in; j++) {
*         for (k = uop_bgn; k < uop_end; k++) {
*           // Read micro op
*           uop_T uop = uop_mem[k];
*           // Read in memory indices
*           acc_idx_T dst_idx = uop.dst_idx;
*           inp_idx_T src_idx = uop.inp_idx;
*           // Update those indices with the following affine functions
*           dst_idx += iter_in * dst_factor_in + iter_out * dst_factor_out;
*           src_idx += iter_in * src_factor_in + iter_out * src_factor_out;
*           // Perform ALU operation
*           if (use_imm) {
*             acc_mem[dst_idx] = alu_op(alu_opcode, acc_mem[dst_idx], imm);
*           } else {
*             acc_mem[dst_idx] = alu_op(alu_opcode, acc_mem[dst_idx], acc_mem[src_idx]);
*           }
*         }
*       }
*     }
*   \endcode
*
*/
typedef struct {
  /*! \brief The instruction opcode */
  uint64_t opcode         : VTA_OPCODE_BIT_WIDTH;
  /*! \brief Pop dependence token from load stage */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from store stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Push dependence token to load stage */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to store stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Reset register */
  uint64_t reset_reg      : 1;
  /*! \brief Micro-op begin address */
  uint64_t uop_bgn        : VTA_LOG_UOP_BUFF_DEPTH;
  /*! \brief Micro-op end address */
  uint64_t uop_end        : VTA_LOG_UOP_BUFF_DEPTH + 1;
  /*! \brief Iterations in the outer uop execution loop */
  uint64_t iter_out       : VTA_LOOP_ITER_WIDTH;
  /*! \brief Iterations in the inner uop execution loop */
  uint64_t iter_in        : VTA_LOOP_ITER_WIDTH;
  /*! \brief Outer loop accumulator memory destination index factor */
  uint64_t dst_factor_out : VTA_LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory destination index factor */
  uint64_t dst_factor_in  : VTA_LOG_ACC_BUFF_DEPTH;
  /*! \brief Outer loop accumulator memory source index factor */
  uint64_t src_factor_out : VTA_LOG_INP_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory source index factor */
  uint64_t src_factor_in  : VTA_LOG_INP_BUFF_DEPTH;
  /*! \brief ALU opcode */
  uint64_t alu_opcode     : VTA_ALU_OPCODE_BIT_WIDTH;
  /*! \brief Use immediate is true */
  uint64_t use_imm        : 1;
  /*! \brief Immediate value: allow negative value */
  int64_t imm            : VTA_ALUOP_IMM_BIT_WIDTH;
} VTAAluInsn;

/*! \brief VTA ALU instruction converter */
union VTAInsn {
  /*! \brief VTA generic instruction */
  VTAGenericInsn generic;
  /*! \brief VTA load/store instruction */
  VTAMemInsn mem;
  /*! \brief VTA GEMM instruction */
  VTAGemInsn gemm;
  /*! \brief VTA ALU instruction */
  VTAAluInsn alu;
};

/*! \brief VTA micro-op for GEMM/ALU instruction */
typedef struct {
  /*! \brief Destination index (indexes accum buffer) */
  uint32_t dst_idx    : VTA_LOG_ACC_BUFF_DEPTH;
  /*! \brief Source index (indexes input buffer for GEMM or accum buffer for ALU) */
  uint32_t src_idx    : VTA_LOG_INP_BUFF_DEPTH;
  /*! \brief Weight index (indexes weight buffer) */
  uint32_t wgt_idx    : VTA_LOG_WGT_BUFF_DEPTH;
} VTAUop;

#ifdef __cplusplus
}
#endif
#endif  // VTA_HW_SPEC_H_
