/*!
 *  Copyright (c) 2018 by Contributors
 * \file hw_spec.h
 * \brief Preprocessor definitions for VTA HLS design and runtime.
 */

#ifndef VTA_HW_SPEC_H_
#define VTA_HW_SPEC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

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
/*! log2 of ALU data type width */
#define VTA_LOG_ALU_WIDTH (VTA_LOG_ACC_WIDTH - 1)
/*! ALU data type width */
#define VTA_ALU_WIDTH (1 << VTA_LOG_ALU_WIDTH)

/*! Batch size (corresponds to A in (A,B)x(B,C) mat mult)*/
#define VTA_BATCH (1 << VTA_LOG_BATCH)
/*! Blocking factor of inner most loop (corresponds to B in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_IN (1 << VTA_LOG_BLOCK_IN)
/*! Blocking factor of the outer loop (corresponds to C in (A,B)x(B,C) mat mult) */
#define VTA_BLOCK_OUT (1 << VTA_LOG_BLOCK_OUT)

/*! Weight vector width */
#define VTA_WGT_VECTOR_WIDTH (VTA_WGT_WIDTH * VTA_BLOCK_IN)
/*! Input vector width */
#define VTA_INP_VECTOR_WIDTH (VTA_INP_WIDTH * VTA_BLOCK_IN)
/*! Accumulator vector width */
#define VTA_ACC_VECTOR_WIDTH (VTA_ACC_WIDTH * VTA_BLOCK_OUT)
/*! Output vector width */
#define VTA_OUT_VECTOR_WIDTH (VTA_OUT_WIDTH * VTA_BLOCK_OUT)

/*! On-chip micro-op buffer size in B */
#define VTA_UOP_BUFF_SIZE (1 << VTA_LOG_UOP_BUFF_SIZE)
/*! On-chip weight buffer size in B */
#define VTA_WGT_BUFF_SIZE (1 << VTA_LOG_WGT_BUFF_SIZE)
/*! On-chip activation buffer size in B */
#define VTA_INP_BUFF_SIZE (1 << VTA_LOG_INP_BUFF_SIZE)
/*! On-chip accumulator buffer size in B */
#define VTA_ACC_BUFF_SIZE (1 << VTA_LOG_ACC_BUFF_SIZE)

/*! Size of instruction buffer element in B */
#define VTA_INS_ELEM_BYTES (VTA_INS_WIDTH / 8)
/*! Size of uop buffer element in B*/
#define VTA_UOP_ELEM_BYTES (VTA_UOP_WIDTH / 8)
/*! Size of activation buffer element in B*/
#define VTA_INP_ELEM_BYTES (VTA_BATCH * VTA_BLOCK_IN * VTA_INP_WIDTH / 8)
/*! Size of weight buffer element in B*/
#define VTA_WGT_ELEM_BYTES (VTA_BLOCK_OUT * VTA_BLOCK_IN * VTA_WGT_WIDTH / 8)
/*! Size of accumulator buffer element in B*/
#define VTA_ACC_ELEM_BYTES (VTA_BATCH * VTA_BLOCK_OUT * VTA_ACC_WIDTH / 8)

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
/*! ALU Instruction: immediate bitwidth*/
#define VTA_ALUOP_IMM_BIT_WIDTH 16
/*! GEMM/ALU Instruction: loop max iter bits */
#define VTA_LOOP_ITER_WIDTH 14

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

// Instruction organization layout:
//
// LOAD/STORE
// _____________________________|_type______________|
// arg 0: opcode                | opcode_T          |
// arg 1: pop_prev_dependence   | bool              |
// arg 2: pop_next_dependence   | bool              |
// arg 3: push_prev_dependence  | bool              |
// arg 4: push_next_dependence  | bool              |
// arg 5: memory_type           | memop_id_T        |
// arg 6: pad_value             | memop_pad_val_T   |
// arg 7: sram_base             | memop_sram_T      |
// arg 8: dram_base             | memop_dram_T      |
// arg 9: y_size                | memop_size_T      |
// arg a: x_size                | memop_size_T      |
// arg b: x_stride              | memop_stride_T    |
// arg c: y_pad_0               | memop_pad_T       |
// arg d: y_pad_1               | memop_pad_T       |
// arg e: x_pad_0               | memop_pad_T       |
// arg f: x_pad_1               | memop_pad_T       |
//
// GEMM
// _____________________________|_type______________|
// arg 0: opcode                | opcode_T          |
// arg 1: pop_prev_dependence   | bool              |
// arg 2: pop_next_dependence   | bool              |
// arg 3: push_prev_dependence  | bool              |
// arg 4: push_next_dependence  | bool              |
// arg 5: reset_reg             | bool              |
// arg 6: uop_bgn               | uop_idx_T         |
// arg 7: uop_end               | uop_idx_T         |
// arg 8: iteration count ax0   | loop_T            |
// arg 9: iteration count ax1   | loop_T            |
// arg a: accum idx factor ax0  | acc_idx_T         |
// arg b: accum idx factor ax1  | acc_idx_T         |
// arg c: input idx factor ax0  | inp_idx_T         |
// arg d: input idx factor ax1  | inp_idx_T         |
// arg e: weight idx factor ax0 | wgt_idx_T         |
// arg f: weight idx factor ax1 | wgt_idx_T         |
//
// ALU
// _____________________________|_type______________|
// arg 0: opcode                | opcode_T          |
// arg 1: pop_prev_dependence   | bool              |
// arg 2: pop_next_dependence   | bool              |
// arg 3: push_prev_dependence  | bool              |
// arg 4: push_next_dependence  | bool              |
// arg 5: reset_reg             | bool              |
// arg 6: uop_bgn               | uop_idx_T         |
// arg 7: uop_end               | uop_idx_T         |
// arg 8: iteration count ax0   | loop_T            |
// arg 9: iteration count ax1   | loop_T            |
// arg a: dst idx factor ax0    | acc_idx_T         |
// arg b: dst idx factor ax1    | acc_idx_T         |
// arg c: src idx factor ax0    | inp_idx_T         |
// arg d: src idx factor ax1    | inp_idx_T         |
// arg e: alu_opcode            | aluop_opcode_T    |
// arg f: use_imm               | bool              |
// arg g: imm                   | alu_imm_T         |

/*! Load/Store instruction start position of the opcode field */
#define VTA_INSN_MEM_0_0 0
/*! Load/Store instruction end position of the opcode field */
#define VTA_INSN_MEM_0_1 (VTA_INSN_MEM_0_0 + VTA_OPCODE_BIT_WIDTH - 1)
/*! Load/Store instruction position of the pop_prev_dep field */
#define VTA_INSN_MEM_1   (VTA_INSN_MEM_0_1 + 1)
/*! Load/Store instruction position of the pop_next_dep field */
#define VTA_INSN_MEM_2   (VTA_INSN_MEM_1 + 1)
/*! Load/Store instruction position of the push_prev_dependence field */
#define VTA_INSN_MEM_3   (VTA_INSN_MEM_2 + 1)
/*! Load/Store instruction position of the push_next_dependence field */
#define VTA_INSN_MEM_4   (VTA_INSN_MEM_3 + 1)
/*! Load/Store instruction start position of the memory_type field */
#define VTA_INSN_MEM_5_0 (VTA_INSN_MEM_4 + 1)
/*! Load/Store instruction end position of the memory_type field */
#define VTA_INSN_MEM_5_1 (VTA_INSN_MEM_5_0 + VTA_MEMOP_ID_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the sram_base field */
#define VTA_INSN_MEM_6_0 (VTA_INSN_MEM_5_1 + 1)
/*! Load/Store instruction end position of the sram_base field */
#define VTA_INSN_MEM_6_1 (VTA_INSN_MEM_6_0 + VTA_MEMOP_SRAM_ADDR_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the dram_base field */
#define VTA_INSN_MEM_7_0 (VTA_INSN_MEM_6_1 + 1)
/*! Load/Store instruction end position of the dram_base field */
#define VTA_INSN_MEM_7_1 (VTA_INSN_MEM_7_0 + VTA_MEMOP_DRAM_ADDR_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the y_size field */
#define VTA_INSN_MEM_8_0 64
/*! Load/Store instruction end position of the y_size field */
#define VTA_INSN_MEM_8_1 (VTA_INSN_MEM_8_0 + VTA_MEMOP_SIZE_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the x_size field */
#define VTA_INSN_MEM_9_0 (VTA_INSN_MEM_8_1 + 1)
/*! Load/Store instruction start position of the x_size field */
#define VTA_INSN_MEM_9_1 (VTA_INSN_MEM_9_0 + VTA_MEMOP_SIZE_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the x_stride field */
#define VTA_INSN_MEM_A_0 (VTA_INSN_MEM_9_1 + 1)
/*! Load/Store instruction end position of the x_stride field */
#define VTA_INSN_MEM_A_1 (VTA_INSN_MEM_A_0 + VTA_MEMOP_STRIDE_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the y_pad_0 field */
#define VTA_INSN_MEM_B_0 (VTA_INSN_MEM_A_1 + 1)
/*! Load/Store instruction start position of the y_pad_0 field */
#define VTA_INSN_MEM_B_1 (VTA_INSN_MEM_B_0 + VTA_MEMOP_PAD_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the y_pad_1 field */
#define VTA_INSN_MEM_C_0 (VTA_INSN_MEM_B_1 + 1)
/*! Load/Store instruction start position of the y_pad_1 field */
#define VTA_INSN_MEM_C_1 (VTA_INSN_MEM_C_0 + VTA_MEMOP_PAD_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the x_pad_0 field */
#define VTA_INSN_MEM_D_0 (VTA_INSN_MEM_C_1 + 1)
/*! Load/Store instruction start position of the x_pad_0 field */
#define VTA_INSN_MEM_D_1 (VTA_INSN_MEM_D_0 + VTA_MEMOP_PAD_BIT_WIDTH - 1)
/*! Load/Store instruction start position of the x_pad_1 field */
#define VTA_INSN_MEM_E_0 (VTA_INSN_MEM_D_1 + 1)
/*! Load/Store instruction start position of the x_pad_1 field */
#define VTA_INSN_MEM_E_1 (VTA_INSN_MEM_E_0 + VTA_MEMOP_PAD_BIT_WIDTH - 1)

/*! GEMM instruction start position of the opcode field */
#define VTA_INSN_GEM_0_0 0
/*! GEMM instruction end position of the opcode field */
#define VTA_INSN_GEM_0_1 (VTA_INSN_GEM_0_0 + VTA_OPCODE_BIT_WIDTH - 1)
/*! GEMM instruction position of the pop_prev_dep field */
#define VTA_INSN_GEM_1   (VTA_INSN_GEM_0_1 + 1)
/*! GEMM instruction position of the pop_next_dep field */
#define VTA_INSN_GEM_2   (VTA_INSN_GEM_1 + 1)
/*! GEMM instruction position of the push_prev_dependence field */
#define VTA_INSN_GEM_3   (VTA_INSN_GEM_2 + 1)
/*! GEMM instruction position of the push_next_dependence field */
#define VTA_INSN_GEM_4   (VTA_INSN_GEM_3 + 1)
/*! GEMM instruction position of the reset register bit */
#define VTA_INSN_GEM_5   (VTA_INSN_GEM_4 + 1)
/*! GEMM instruction start position of the uop_bgn field */
#define VTA_INSN_GEM_6_0 (VTA_INSN_GEM_5 + 1)
/*! GEMM instruction end position of the uop_bgn field */
#define VTA_INSN_GEM_6_1 (VTA_INSN_GEM_6_0 + VTA_LOG_UOP_BUFF_DEPTH - 1)
/*! GEMM instruction start position of the uop_end field */
#define VTA_INSN_GEM_7_0 (VTA_INSN_GEM_6_1 + 1)
/*! GEMM instruction end position of the uop_end field */
#define VTA_INSN_GEM_7_1 (VTA_INSN_GEM_7_0 + VTA_LOG_UOP_BUFF_DEPTH + 1 - 1)
/*! GEMM instruction start position of the iter_out field */
#define VTA_INSN_GEM_8_0 (VTA_INSN_GEM_7_1 + 1)
/*! GEMM instruction end position of the iter_out field */
#define VTA_INSN_GEM_8_1 (VTA_INSN_GEM_8_0 + VTA_LOOP_ITER_WIDTH - 1)
/*! GEMM instruction start position of the iter_in field */
#define VTA_INSN_GEM_9_0 (VTA_INSN_GEM_8_1 + 1)
/*! GEMM instruction end position of the iter_in field */
#define VTA_INSN_GEM_9_1 (VTA_INSN_GEM_9_0 + VTA_LOOP_ITER_WIDTH - 1)
/*! GEMM instruction start position of the dst_factor_out field */
#define VTA_INSN_GEM_A_0 64
/*! GEMM instruction end position of the dst_factor_out field */
#define VTA_INSN_GEM_A_1 (VTA_INSN_GEM_A_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
/*! GEMM instruction start position of the dst_factor_in field */
#define VTA_INSN_GEM_B_0 (VTA_INSN_GEM_A_1 + 1)
/*! GEMM instruction end position of the dst_factor_in field */
#define VTA_INSN_GEM_B_1 (VTA_INSN_GEM_B_0 + VTA_LOG_ACC_BUFF_DEPTH - 1)
/*! GEMM instruction start position of the src_factor_out field */
#define VTA_INSN_GEM_C_0 (VTA_INSN_GEM_B_1 + 1)
/*! GEMM instruction end position of the src_factor_out field */
#define VTA_INSN_GEM_C_1 (VTA_INSN_GEM_C_0 + VTA_LOG_INP_BUFF_DEPTH - 1)
/*! GEMM instruction start position of the src_factor_in field */
#define VTA_INSN_GEM_D_0 (VTA_INSN_GEM_C_1 + 1)
/*! GEMM instruction end position of the src_factor_in field */
#define VTA_INSN_GEM_D_1 (VTA_INSN_GEM_D_0 + VTA_LOG_INP_BUFF_DEPTH - 1)

/*! GEMM instruction start position of the wgt_factor_out field */
#define VTA_INSN_GEM_E_0 (VTA_INSN_GEM_D_1 + 1)
/*! GEMM instruction end position of the wgt_factor_out field */
#define VTA_INSN_GEM_E_1 (VTA_INSN_GEM_E_0 + VTA_LOG_WGT_BUFF_DEPTH - 1)
/*! GEMM instruction start position of the wgt_factor_in field */
#define VTA_INSN_GEM_F_0 (VTA_INSN_GEM_E_1 + 1)
/*! GEMM instruction end position of the wgt_factor_in field */
#define VTA_INSN_GEM_F_1 (VTA_INSN_GEM_F_0 + VTA_LOG_WGT_BUFF_DEPTH - 1)

/*! ALU instruction start position of the alu_opcode field */
#define VTA_INSN_ALU_E_0 (VTA_INSN_GEM_D_1 + 1)
/*! ALU instruction end position of the alu_opcode field */
#define VTA_INSN_ALU_E_1 (VTA_INSN_ALU_E_0 + VTA_ALU_OPCODE_BIT_WIDTH - 1)
/*! ALU instruction position of the use_imm field */
#define VTA_INSN_ALU_F   (VTA_INSN_ALU_E_1 + 1)
/*! ALU instruction start position of the immediate field */
#define VTA_INSN_ALU_G_0 (VTA_INSN_ALU_F + 1)
/*! ALU instruction end position of the immediate field */
#define VTA_INSN_ALU_G_1 (VTA_INSN_ALU_G_0 + VTA_ALUOP_IMM_BIT_WIDTH - 1)

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
  uint64_t word_0         : 64;
  uint64_t word_1         : 64;
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
