/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_defines.h
 * \brief Preprocessor definitions for VTA HLS design and runtime.
 */
#ifndef VTA_DEFINES_H_
#define VTA_DEFINES_H_

#include <stdint.h>

/*! log2 of instruction data type width */
#define LOG_INS_WIDTH 7
/*! Instruction data type width */
#define INS_WIDTH (1<<LOG_INS_WIDTH)
/*! log2 of micro op data type width */
#define LOG_UOP_WIDTH 5
/*! Micro Op data type width */
#define UOP_WIDTH (1<<LOG_UOP_WIDTH)
/*! Weight data type width */
#define WGT_WIDTH (1<<LOG_WGT_WIDTH)
/*! Input data type width */
#define INP_WIDTH (1<<LOG_INP_WIDTH)
/*! Output data type width */
#define OUT_WIDTH (1<<LOG_OUT_WIDTH)
/*! Accumulator data type width */
#define ACC_WIDTH (1<<LOG_ACC_WIDTH)
/*! log2 of ALU data type width */
#define LOG_ALU_WIDTH (LOG_ACC_WIDTH-1)
/*! ALU data type width */
#define ALU_WIDTH (1<<LOG_ALU_WIDTH)

/*! Batch size (corresponds to A in (A,B)x(B,C) mat mult)*/
#define BATCH (1<<LOG_BATCH)
/*! Blocking factor of inner most loop (corresponds to B in (A,B)x(B,C) mat mult) */
#define BLOCK_IN (1<<LOG_BLOCK_IN)
/*! Blocking factor of the outer loop (corresponds to C in (A,B)x(B,C) mat mult) */
#define BLOCK_OUT (1<<LOG_BLOCK_OUT)

/*! Weight vector width */
#define WGT_VECTOR_WIDTH (WGT_WIDTH*BLOCK_IN)
/*! Input vector width */
#define INP_VECTOR_WIDTH (INP_WIDTH*BLOCK_IN)
/*! Accumulator vector width */
#define ACC_VECTOR_WIDTH (ACC_WIDTH*BLOCK_OUT)
/*! Output vector width */
#define OUT_VECTOR_WIDTH (OUT_WIDTH*BLOCK_OUT)

/*! On-chip micro-op buffer size in B */
#define UOP_BUFF_SIZE (1<<LOG_UOP_BUFF_SIZE)
/*! On-chip weight buffer size in B */
#define WGT_BUFF_SIZE (1<<LOG_WGT_BUFF_SIZE)
/*! On-chip activation buffer size in B */
#define INP_BUFF_SIZE (1<<LOG_INP_BUFF_SIZE)
/*! On-chip accumulator buffer size in B */
#define ACC_BUFF_SIZE (1<<LOG_ACC_BUFF_SIZE)

/*! Size of instruction buffer element in B */
#define INS_ELEM_BYTES (INS_WIDTH/8)
/*! Size of uop buffer element in B*/
#define UOP_ELEM_BYTES (UOP_WIDTH/8)
/*! Size of activation buffer element in B*/
#define INP_ELEM_BYTES (BATCH*BLOCK_IN*INP_WIDTH/8)
/*! Size of weight buffer element in B*/
#define WGT_ELEM_BYTES (BLOCK_OUT*BLOCK_IN*WGT_WIDTH/8)
/*! Size of accumulator buffer element in B*/
#define ACC_ELEM_BYTES (BATCH*BLOCK_OUT*ACC_WIDTH/8)

/*! On-chip micro-op buffer depth */
#define UOP_BUFF_DEPTH (UOP_BUFF_SIZE/UOP_ELEM_BYTES)
/*! log2 of on-chip micro-op buffer depth */
#define LOG_UOP_BUFF_DEPTH (LOG_UOP_BUFF_SIZE-LOG_UOP_WIDTH+3)
// ! \brief On-chip weight buffer depth
#define WGT_BUFF_DEPTH (WGT_BUFF_SIZE/WGT_ELEM_BYTES)
/*! log2 of weight micro-op buffer depth */
#define LOG_WGT_BUFF_DEPTH (LOG_WGT_BUFF_SIZE-LOG_BLOCK_OUT-LOG_BLOCK_IN-LOG_WGT_WIDTH+3)
/*! On-chip activation buffer depth */
#define INP_BUFF_DEPTH (INP_BUFF_SIZE/INP_ELEM_BYTES)
/*! log2 of activation micro-op buffer depth */
#define LOG_INP_BUFF_DEPTH (LOG_INP_BUFF_SIZE-LOG_BATCH-LOG_BLOCK_IN-LOG_INP_WIDTH+3)
/*! On-chip accumulator buffer depth */
#define ACC_BUFF_DEPTH (ACC_BUFF_SIZE/ACC_ELEM_BYTES)
/*! log2 of on-chip accumulator buffer depth */
#define LOG_ACC_BUFF_DEPTH (LOG_ACC_BUFF_SIZE-LOG_BATCH-LOG_BLOCK_OUT-LOG_ACC_WIDTH+3)

/*! Instruction opcode field bitwidth */
#define OPCODE_BIT_WIDTH 3
/*! ALU opcode field bitwidth */
#define ALU_OPCODE_BIT_WIDTH 3
/*! ALU instruction reset mode bitwidth */
#define ALU_RESET_BIT_WIDTH 2

/*! Opcode: load encoding */
#define OPCODE_LOAD 0
/*! Opcode: store encoding */
#define OPCODE_STORE 1
/*! Opcode: GEMM encoding */
#define OPCODE_GEMM 2
/*! Opcode: finish encoding */
#define OPCODE_FINISH 3
/*! Opcode: ALU encoding */
#define OPCODE_ALU 4

/*! ALU opcode: unary min op */
#define ALU_OPCODE_MIN 0
/*! ALU opcode: unary max op */
#define ALU_OPCODE_MAX 1
/*! ALU opcode: binary add op */
#define ALU_OPCODE_ADD 2
/*! ALU opcode: binary sub op [NOT IMPLEMENTED] */
#define ALU_OPCODE_SUB 3
/*! ALU opcode: binary mul op  [NOT IMPLEMENTED] */
#define ALU_OPCODE_MUL 4
/*! ALU opcode: shift left by immediate op */
#define ALU_OPCODE_SHL 5
/*! ALU opcode: shift right by immediate op [NOT IMPLEMENTED] */
#define ALU_OPCODE_SHR 6

/*! ALU instruction reset mode: set to min */
#define ALU_RESET_MIN 3
/*! ALU instruction reset mode: set to zero */
#define ALU_RESET_ZERO 0
/*! ALU instruction reset mode: no reset */
#define ALU_NO_RESET 2
/*! ALU instruction reset mode: set to max */
#define ALU_RESET_MAX 1

/*! Memory type field bitwidth */
#define MEMOP_ID_BIT_WIDTH 2
/*! Load/Store Instruction: DRAM address width*/
#define MEMOP_SRAM_ADDR_BIT_WIDTH 16
/*! Load/Store Instruction: DRAM address width*/
#define MEMOP_DRAM_ADDR_BIT_WIDTH 32
/*! Load/Store Instruction: transfer size width*/
#define MEMOP_SIZE_BIT_WIDTH 16
/*! Load/Store Instruction: stride size width*/
#define MEMOP_STRIDE_BIT_WIDTH 16
/*! Load/Store Instruction: padding width*/
#define MEMOP_PAD_BIT_WIDTH 4
/*! Load/Store Instruction: padding value encoding width*/
#define MEMOP_PAD_VAL_BIT_WIDTH 2
/*! ALU Instruction: immediate bitwidth*/
#define ALUOP_IMM_BIT_WIDTH 16
/*! GEMM/ALU Instruction: loop max iter bits */
#define LOOP_ITER_WIDTH 15

/*! Mem ID constant: uop memory */
#define MEM_ID_UOP 0
/*! Mem ID constant: weight memory */
#define MEM_ID_WGT 1
/*! Mem ID constant: input memory */
#define MEM_ID_INP 2
/*! Mem ID constant: accumulator/bias memory */
#define MEM_ID_ACC 3
/*! Mem ID constant: output store buffer */
#define MEM_ID_OUT 4

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
// arg 5: uop_bgn               | uop_idx_T         |
// arg 6: uop_end               | uop_idx_T         |
// arg 7: iteration count ax0   | loop_T            |
// arg 8: iteration count ax1   | loop_T            |
// arg 9: accum idx factor ax0  | acc_idx_T         |
// arg a: accum idx factor ax1  | acc_idx_T         |
// arg b: input idx factor ax0  | acc_idx_T         |
// arg c: input idx factor ax1  | acc_idx_T         |
// arg d: weight idx factor ax0 | wgt_idx_T         |
// arg e: weight idx factor ax1 | wgt_idx_T         |
//
// ALU
// _____________________________|_type______________|
// arg 0: opcode                | opcode_T          |
// arg 1: pop_prev_dependence   | bool              |
// arg 2: pop_next_dependence   | bool              |
// arg 3: push_prev_dependence  | bool              |
// arg 4: push_next_dependence  | bool              |
// arg 5: uop_bgn               | uop_idx_T         |
// arg 6: uop_end               | uop_idx_T         |
// arg 7: iteration count ax0   | loop_T            |
// arg 8: iteration count ax1   | loop_T            |
// arg 9: dst idx factor ax0    | acc_idx_T         |
// arg a: dst idx factor ax1    | acc_idx_T         |
// arg b: src idx factor ax0    | acc_idx_T         |
// arg c: src idx factor ax1    | acc_idx_T         |
// arg d: alu_opcode            | aluop_opcode_T    |
// arg e: use_imm               | bool              |
// arg f: imm                   | alu_imm_T         |

/*! Load/Store instruction start position of the opcode field */
#define INSN_MEM_0_0 0
/*! Load/Store instruction end position of the opcode field */
#define INSN_MEM_0_1 (INSN_MEM_0_0+OPCODE_BIT_WIDTH-1)
/*! Load/Store instruction position of the pop_prev_dep field */
#define INSN_MEM_1   (INSN_MEM_0_1+1)
/*! Load/Store instruction position of the pop_next_dep field */
#define INSN_MEM_2   (INSN_MEM_1+1)
/*! Load/Store instruction position of the push_prev_dependence field */
#define INSN_MEM_3   (INSN_MEM_2+1)
/*! Load/Store instruction position of the push_next_dependence field */
#define INSN_MEM_4   (INSN_MEM_3+1)
/*! Load/Store instruction start position of the memory_type field */
#define INSN_MEM_5_0 (INSN_MEM_4+1)
/*! Load/Store instruction end position of the memory_type field */
#define INSN_MEM_5_1 (INSN_MEM_5_0+MEMOP_ID_BIT_WIDTH-1)
/*! Load/Store instruction start position of the sram_base field */
#define INSN_MEM_6_0 (INSN_MEM_5_1+1)
/*! Load/Store instruction end position of the sram_base field */
#define INSN_MEM_6_1 (INSN_MEM_6_0+MEMOP_SRAM_ADDR_BIT_WIDTH-1)
/*! Load/Store instruction start position of the dram_base field */
#define INSN_MEM_7_0 (INSN_MEM_6_1+1)
/*! Load/Store instruction end position of the dram_base field */
#define INSN_MEM_7_1 (INSN_MEM_7_0+MEMOP_DRAM_ADDR_BIT_WIDTH-1)
/*! Load/Store instruction start position of the y_size field */
#define INSN_MEM_8_0 64
/*! Load/Store instruction end position of the y_size field */
#define INSN_MEM_8_1 (INSN_MEM_8_0+MEMOP_SIZE_BIT_WIDTH-1)
/*! Load/Store instruction start position of the x_size field */
#define INSN_MEM_9_0 (INSN_MEM_8_1+1)
/*! Load/Store instruction start position of the x_size field */
#define INSN_MEM_9_1 (INSN_MEM_9_0+MEMOP_SIZE_BIT_WIDTH-1)
/*! Load/Store instruction start position of the x_stride field */
#define INSN_MEM_A_0 (INSN_MEM_9_1+1)
/*! Load/Store instruction end position of the x_stride field */
#define INSN_MEM_A_1 (INSN_MEM_A_0+MEMOP_STRIDE_BIT_WIDTH-1)
/*! Load/Store instruction start position of the y_pad_0 field */
#define INSN_MEM_B_0 (INSN_MEM_A_1+1)
/*! Load/Store instruction start position of the y_pad_0 field */
#define INSN_MEM_B_1 (INSN_MEM_B_0+MEMOP_PAD_BIT_WIDTH-1)
/*! Load/Store instruction start position of the y_pad_1 field */
#define INSN_MEM_C_0 (INSN_MEM_B_1+1)
/*! Load/Store instruction start position of the y_pad_1 field */
#define INSN_MEM_C_1 (INSN_MEM_C_0+MEMOP_PAD_BIT_WIDTH-1)
/*! Load/Store instruction start position of the x_pad_0 field */
#define INSN_MEM_D_0 (INSN_MEM_C_1+1)
/*! Load/Store instruction start position of the x_pad_0 field */
#define INSN_MEM_D_1 (INSN_MEM_D_0+MEMOP_PAD_BIT_WIDTH-1)
/*! Load/Store instruction start position of the x_pad_1 field */
#define INSN_MEM_E_0 (INSN_MEM_D_1+1)
/*! Load/Store instruction start position of the x_pad_1 field */
#define INSN_MEM_E_1 (INSN_MEM_E_0+MEMOP_PAD_BIT_WIDTH-1)

/*! GEMM instruction start position of the opcode field */
#define INSN_GEM_0_0 0
/*! GEMM instruction end position of the opcode field */
#define INSN_GEM_0_1 (INSN_GEM_0_0+OPCODE_BIT_WIDTH-1)
/*! GEMM instruction position of the pop_prev_dep field */
#define INSN_GEM_1   (INSN_GEM_0_1+1)
/*! GEMM instruction position of the pop_next_dep field */
#define INSN_GEM_2   (INSN_GEM_1+1)
/*! GEMM instruction position of the push_prev_dependence field */
#define INSN_GEM_3   (INSN_GEM_2+1)
/*! GEMM instruction position of the push_next_dependence field */
#define INSN_GEM_4   (INSN_GEM_3+1)
/*! GEMM instruction start position of the uop_bgn field */
#define INSN_GEM_5_0 (INSN_GEM_4+1)
/*! GEMM instruction end position of the uop_bgn field */
#define INSN_GEM_5_1 (INSN_GEM_5_0+LOG_UOP_BUFF_DEPTH-1)
/*! GEMM instruction start position of the uop_end field */
#define INSN_GEM_6_0 (INSN_GEM_5_1+1)
/*! GEMM instruction end position of the uop_end field */
#define INSN_GEM_6_1 (INSN_GEM_6_0+LOG_UOP_BUFF_DEPTH+1-1)
/*! GEMM instruction start position of the iter_out field */
#define INSN_GEM_7_0 (INSN_GEM_6_1+1)
/*! GEMM instruction end position of the iter_out field */
#define INSN_GEM_7_1 (INSN_GEM_7_0+LOOP_ITER_WIDTH-1)
/*! GEMM instruction start position of the iter_in field */
#define INSN_GEM_8_0 (INSN_GEM_7_1+1)
/*! GEMM instruction end position of the iter_in field */
#define INSN_GEM_8_1 (INSN_GEM_8_0+LOOP_ITER_WIDTH-1)
/*! GEMM instruction start position of the dst_factor_out field */
#define INSN_GEM_9_0 64
/*! GEMM instruction end position of the dst_factor_out field */
#define INSN_GEM_9_1 (INSN_GEM_9_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM instruction start position of the dst_factor_in field */
#define INSN_GEM_A_0 (INSN_GEM_9_1+1)
/*! GEMM instruction end position of the dst_factor_in field */
#define INSN_GEM_A_1 (INSN_GEM_A_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM instruction start position of the src_factor_out field */
#define INSN_GEM_B_0 (INSN_GEM_A_1+1)
/*! GEMM instruction end position of the src_factor_out field */
#define INSN_GEM_B_1 (INSN_GEM_B_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM instruction start position of the src_factor_in field */
#define INSN_GEM_C_0 (INSN_GEM_B_1+1)
/*! GEMM instruction end position of the src_factor_in field */
#define INSN_GEM_C_1 (INSN_GEM_C_0+LOG_ACC_BUFF_DEPTH-1)

/*! GEMM instruction start position of the wgt_factor_out field */
#define INSN_GEM_D_0 (INSN_GEM_C_1+1)
/*! GEMM instruction end position of the wgt_factor_out field */
#define INSN_GEM_D_1 (INSN_GEM_D_0+LOG_WGT_BUFF_DEPTH-1)
/*! GEMM instruction start position of the wgt_factor_in field */
#define INSN_GEM_E_0 (INSN_GEM_D_1+1)
/*! GEMM instruction end position of the wgt_factor_in field */
#define INSN_GEM_E_1 (INSN_GEM_E_0+LOG_WGT_BUFF_DEPTH-1)

/*! ALU instruction start position of the alu_opcode field */
#define INSN_ALU_D_0 (INSN_GEM_C_1+1)
/*! ALU instruction end position of the alu_opcode field */
#define INSN_ALU_D_1 (INSN_ALU_D_0+ALU_OPCODE_BIT_WIDTH-1)
/*! ALU instruction position of the use_imm field */
#define INSN_ALU_E   (INSN_ALU_D_1+1)
/*! ALU instruction start position of the immediate field */
#define INSN_ALU_F_0 (INSN_ALU_E+1)
/*! ALU instruction end position of the immediate field */
#define INSN_ALU_F_1 (INSN_ALU_F_0+ALUOP_IMM_BIT_WIDTH-1)

/*! GEMM Micro-op position of the reset_out field */
#define UOP_GEM_0 0
/*! GEMM Micro-op start position of the acc_idx field */
#define UOP_GEM_1_0 (UOP_GEM_0+1)
/*! GEMM Micro-op end position of the acc_idx field */
#define UOP_GEM_1_1 (UOP_GEM_1_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM Micro-op start position of the inp_idx field */
#define UOP_GEM_2_0 (UOP_GEM_1_1+1)
/*! GEMM Micro-op end position of the inp_idx field */
#define UOP_GEM_2_1 (UOP_GEM_2_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM Micro-op start position of the wgt_idx field */
#define UOP_GEM_3_0 (UOP_GEM_2_1+1)
/*! GEMM Micro-op end position of the wgt_idx field */
#define UOP_GEM_3_1 (UOP_GEM_3_0+LOG_WGT_BUFF_DEPTH-1)

/*! GEMM Micro-op position of the reset_out field */
#define UOP_ALU_0 0
/*! GEMM Micro-op start position of the acc_idx field */
#define UOP_ALU_1_0 (UOP_ALU_0+1)
/*! GEMM Micro-op end position of the acc_idx field */
#define UOP_ALU_1_1 (UOP_ALU_1_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM Micro-op start position of the inp_idx field */
#define UOP_ALU_2_0 (UOP_ALU_1_1+1)
/*! GEMM Micro-op end position of the inp_idx field */
#define UOP_ALU_2_1 (UOP_ALU_2_0+LOG_ACC_BUFF_DEPTH-1)
/*! GEMM Micro-op start position of the wgt_idx field */
#define UOP_ALU_3_0 (UOP_ALU_2_1+1)
/*! GEMM Micro-op end position of the wgt_idx field */
#define UOP_ALU_3_1 (UOP_ALU_3_0+LOG_WGT_BUFF_DEPTH-1)

/*! \brief VTA generic instruction */
typedef struct {
  uint64_t word_0     : 64;
  uint64_t word_1     : 64;
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
  uint64_t opcode         : OPCODE_BIT_WIDTH;
  /*! \brief Unused in this instruction */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from GEMM stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Unused in this instruction */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to GEMM stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Source/destination SRAM for store/load instruction */
  uint64_t memory_type    : MEMOP_ID_BIT_WIDTH;
  /*! \brief SRAM base address (pointer to memory elem type) */
  uint64_t sram_base      : MEMOP_SRAM_ADDR_BIT_WIDTH;
  /*! \brief DRAM base address (pointer to memory elem type) */
  uint64_t dram_base      : MEMOP_DRAM_ADDR_BIT_WIDTH;
  /*! \brief 2D access pattern: y-size */
  uint64_t y_size         : MEMOP_SIZE_BIT_WIDTH;
  /*! \brief 2D access pattern: x-size (in terms of memory elements) */
  uint64_t x_size         : MEMOP_SIZE_BIT_WIDTH;
  /*! \brief 2D access pattern: x-stride (in terms of memory elements) */
  uint64_t x_stride       : MEMOP_STRIDE_BIT_WIDTH;
  /*! \brief 2D access pattern: start padding along y dimension */
  uint64_t y_pad_0        : MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: end padding along y dimension */
  uint64_t y_pad_1        : MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: start padding along x dimension */
  uint64_t x_pad_0        : MEMOP_PAD_BIT_WIDTH;
  /*! \brief 2D access pattern: end padding along x dimension */
  uint64_t x_pad_1        : MEMOP_PAD_BIT_WIDTH;
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
  uint64_t opcode         : OPCODE_BIT_WIDTH;
  /*! \brief Pop dependence token from load stage */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from store stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Push dependence token to load stage */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to store stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Micro-op begin address */
  uint64_t uop_bgn        : LOG_UOP_BUFF_DEPTH;
  /*! \brief Micro-op end address */
  uint64_t uop_end        : LOG_UOP_BUFF_DEPTH+1;
  /*! \brief Iterations in the outer uop execution loop */
  uint64_t iter_out       : LOOP_ITER_WIDTH;
  /*! \brief Iterations in the inner uop execution loop */
  uint64_t iter_in        : LOOP_ITER_WIDTH;
  /*! \brief Outer loop accumulator memory index factor */
  uint64_t dst_factor_out : LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory index factor */
  uint64_t dst_factor_in  : LOG_ACC_BUFF_DEPTH;
  /*! \brief Outer loop input memory index factor */
  uint64_t src_factor_out : LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop input memory index factor */
  uint64_t src_factor_in  : LOG_ACC_BUFF_DEPTH;
  /*! \brief Outer loop weight memory index factor */
  uint64_t wgt_factor_out : LOG_WGT_BUFF_DEPTH;
  /*! \brief Inner loop weight memory index factor */
  uint64_t wgt_factor_in  : LOG_WGT_BUFF_DEPTH;
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
  uint64_t opcode         : OPCODE_BIT_WIDTH;
  /*! \brief Pop dependence token from load stage */
  uint64_t pop_prev_dep   : 1;
  /*! \brief Pop dependence token from store stage */
  uint64_t pop_next_dep   : 1;
  /*! \brief Push dependence token to load stage */
  uint64_t push_prev_dep  : 1;
  /*! \brief Push dependence token to store stage */
  uint64_t push_next_dep  : 1;
  /*! \brief Micro-op begin address */
  uint64_t uop_bgn        : LOG_UOP_BUFF_DEPTH;
  /*! \brief Micro-op end address */
  uint64_t uop_end        : LOG_UOP_BUFF_DEPTH+1;
  /*! \brief Iterations in the outer uop execution loop */
  uint64_t iter_out       : LOOP_ITER_WIDTH;
  /*! \brief Iterations in the inner uop execution loop */
  uint64_t iter_in        : LOOP_ITER_WIDTH;
  /*! \brief Outer loop accumulator memory destination index factor */
  uint64_t dst_factor_out : LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory destination index factor */
  uint64_t dst_factor_in  : LOG_ACC_BUFF_DEPTH;
  /*! \brief Outer loop accumulator memory source index factor */
  uint64_t src_factor_out : LOG_ACC_BUFF_DEPTH;
  /*! \brief Inner loop accumulator memory source index factor */
  uint64_t src_factor_in  : LOG_ACC_BUFF_DEPTH;
  /*! \brief ALU opcode */
  uint64_t alu_opcode     : ALU_OPCODE_BIT_WIDTH;
  /*! \brief Use immediate is true */
  uint64_t use_imm        : 1;
  /*! \brief Immediate value */
  uint64_t imm            : ALUOP_IMM_BIT_WIDTH;
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
  /*! \brief Initialize acc_mem at index dst_idx to 0*/
  uint32_t reset_out  : 1;
  /*! \brief Destination index (indexes accum buffer) */
  uint32_t dst_idx    : LOG_ACC_BUFF_DEPTH;
  /*! \brief Source index (indexes input buffer for GEMM or accum buffer for ALU) */
  uint32_t src_idx    : LOG_ACC_BUFF_DEPTH;
  /*! \brief Weight index (indexes weight buffer) */
  uint32_t wgt_idx    : LOG_WGT_BUFF_DEPTH;
} VTAUop;

#endif // VTA_DEFINES_H_
