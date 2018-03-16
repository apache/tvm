/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_typedefs.h
 * \brief Type definitions for VTA HLS design.
 */
#ifndef VTA_TYPEDEFS_H_
#define VTA_TYPEDEFS_H_

#include <assert.h>
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

#include "vta_params.h"

/* \typedef uop_T Micro-op datatype*/
typedef ap_uint<UOP_WIDTH> uop_T;

/* \typedef inp_T Input datatype*/
typedef ap_int<INP_WIDTH> inp_T;

/* \typedef wgt_T Weight datatype*/
typedef ap_int<WGT_WIDTH> wgt_T;

/* \typedef out_T Output datatype*/
typedef ap_int<OUT_WIDTH> out_T;

/* \typedef acc_T Accumulator datatype*/
typedef ap_int<ACC_WIDTH> acc_T;

/* \typedef mul_T Multiplier output datatype*/
typedef ap_int<WGT_WIDTH+INP_WIDTH+1> mul_T;

/* \typedef sum_T GEMM accumulator datatype*/
typedef ap_int<WGT_WIDTH+INP_WIDTH+LOG_BLOCK_IN+1> sum_T;

/* \typedef inp_vec_T Input vector datatype*/
typedef ap_uint<INP_WIDTH*BLOCK_IN> inp_vec_T;

/* \typedef wgt_vec_T Weight vector datatype*/
typedef ap_uint<WGT_WIDTH*BLOCK_IN> wgt_vec_T;

/* \typedef acc_vec_T Accumulator vector datatype*/
typedef ap_uint<ACC_WIDTH*BLOCK_OUT> acc_vec_T;

/* \typedef out_vec_T Output vector datatype*/
typedef ap_uint<OUT_WIDTH*BLOCK_OUT> out_vec_T;

/* \typedef uop_idx_T Micro-op SRAM index datatype*/
typedef ap_uint<LOG_UOP_BUFF_DEPTH+1> uop_idx_T;

/* \typedef inp_idx_T Input SRAM index datatype*/
typedef ap_uint<LOG_INP_BUFF_DEPTH+1> inp_idx_T;

/* \typedef wgt_idx_T Weight SRAM index datatype*/
typedef ap_uint<LOG_WGT_BUFF_DEPTH+1> wgt_idx_T;

/* \typedef acc_idx_T Accumulator SRAM index datatype*/
typedef ap_uint<LOG_ACC_BUFF_DEPTH+1> acc_idx_T;

/* \typedef opcode_T Opcode datatype*/
typedef ap_uint<OPCODE_BIT_WIDTH> opcode_T;

/* \typedef insn_T Instruction datatype*/
typedef ap_uint<INS_WIDTH> insn_T;

/* \typedef loop_T Loop bound datatype*/
typedef ap_uint<LOOP_ITER_WIDTH> loop_T;

/* \typedef memop_id_T Memory operation ID datatype*/
typedef ap_uint<MEMOP_ID_BIT_WIDTH> memop_id_T;

/* \typedef memop_sram_T Memory operation SRAM index datatype*/
typedef ap_uint<MEMOP_SRAM_ADDR_BIT_WIDTH> memop_sram_T;

/* \typedef memop_dram_T Memory operation DRAM index datatype*/
typedef ap_uint<MEMOP_DRAM_ADDR_BIT_WIDTH> memop_dram_T;

/* \typedef memop_size_T Memory operation range datatype*/
typedef ap_uint<MEMOP_SIZE_BIT_WIDTH> memop_size_T;

/* \typedef memop_stride_T Memory operation stride datatype*/
typedef ap_uint<MEMOP_STRIDE_BIT_WIDTH> memop_stride_T;

/* \typedef memop_pad_T Memory operation pad width datatype*/
typedef ap_uint<MEMOP_PAD_BIT_WIDTH> memop_pad_T;

/* \typedef aluop_opcode_T ALU operation opcode datatype*/
typedef ap_uint<ALU_OPCODE_BIT_WIDTH> aluop_opcode_T;

/* \typedef aluop_opcode_T ALU operation immediate datatype*/
typedef ap_int<ALUOP_IMM_BIT_WIDTH> aluop_imm_T;

/* \typedef aluop_opcode_T ALU operation shift immediate datatype*/
typedef ap_uint<LOG_ACC_WIDTH> aluop_sh_imm_T;

#endif // VTA_TYPEDEFS_H_
