/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta.cpp
 * \brief VTA HLS design.
 */

#include <stdio.h>
#include <stdlib.h>

#define __CORRECT_ISO_CPP_STDLIB_H_PROTO

#include "vta.h"

void _memcpy(
  unsigned char * __dst, 
  const unsigned char * __src, 
  size_t __n) {
#pragma unroll 4
  for (int i = 0; i < __n; ++i) {
    __dst[i] = __src[i];
  }
}

void stream_copy(
  ihc::stream_out<bool> &__src,
  ihc::stream_in<bool> &__dst) {
  while (!__src.empty()) {
    __dst.write(__src.read());
  }
}

component void fetch(
  uint32_t insn_count,
  ihc::stream_in<insn_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > &insns,
  ihc::stream_out<insn_T> &load_queue,
  ihc::stream_out<insn_T> &gemm_queue,
  ihc::stream_out<insn_T> &store_queue) {
  for (int pc = 0; pc < insn_count; pc++) {
    // Read instruction fields
    insn_T insn = insns.read();
    // Do some partial decoding
    opcode_T opcode = insn.slc<VTA_INSN_MEM_0_1-VTA_INSN_MEM_0_0+1>(VTA_INSN_MEM_0_0);
    memop_id_T memory_type = insn.slc<VTA_INSN_MEM_5_1-VTA_INSN_MEM_5_0+1>(VTA_INSN_MEM_5_0);
    // Push to appropriate instruction queue
    if (opcode == VTA_OPCODE_STORE) {
      store_queue.write(insn);
    } else if (opcode == VTA_OPCODE_LOAD &&
          (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT)) {
      load_queue.write(insn);
    } else {
      gemm_queue.write(insn);
    }
  }
}

component void load(
  ihc::stream_in<inp_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > &inputs,
  ihc::stream_in<wgt_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > &weights,
  ihc::stream_in<insn_T> &load_queue,
  ihc::stream_in<bool> &g2l_dep_queue,
  ihc::stream_out<bool> &l2g_dep_queue,
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT]
  ) {
  // Pop load instruction
  insn_T insn = load_queue.read();

  // Decode instruction
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.slc<VTA_INSN_MEM_5_1-VTA_INSN_MEM_5_0+1>(VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.slc<VTA_INSN_MEM_6_1-VTA_INSN_MEM_6_0+1>(VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.slc<VTA_INSN_MEM_7_1-VTA_INSN_MEM_7_0+1>(VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.slc<VTA_INSN_MEM_8_1-VTA_INSN_MEM_8_0+1>(VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.slc<VTA_INSN_MEM_9_1-VTA_INSN_MEM_9_0+1>(VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.slc<VTA_INSN_MEM_A_1-VTA_INSN_MEM_A_0+1>(VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.slc<VTA_INSN_MEM_B_1-VTA_INSN_MEM_B_0+1>(VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.slc<VTA_INSN_MEM_C_1-VTA_INSN_MEM_C_0+1>(VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.slc<VTA_INSN_MEM_D_1-VTA_INSN_MEM_D_0+1>(VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.slc<VTA_INSN_MEM_E_1-VTA_INSN_MEM_E_0+1>(VTA_INSN_MEM_E_0);

  // Pop dependence token if instructed
  if (pop_next_dependence) {
    g2l_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = sram_base;
  memop_dram_T dram_idx = dram_base;

  // Pre-compute dimensions, and offsets
  memop_size_T y_size_total = y_pad_0 + y_size + y_pad_1;
  memop_size_T x_size_total = x_pad_0 + x_size + x_pad_1;
  memop_sram_T y_offset = x_size_total * y_pad_0;

  // Skip padding along y dimension
  sram_idx += y_offset;

  // Perform data transfer from DRAM
  for (int y = 0; y < y_size; y++) {
    // Skip padding along x dimension
    sram_idx += x_pad_0;
    // Perform data transfer
    if (memory_type == VTA_MEM_ID_INP) {
      // _memcpy((unsigned char*)&inp_mem[sram_idx][0],
      //        (const unsigned char*)&inputs[dram_idx * VTA_BATCH],
      //        x_size * VTA_INP_ELEM_BYTES);
      for (int x = 0; x < x_size; x++) {
        inp_mem[sram_idx][0] = inputs.read();
      }
    } else {
      // _memcpy((unsigned char*)&wgt_mem[sram_idx][0],
      //        (const unsigned char*) &weights[dram_idx * VTA_BLOCK_OUT],
      //        x_size * VTA_WGT_ELEM_BYTES);
      for (int x = 0; x < x_size; x++) {
        wgt_mem[sram_idx][0] = weights.read();
      }
    }
    sram_idx += x_size;
    dram_idx += x_stride;
    // Skip padding along x dimension
    sram_idx += x_pad_1;
  }

  // Reset SRAM index
  sram_idx = sram_base;
  // Pad x/y edges with zeros
  for (int y = 0; y < y_size_total; y++) {
    if (y < y_pad_0 || y >= y_pad_0 + y_size) {
      for (int x = 0; x < x_size_total; x++) {
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
#pragma unroll 4
          for (int i = 0; i < VTA_BLOCK_OUT; i++) {
            wgt_mem[sram_idx][i] = 0;
          }
        }
        sram_idx++;
      }
    } else {
      for (int x = 0; x < x_pad_0; x++) {
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
#pragma unroll 4
          for (int i = 0; i < VTA_BLOCK_OUT; i++) {
            wgt_mem[sram_idx][i] = 0;
          }
        }
        sram_idx++;
      }
      sram_idx += x_size;
      for (int x = 0; x < x_pad_1; x++) {
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
#pragma unroll 4
          for (int i = 0; i < VTA_BLOCK_OUT; i++) {
            wgt_mem[sram_idx][i] = 0;
          }
        }
        sram_idx++;
      }
    }
  }

  // Push dependence token if instructed
  if (push_next_dependence) {
    l2g_dep_queue.write(1);
  }
}

component void compute(
  hls_avalon_slave_memory_argument(1*sizeof(uint32_t)) uint32_t * done0,
  ihc::stream_in<uop_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > & uops,
  ihc::stream_in<acc_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > & biases,
  ihc::stream_in<insn_T> &gemm_queue,
  ihc::stream_in<bool> &l2g_dep_queue,
  ihc::stream_in<bool> &s2g_dep_queue,
  ihc::stream_out<bool> &g2l_dep_queue,
  ihc::stream_out<bool> &g2s_dep_queue,
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT],
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]
  ) {
  // Micro-op storage
  static uop_T uop_mem[VTA_UOP_BUFF_DEPTH];

  // Accumulator storage
  static acc_vec_T acc_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH];

  // Pop GEMM instruction
  insn_T insn = gemm_queue.read();

  // Decode
  opcode_T opcode = insn.slc<VTA_INSN_MEM_0_1-VTA_INSN_MEM_0_0+1>(VTA_INSN_MEM_0_0);
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];

  // Pop dependence token if instructed
  if (pop_prev_dependence) {
    l2g_dep_queue.read();
  }
  if (pop_next_dependence) {
    s2g_dep_queue.read();
  }

  // Perform action based on opcode
  if (opcode == VTA_OPCODE_FINISH) {
    // Set done flag if we reach a FINISH instruction
    *done0 = 1;
  } else if (opcode == VTA_OPCODE_LOAD || opcode == VTA_OPCODE_STORE) {
    // Set done value
    *done0 = 0;

    // Decode instruction
    memop_id_T memory_type = insn.slc<VTA_INSN_MEM_5_1-VTA_INSN_MEM_5_0+1>(VTA_INSN_MEM_5_0);
    memop_sram_T sram_base = insn.slc<VTA_INSN_MEM_6_1-VTA_INSN_MEM_6_0+1>(VTA_INSN_MEM_6_0);
    memop_dram_T dram_base = insn.slc<VTA_INSN_MEM_7_1-VTA_INSN_MEM_7_0+1>(VTA_INSN_MEM_7_0);
    memop_size_T y_size = insn.slc<VTA_INSN_MEM_8_1-VTA_INSN_MEM_8_0+1>(VTA_INSN_MEM_8_0);
    memop_size_T x_size = insn.slc<VTA_INSN_MEM_9_1-VTA_INSN_MEM_9_0+1>(VTA_INSN_MEM_9_0);
    memop_stride_T x_stride = insn.slc<VTA_INSN_MEM_A_1-VTA_INSN_MEM_A_0+1>(VTA_INSN_MEM_A_0);
    memop_pad_T y_pad_0 = insn.slc<VTA_INSN_MEM_B_1-VTA_INSN_MEM_B_0+1>(VTA_INSN_MEM_B_0);
    memop_pad_T y_pad_1 = insn.slc<VTA_INSN_MEM_C_1-VTA_INSN_MEM_C_0+1>(VTA_INSN_MEM_C_0);
    memop_pad_T x_pad_0 = insn.slc<VTA_INSN_MEM_D_1-VTA_INSN_MEM_D_0+1>(VTA_INSN_MEM_D_0);
    memop_pad_T x_pad_1 = insn.slc<VTA_INSN_MEM_E_1-VTA_INSN_MEM_E_0+1>(VTA_INSN_MEM_E_0);

    // Initialize indices
    memop_sram_T sram_idx = sram_base;
    memop_dram_T dram_idx = dram_base;

    // Pre-compute dimensions, and offsets
    memop_size_T y_size_total = y_pad_0 + y_size + y_pad_1;
    memop_size_T x_size_total = x_pad_0 + x_size + x_pad_1;
    memop_sram_T y_offset = x_size_total * y_pad_0;

    if (memory_type == VTA_MEM_ID_UOP) {
      // Perform data transfer
      // _memcpy((unsigned char*)&uop_mem[sram_base],
      //        (const unsigned char*) &uops[dram_base],
      //        x_size * VTA_UOP_ELEM_BYTES);
      for (int x = 0; x < x_size; x++) {
        uop_mem[sram_base] = uops.read();
      }
    } else {
      // Skip vertical padding
      sram_idx += y_offset;
      // Perform data transfer from DRAM
      for (int y = 0; y < y_size; y++) {
        // Skip padding along x dimension
        sram_idx += x_pad_0;
        // Perform data transfer
        // _memcpy((unsigned char*) &acc_mem[sram_idx][0],
        //        (const unsigned char*) &biases[dram_idx * VTA_BATCH],
        //        x_size*VTA_ACC_ELEM_BYTES);
        for (int x = 0; x < x_size; x++) {
          acc_mem[sram_idx][0] = biases.read();
        }
        sram_idx += x_size;
        dram_idx += x_stride;
        // Skip padding along x dimension
        sram_idx += x_pad_1;
      }
    }
  } else if (opcode == VTA_OPCODE_GEMM || opcode == VTA_OPCODE_ALU) {
    // Set done value
    *done0 = 0;

    // Decode
    bool reset_out = insn[VTA_INSN_GEM_5];
    uop_idx_T uop_bgn = insn.slc<VTA_INSN_GEM_6_1-VTA_INSN_GEM_6_0+1>(VTA_INSN_GEM_6_0);
    uop_idx_T uop_end = insn.slc<VTA_INSN_GEM_7_1-VTA_INSN_GEM_7_0+1>(VTA_INSN_GEM_7_0);
    loop_T iter_out  = insn.slc<VTA_INSN_GEM_8_1-VTA_INSN_GEM_8_0+1>(VTA_INSN_GEM_8_0);
    loop_T iter_in  = insn.slc<VTA_INSN_GEM_9_1-VTA_INSN_GEM_9_0+1>(VTA_INSN_GEM_9_0);
    acc_idx_T dst_factor_out = insn.slc<VTA_INSN_GEM_A_1-VTA_INSN_GEM_A_0+1>(VTA_INSN_GEM_A_0);
    acc_idx_T dst_factor_in = insn.slc<VTA_INSN_GEM_B_1-VTA_INSN_GEM_B_0+1>(VTA_INSN_GEM_B_0);
    inp_idx_T src_factor_out = insn.slc<VTA_INSN_GEM_C_1-VTA_INSN_GEM_C_0+1>(VTA_INSN_GEM_C_0);
    inp_idx_T src_factor_in = insn.slc<VTA_INSN_GEM_D_1-VTA_INSN_GEM_D_0+1>(VTA_INSN_GEM_D_0);

    // GEMM-specific fields
    wgt_idx_T wgt_factor_out = insn.slc<VTA_INSN_GEM_E_1-VTA_INSN_GEM_E_0+1>(VTA_INSN_GEM_E_0);
    wgt_idx_T wgt_factor_in = insn.slc<VTA_INSN_GEM_F_1-VTA_INSN_GEM_F_0+1>(VTA_INSN_GEM_F_0);

    // ALU-specific field
    aluop_opcode_T alu_opcode = insn.slc<VTA_INSN_ALU_E_1-VTA_INSN_ALU_E_0+1>(VTA_INSN_ALU_E_0);
    bool use_imm = insn[VTA_INSN_ALU_F];
    aluop_imm_T imm = insn.slc<VTA_INSN_ALU_G_1-VTA_INSN_ALU_G_0+1>(VTA_INSN_ALU_G_0);
    acc_idx_T dst_offset_out = 0;
    inp_idx_T src_offset_out = 0;
    wgt_idx_T wgt_offset_out = 0;

    // Outer Loop
#pragma ivdep array(acc_mem)
    for (int it_out = 0; it_out < iter_out; it_out++) {
      acc_idx_T dst_offset_in = dst_offset_out;
      inp_idx_T src_offset_in = src_offset_out;
      wgt_idx_T wgt_offset_in = wgt_offset_out;

      // Inner Loop
      for (int it_in = 0; it_in < iter_in; it_in++) {
        // Perform appropriate computation based on opcode
        if (opcode == VTA_OPCODE_GEMM) {
          // Iterate over micro op
          for (int upc = uop_bgn; upc < uop_end; upc++) {
            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode indices
            acc_idx_T dst_idx =
              uop.slc<VTA_UOP_GEM_0_1-VTA_UOP_GEM_0_0+1>(VTA_UOP_GEM_0_0) + dst_offset_in;
            inp_idx_T src_idx =
              uop.slc<VTA_UOP_GEM_1_1-VTA_UOP_GEM_1_0+1>(VTA_UOP_GEM_1_0) + src_offset_in;
            wgt_idx_T wgt_idx =
              uop.slc<VTA_UOP_GEM_2_1-VTA_UOP_GEM_2_0+1>(VTA_UOP_GEM_2_0) + wgt_offset_in;

            // Read weight matrix
            wgt_vec_T w_matrix[VTA_BLOCK_OUT];
#pragma unroll
            for (int i = 0; i < VTA_BLOCK_OUT; i++) {
              w_matrix[i] = wgt_mem[wgt_idx][i];
            }
            // Read input matrix and accum matrix
            acc_vec_T o_matrix[VTA_BATCH];
            inp_vec_T i_matrix[VTA_BATCH];
            for (int i = 0; i < VTA_BATCH; i++) {
              o_matrix[i] = acc_mem[dst_idx][i];
              i_matrix[i] = inp_mem[src_idx][i];
            }
            // Result matrices
            acc_vec_T acc_mem_val[VTA_BATCH];
            out_vec_T st_buf_val[VTA_BATCH];

            // Inner GEMM loop
            for (int i = 0; i < VTA_BATCH; i++) {
#pragma unroll 8
              for (int b = 0; b < VTA_BLOCK_OUT; b++) {
                // Initialize the accumulator values
                acc_T accum =
                  o_matrix[i].slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH);
                // Dot product sum
                sum_T tmp = 0;
                // Inner matrix multiplication loop (input channel/feature)
#pragma unroll
                for (int k = 0; k < VTA_BLOCK_IN; k++) {
                  wgt_T w_elem =
                    w_matrix[b].slc<VTA_WGT_WIDTH>(k * VTA_WGT_WIDTH);
                  inp_T i_elem =
                    i_matrix[i].slc<VTA_INP_WIDTH>(k * VTA_INP_WIDTH);
                  mul_T prod = i_elem * w_elem;
                  tmp += (sum_T) prod;
                }
                // Update summation
                accum += (acc_T) tmp;
                // Update result vector
                acc_mem_val[i].set_slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH,
                                                      reset_out ? (acc_T) 0 : accum);
                st_buf_val[i].set_slc<VTA_OUT_WIDTH>(b * VTA_OUT_WIDTH,
                                                     (out_T) accum.slc<VTA_OUT_WIDTH>(0));
              }
              // Write to buffers
              acc_mem[dst_idx][i] = acc_mem_val[i];
              out_mem[dst_idx][i] = st_buf_val[i];
            }
          }
        }
#ifndef NO_ALU
        else if (opcode == VTA_OPCODE_ALU) {
          // Iterate over micro op
          for (int upc = uop_bgn; upc < uop_end; upc++) {
            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode
            acc_idx_T dst_idx =
              uop.slc<VTA_UOP_ALU_0_1-VTA_UOP_ALU_0_0+1>(VTA_UOP_ALU_0_0) + dst_offset_in;
            acc_idx_T src_idx =
              uop.slc<VTA_UOP_ALU_1_1-VTA_UOP_ALU_1_0+1>(VTA_UOP_ALU_1_0) + src_offset_in;

            // Perform ALU op over matrix elements
            for (int i = 0; i < VTA_BATCH; i++) {
              // Read input matrix and accum matrix
              acc_vec_T dst_vector = acc_mem[dst_idx][i];
              acc_vec_T src_vector = acc_mem[src_idx][i];
              // Result matrices
              acc_vec_T cmp_res;
              acc_vec_T add_res;
              acc_vec_T shr_res;
              out_vec_T short_cmp_res;
              out_vec_T short_add_res;
              out_vec_T short_shr_res;
              // Results vector
              acc_vec_T res_vec = 0;
#pragma ii 1
              for (int b = 0; b < VTA_BLOCK_OUT; b++) {
                // Read in operands
                acc_T src_0 = dst_vector.slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH);
                acc_T src_1 = use_imm ? (acc_T) imm : (acc_T) src_vector.slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH);
                // Compute Min/Max
                acc_T mix_val = src_0 < src_1 ?
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
                cmp_res.set_slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH, mix_val);
                short_cmp_res.set_slc<VTA_OUT_WIDTH>(b * VTA_OUT_WIDTH, mix_val.slc<VTA_OUT_WIDTH>(0));
                // Compute Sum
                acc_T add_val = src_0.slc<VTA_ACC_WIDTH>(0) + src_1.slc<VTA_ACC_WIDTH>(0);
                add_res.set_slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH, add_val);
                short_add_res.set_slc<VTA_OUT_WIDTH>(b * VTA_OUT_WIDTH, add_val.slc<VTA_OUT_WIDTH>(0));
                // Compute Shift Right
                acc_T shr_val = src_0 >> (aluop_sh_imm_T) src_1.slc<VTA_LOG_ACC_WIDTH>(0);
                shr_res.set_slc<VTA_ACC_WIDTH>(b * VTA_ACC_WIDTH, shr_val);
                short_shr_res.set_slc<VTA_OUT_WIDTH>(b * VTA_OUT_WIDTH, shr_val.slc<VTA_OUT_WIDTH>(0));
              }

              // Store to accum memory/store buffer
              // if (alu_opcode == VTA_ALU_OPCODE_MIN ||
              //     alu_opcode == VTA_ALU_OPCODE_MAX) {
              //   acc_mem[dst_idx][i] = cmp_res;
              //   out_mem[dst_idx][i] = short_cmp_res;
              // } else if (alu_opcode == VTA_ALU_OPCODE_ADD) {
              //   acc_mem[dst_idx][i] = add_res;
              //   out_mem[dst_idx][i] = short_add_res;
              // } else if (alu_opcode == VTA_ALU_OPCODE_SHR) {
              //   acc_mem[dst_idx][i] = shr_res;
              //   out_mem[dst_idx][i] = short_shr_res;
              // }
            }
          }
        }
#endif  // NO_ALU

        // Update offsets
        dst_offset_in += dst_factor_in;
        src_offset_in += src_factor_in;
        wgt_offset_in += wgt_factor_in;
      }

      // Update offsets
      dst_offset_out += dst_factor_out;
      src_offset_out += src_factor_out;
      wgt_offset_out += wgt_factor_out;
    }
  }

  // Push dependence token if instructed
  if (push_prev_dependence) {
    g2l_dep_queue.write(1);
  }
  if (push_next_dependence) {
    g2s_dep_queue.write(1);
  }
}

component void store(
  // hls_avalon_slave_memory_argument(VTA_ACC_BUFF_DEPTH*VTA_BATCH*VTA_ACC_ELEM_BYTES) out_vec_T * outputs,
  // ihc::mm_master<out_vec_T, ihc::aspace<6>, ihc::dwidth<64>, ihc::waitrequest<true>,
  //                ihc::align<8*sizeof(uint64_t)>, ihc::latency<0> >& outputs,
  ihc::stream_out<out_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > & outputs,
  ihc::stream_in<insn_T> &store_queue,
  ihc::stream_in<bool> &g2s_dep_queue,
  ihc::stream_out<bool> &s2g_dep_queue,
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]
  ) {
  // Load buffer
  insn_T insn = store_queue.read();

  // Decode
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.slc<VTA_INSN_MEM_5_1-VTA_INSN_MEM_5_0+1>(VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.slc<VTA_INSN_MEM_6_1-VTA_INSN_MEM_6_0+1>(VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.slc<VTA_INSN_MEM_7_1-VTA_INSN_MEM_7_0+1>(VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.slc<VTA_INSN_MEM_8_1-VTA_INSN_MEM_8_0+1>(VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.slc<VTA_INSN_MEM_9_1-VTA_INSN_MEM_9_0+1>(VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.slc<VTA_INSN_MEM_A_1-VTA_INSN_MEM_A_0+1>(VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.slc<VTA_INSN_MEM_B_1-VTA_INSN_MEM_B_0+1>(VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.slc<VTA_INSN_MEM_C_1-VTA_INSN_MEM_C_0+1>(VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.slc<VTA_INSN_MEM_D_1-VTA_INSN_MEM_D_0+1>(VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.slc<VTA_INSN_MEM_E_1-VTA_INSN_MEM_E_0+1>(VTA_INSN_MEM_E_0);

  // Pop dependence token if instructed
  if (pop_prev_dependence) {
    g2s_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = sram_base;
  memop_dram_T dram_idx = dram_base;

  // Skip padding along y dimension
  memop_sram_T y_offset = (x_pad_0 + x_size + x_pad_1) * y_pad_0;
  sram_idx += y_offset;
// Force this computation to be done with LUTs to avoid using too many DSPs
#pragma HLS RESOURCE variable = y_offset core = Mul_LUT

  // Copy along y dimension
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE rewind
    // Skip padding along x dimension
    sram_idx += x_pad_0;
    // Perform data transfer
    // _memcpy((unsigned char*)(&outputs[dram_idx*VTA_BATCH]),
    //         (const unsigned char*) &out_mem[sram_idx][0],
    //   x_size * VTA_INP_ELEM_BYTES);
    for (int x = 0; x < x_size; x++) {
      outputs.write(out_mem[sram_idx][0]);
    }
    sram_idx += x_size;
    dram_idx += x_stride;
    // Skip padding along x dimension
    sram_idx += x_pad_1;
  }

  // Push dependence token if instructed
  if (push_prev_dependence) {
    s2g_dep_queue.write(1);
  }
}

void vta(
  uint32_t insn_count,
  insn_T *insns,
  uop_T *uops,
  inp_vec_T *inputs,
  wgt_vec_T *weights,
  acc_vec_T *biases,
  out_vec_T *outputs) {
  // Instantiate temporary instruction queues (used for peeking)
  ihc::stream_out<insn_T> tmp_load_queue;
  ihc::stream_out<insn_T> tmp_gemm_queue;
  ihc::stream_out<insn_T> tmp_store_queue;

  // Instatiate physical instruction queues
  ihc::stream_in<insn_T> load_queue;
  ihc::stream_in<insn_T> gemm_queue;
  ihc::stream_in<insn_T> store_queue;

  // Dependence input queues
  ihc::stream_out<bool> l2g_dep_queue;
  ihc::stream_out<bool> s2g_dep_queue;
  ihc::stream_in<bool> g2l_dep_queue;
  ihc::stream_in<bool> g2s_dep_queue;

  ihc::stream_in<bool> l2g_dep_queue_cmp;
  ihc::stream_in<bool> s2g_dep_queue_cmp;
  ihc::stream_out<bool> g2l_dep_queue_cmp;
  ihc::stream_out<bool> g2s_dep_queue_cmp;

  // Instantiate memories
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH];
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT];
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH];

  ihc::stream_in<insn_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > insns_stream;
  ihc::stream_in<inp_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > inputs_stream;
  ihc::stream_in<wgt_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > weights_stream;
  ihc::stream_in<uop_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > uops_stream;
  ihc::stream_in<acc_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > biases_stream;
  ihc::stream_out<out_vec_T, ihc::usesPackets<false>, ihc::bitsPerSymbol<8> > outputs_stream;
  for (int i = 0; i < insn_count; i++) {
    insns_stream.write(insns);
  }
  for (int i = 0; i < VTA_INP_BUFF_DEPTH*VTA_BATCH; i++) {
    inputs_stream.write(insns);
  }
  for (int i = 0; i < VTA_WGT_BUFF_DEPTH*VTA_BLOCK_OUT; i++) {
    weights_stream.write(insns);
  }
  for (int i = 0; i < VTA_UOP_BUFF_DEPTH; i++) {
    uops_stream.write(uops);
  }
  for (int i = 0; i < VTA_ACC_BUFF_DEPTH*VTA_BATCH; i++) {
    biases_stream.write(biases);
  }

  // Push all instructions into the queues
  fetch(insn_count, insns_stream, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

  // Global done indicator
  uint32_t done = 0;

  // Temporary instructions
  insn_T tmp_load;
  insn_T tmp_gemv;
  insn_T tmp_store;

  // Peeking status
  bool tmp_load_popped = false;
  bool tmp_gemm_popped = false;
  bool tmp_store_popped = false;
  int exit_counter = 0;

  // Main control loop
  while (true) {
    // First execute as many load instructions as possible
    while (!tmp_load_queue.empty() || tmp_load_popped == true) {
      // Pop the load instruction
      if (!tmp_load_popped) {
        tmp_load = tmp_load_queue.read();
        tmp_load_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_next_dependence = tmp_load[VTA_INSN_MEM_2];
      if ((pop_next_dependence && !g2l_dep_queue.empty()) ||
          !pop_next_dependence) {
        // Push the instruction in the load queue
        load_queue.write(tmp_load);
        tmp_load_popped = false;
        load(inputs_stream, weights_stream, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Next execute as many gemm instructions as possible
    while (!tmp_gemm_queue.empty() || tmp_gemm_popped == true) {
      // Pop the gemm instruction
      if (!tmp_gemm_popped) {
        tmp_gemv = tmp_gemm_queue.read();
        tmp_gemm_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_prev_dependence = tmp_gemv[VTA_INSN_MEM_1];
      bool pop_next_dependence = tmp_gemv[VTA_INSN_MEM_2];
      if (
        (pop_prev_dependence && !l2g_dep_queue.empty() &&
         pop_next_dependence && !s2g_dep_queue.empty()) ||
        (!pop_prev_dependence && pop_next_dependence &&
         !s2g_dep_queue.empty()) ||
        (pop_prev_dependence && !l2g_dep_queue.empty() &&
        !pop_next_dependence) ||
        (!pop_prev_dependence && !pop_next_dependence)
      ) {
        // Push the instruction in the load queue
        gemm_queue.write(tmp_gemv);
        tmp_gemm_popped = false;
        stream_copy(l2g_dep_queue, l2g_dep_queue_cmp);
        stream_copy(s2g_dep_queue, s2g_dep_queue_cmp);
        compute(&done, uops_stream, biases_stream, gemm_queue, l2g_dep_queue_cmp, s2g_dep_queue_cmp,
                g2l_dep_queue_cmp, g2s_dep_queue_cmp, inp_mem, wgt_mem, out_mem);
        stream_copy(g2l_dep_queue_cmp, g2l_dep_queue);
        stream_copy(g2s_dep_queue_cmp, g2s_dep_queue);
      } else {
        // Execution of load stage pending on completion of other stages,
        // so break here...
        break;
      }
    }
    // Finally execute as many store instructions as possible
    while (!tmp_store_queue.empty() || tmp_store_popped == true) {
      // Pop the load instruction
      if (!tmp_store_popped) {
        tmp_store = tmp_store_queue.read();
        tmp_store_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_prev_dependence = tmp_store[VTA_INSN_MEM_1];
      if ((pop_prev_dependence && !g2s_dep_queue.empty()) ||
          !pop_prev_dependence) {
        // Push the instruction in the load queue
        store_queue.write(tmp_store);
        tmp_store_popped = false;
        store(outputs_stream, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Check if we get a signal that we are done
    if (done) {
      break;
    }
    exit_counter++;
    if (exit_counter > 1000) {
      if (tmp_load_popped) {
        if (g2l_dep_queue.empty()) {
          printf("waiting on g2l\n");
        }
      }
      if (tmp_gemm_popped) {
        if (l2g_dep_queue.empty() && tmp_gemv[VTA_INSN_MEM_1]) {
          printf("waiting on l2g\n");
        }
        if (s2g_dep_queue.empty() && tmp_gemv[VTA_INSN_MEM_2]) {
          printf("waiting on s2g\n");
        }
      }
      if (tmp_store_popped) {
        if (g2s_dep_queue.empty()) {
          printf("waiting on g2s\n");
        }
      }
      break;
    }
  }

  // Ensure that the tokens are empty
  int l2g_count = -1;
  int s2g_count = -1;
  int g2l_count = -1;
  int g2s_count = -1;
  bool success = false;
  do {
    l2g_dep_queue.tryRead(success);
    l2g_count++;
  } while (success);
  do {
    s2g_dep_queue.tryRead(success);
    s2g_count++;
  } while (success);
  do {
    g2l_dep_queue.tryRead(success);
    g2l_count++;
  } while (success);
  do {
    g2s_dep_queue.tryRead(success);
    g2s_count++;
  } while (success);

  assert(l2g_count == 0 && g2s_count == 0 && g2l_count == 0 && g2s_count == 0);
}
