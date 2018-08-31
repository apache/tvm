/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta.cpp
 * \brief VTA HLS design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vta.h"

void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  INSN_DECODE: for (int pc = 0; pc < insn_count; pc++) {
#pragma HLS PIPELINE II = 1
    // Read instruction fields
    insn_T insn = insns[pc];
    // Do some partial decoding
    opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
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

void load(
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT]
  ) {
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = g2l_dep_queue
#pragma HLS INTERFACE axis port = l2g_dep_queue
#pragma HLS INTERFACE bram port = wgt_mem
#pragma HLS INTERFACE bram port = inp_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  // Pop load instruction
  insn_T insn = load_queue.read();

  // Decode instruction
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

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
// Force this computation to be done with LUTs to avoid using too many DSPs
#pragma HLS RESOURCE variable = y_offset core = Mul_LUT

  // Skip padding along y dimension
  sram_idx += y_offset;

  // Perform data transfer from DRAM
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE rewind
    // Skip padding along x dimension
    sram_idx += x_pad_0;
    // Perform data transfer
    if (memory_type == VTA_MEM_ID_INP) {
      memcpy(&inp_mem[sram_idx][0],
             (const inp_vec_T*) &inputs[dram_idx * VTA_BATCH],
             x_size * VTA_INP_ELEM_BYTES);
    } else {
      memcpy(&wgt_mem[sram_idx][0],
             (const wgt_vec_T*) &weights[dram_idx * VTA_BLOCK_OUT],
             x_size * VTA_WGT_ELEM_BYTES);
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
#pragma HLS PIPELINE II = 1 rewind
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
          for (int i = 0; i < VTA_BLOCK_OUT; i++) {
            wgt_mem[sram_idx][i] = 0;
          }
        }
        sram_idx++;
      }
    } else {
      for (int x = 0; x < x_pad_0; x++) {
#pragma HLS PIPELINE II = 1 rewind
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
          for (int i = 0; i < VTA_BLOCK_OUT; i++) {
            wgt_mem[sram_idx][i] = 0;
          }
        }
        sram_idx++;
      }
      sram_idx += x_size;
      for (int x = 0; x < x_pad_1; x++) {
#pragma HLS PIPELINE II = 1 rewind
        if (memory_type == VTA_MEM_ID_INP) {
          for (int i = 0; i < VTA_BATCH; i++) {
            inp_mem[sram_idx][i] = 0;
          }
        } else {
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

void compute(
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile acc_vec_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH],
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT],
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]
  ) {
#pragma HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uop_port
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = l2g_dep_queue
#pragma HLS INTERFACE axis port = s2g_dep_queue
#pragma HLS INTERFACE axis port = g2l_dep_queue
#pragma HLS INTERFACE axis port = g2s_dep_queue
#pragma HLS INTERFACE bram port = inp_mem
#pragma HLS INTERFACE bram port = wgt_mem
#pragma HLS INTERFACE bram port = out_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
// This is necessary connect the SRAM to the load module
#pragma HLS RESOURCE variable = wgt_mem core = RAM_1P

  // Micro-op storage
  static uop_T uop_mem[VTA_UOP_BUFF_DEPTH];

  // Accumulator storage
  static acc_vec_T acc_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH];
#pragma HLS ARRAY_PARTITION variable = acc_mem complete dim = 2

  // Pop GEMM instruction
  insn_T insn = gemm_queue.read();

  // Decode
  opcode_T opcode = insn.range(VTA_INSN_MEM_0_1, VTA_INSN_MEM_0_0);
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
    done = 1;
  } else if (opcode == VTA_OPCODE_LOAD || opcode == VTA_OPCODE_STORE) {
    // Set done value
    done = 0;

    // Decode instruction
    memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
    memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
    memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
    memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
    memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
    memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
    memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
    memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
    memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
    memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

    // Initialize indices
    memop_sram_T sram_idx = sram_base;
    memop_dram_T dram_idx = dram_base;

    // Pre-compute dimensions, and offsets
    memop_size_T y_size_total = y_pad_0 + y_size + y_pad_1;
    memop_size_T x_size_total = x_pad_0 + x_size + x_pad_1;
    memop_sram_T y_offset = x_size_total * y_pad_0;
// Force this computation to be done with LUTs to avoid using too many DSPs
#pragma HLS RESOURCE variable = y_offset core = Mul_LUT

    if (memory_type == VTA_MEM_ID_UOP) {
      // Perform data transfer
      memcpy(&uop_mem[sram_base],
             (const uop_T*) &uops[dram_base],
             x_size * sizeof(uop_T));
    } else {
      // Skip vertical padding
      sram_idx += y_offset;
      // Perform data transfer from DRAM
      for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE rewind
        // Skip padding along x dimension
        sram_idx += x_pad_0;
        // Perform data transfer
        memcpy(&acc_mem[sram_idx][0],
               (const acc_vec_T*) &biases[dram_idx * VTA_BATCH],
               x_size*VTA_ACC_ELEM_BYTES);
        sram_idx += x_size;
        dram_idx += x_stride;
        // Skip padding along x dimension
        sram_idx += x_pad_1;
      }
    }
  } else if (opcode == VTA_OPCODE_GEMM || opcode == VTA_OPCODE_ALU) {
    // Set done value
    done = 0;

    // Decode
    bool reset_out = insn[VTA_INSN_GEM_5];
    uop_idx_T uop_bgn = insn.range(VTA_INSN_GEM_6_1, VTA_INSN_GEM_6_0);
    uop_idx_T uop_end = insn.range(VTA_INSN_GEM_7_1, VTA_INSN_GEM_7_0);
    loop_T iter_out  = insn.range(VTA_INSN_GEM_8_1, VTA_INSN_GEM_8_0);
    loop_T iter_in  = insn.range(VTA_INSN_GEM_9_1, VTA_INSN_GEM_9_0);
    acc_idx_T dst_factor_out = insn.range(VTA_INSN_GEM_A_1, VTA_INSN_GEM_A_0);
    acc_idx_T dst_factor_in = insn.range(VTA_INSN_GEM_B_1, VTA_INSN_GEM_B_0);
    inp_idx_T src_factor_out = insn.range(VTA_INSN_GEM_C_1, VTA_INSN_GEM_C_0);
    inp_idx_T src_factor_in = insn.range(VTA_INSN_GEM_D_1, VTA_INSN_GEM_D_0);

    // GEMM-specific fields
    wgt_idx_T wgt_factor_out = insn.range(VTA_INSN_GEM_E_1, VTA_INSN_GEM_E_0);
    wgt_idx_T wgt_factor_in = insn.range(VTA_INSN_GEM_F_1, VTA_INSN_GEM_F_0);

    // ALU-specific field
    aluop_opcode_T alu_opcode = insn.range(VTA_INSN_ALU_E_1, VTA_INSN_ALU_E_0);
    bool use_imm = insn[VTA_INSN_ALU_F];
    aluop_imm_T imm = insn.range(VTA_INSN_ALU_G_1, VTA_INSN_ALU_G_0);
    acc_idx_T dst_offset_out = 0;
    inp_idx_T src_offset_out = 0;
    wgt_idx_T wgt_offset_out = 0;

    // Outer Loop
    EXE_OUT_LOOP: for (int it_out = 0; it_out < iter_out; it_out++) {
#pragma HLS DEPENDENCE variable = acc_mem inter false
      acc_idx_T dst_offset_in = dst_offset_out;
      inp_idx_T src_offset_in = src_offset_out;
      wgt_idx_T wgt_offset_in = wgt_offset_out;

      // Inner Loop
      EXE_IN_LOOP: for (int it_in = 0; it_in < iter_in; it_in++) {
        // Perform appropriate computation based on opcode
        if (opcode == VTA_OPCODE_GEMM) {
          // Iterate over micro op
          READ_GEMM_UOP: for (int upc = uop_bgn; upc < uop_end; upc++) {
#pragma HLS PIPELINE II = 1 rewind

            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode indices
            acc_idx_T dst_idx =
                uop.range(VTA_UOP_GEM_0_1, VTA_UOP_GEM_0_0) + dst_offset_in;
            inp_idx_T src_idx =
                uop.range(VTA_UOP_GEM_1_1, VTA_UOP_GEM_1_0) + src_offset_in;
            wgt_idx_T wgt_idx =
                uop.range(VTA_UOP_GEM_2_1, VTA_UOP_GEM_2_0) + wgt_offset_in;

            // Read weight matrix
            wgt_vec_T w_matrix[VTA_BLOCK_OUT];
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
              for (int b = 0; b < VTA_BLOCK_OUT; b++) {
                // Initialize the accumulator values
                acc_T accum =
                  o_matrix[i].range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH);
                // Dot product sum
                sum_T tmp = 0;
                // Inner matrix multiplication loop (input channel/feature)
                for (int k = 0; k < VTA_BLOCK_IN; k++) {
                  wgt_T w_elem =
                      w_matrix[b].range((k + 1) * VTA_WGT_WIDTH - 1, k * VTA_WGT_WIDTH);
                  inp_T i_elem =
                      i_matrix[i].range((k + 1) * VTA_INP_WIDTH - 1, k * VTA_INP_WIDTH);
                  mul_T prod = i_elem * w_elem;
#ifdef NO_DSP
#pragma HLS RESOURCE variable = prod core = Mul_LUT
#endif //  NO_DSP
                  tmp += (sum_T) prod;
                }
                // Update summation
                accum += (acc_T) tmp;
                // Update result vector
                acc_mem_val[i].range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH) =
                    reset_out ? (acc_T) 0 : accum;
                st_buf_val[i].range((b + 1) * VTA_OUT_WIDTH - 1, b * VTA_OUT_WIDTH) =
                    (out_T) accum.range(VTA_OUT_WIDTH - 1, 0);
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
          READ_ALU_UOP: for (int upc = uop_bgn; upc < uop_end; upc++) {
            // Read micro-op fields
            uop_T uop = uop_mem[upc];

            // Decode
            acc_idx_T dst_idx =
                uop.range(VTA_UOP_ALU_0_1, VTA_UOP_ALU_0_0) + dst_offset_in;
            acc_idx_T src_idx =
                uop.range(VTA_UOP_ALU_1_1, VTA_UOP_ALU_1_0) + src_offset_in;

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
              for (int b = 0; b < VTA_BLOCK_OUT; b++) {
#pragma HLS PIPELINE II = 1 rewind
                // Read in operands
                acc_T src_0 = dst_vector.range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH);
                acc_T src_1 = use_imm ?
                    (acc_T) imm :
                    src_vector.range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH);
                // Compute Min/Max
                acc_T mix_val = src_0 < src_1 ?
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
                    (alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
                cmp_res.range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH) = mix_val;
                short_cmp_res.range((b + 1) * VTA_OUT_WIDTH - 1, b * VTA_OUT_WIDTH) =
                    (out_T) mix_val.range(VTA_OUT_WIDTH - 1, 0);
                // Compute Sum
                acc_T add_val =
                    src_0.range(VTA_ACC_WIDTH - 1, 0) + src_1.range(VTA_ACC_WIDTH - 1, 0);
                add_res.range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH) = add_val;
                short_add_res.range((b + 1) * VTA_OUT_WIDTH - 1, b * VTA_OUT_WIDTH) =
                    (out_T) add_val.range(VTA_OUT_WIDTH - 1, 0);
                // Compute Shift Right
                acc_T shr_val =
                    src_0 >> (aluop_sh_imm_T) src_1.range(VTA_LOG_ACC_WIDTH - 1, 0);
                shr_res.range((b + 1) * VTA_ACC_WIDTH - 1, b * VTA_ACC_WIDTH) = shr_val;
                short_shr_res.range((b + 1) * VTA_OUT_WIDTH - 1, b * VTA_OUT_WIDTH) =
                    (out_T) shr_val.range(VTA_OUT_WIDTH-1, 0);
              }

              // Store to accum memory/store buffer
              if (alu_opcode == VTA_ALU_OPCODE_MIN ||
                  alu_opcode == VTA_ALU_OPCODE_MAX) {
                acc_mem[dst_idx][i] = cmp_res;
                out_mem[dst_idx][i] = short_cmp_res;
              } else if (alu_opcode == VTA_ALU_OPCODE_ADD) {
                acc_mem[dst_idx][i] = add_res;
                out_mem[dst_idx][i] = short_add_res;
              } else if (alu_opcode == VTA_ALU_OPCODE_SHR) {
                acc_mem[dst_idx][i] = shr_res;
                out_mem[dst_idx][i] = short_shr_res;
              }
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

void store(
  volatile out_vec_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH]
  ) {
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE axis port = g2s_dep_queue
#pragma HLS INTERFACE axis port = s2g_dep_queue
#pragma HLS INTERFACE bram port = out_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  // Load buffer
  insn_T insn = store_queue.read();

  // Decode
  bool pop_prev_dependence = insn[VTA_INSN_MEM_1];
  bool pop_next_dependence = insn[VTA_INSN_MEM_2];
  bool push_prev_dependence = insn[VTA_INSN_MEM_3];
  bool push_next_dependence = insn[VTA_INSN_MEM_4];
  memop_id_T memory_type = insn.range(VTA_INSN_MEM_5_1, VTA_INSN_MEM_5_0);
  memop_sram_T sram_base = insn.range(VTA_INSN_MEM_6_1, VTA_INSN_MEM_6_0);
  memop_dram_T dram_base = insn.range(VTA_INSN_MEM_7_1, VTA_INSN_MEM_7_0);
  memop_size_T y_size = insn.range(VTA_INSN_MEM_8_1, VTA_INSN_MEM_8_0);
  memop_size_T x_size = insn.range(VTA_INSN_MEM_9_1, VTA_INSN_MEM_9_0);
  memop_stride_T x_stride = insn.range(VTA_INSN_MEM_A_1, VTA_INSN_MEM_A_0);
  memop_pad_T y_pad_0 = insn.range(VTA_INSN_MEM_B_1, VTA_INSN_MEM_B_0);
  memop_pad_T y_pad_1 = insn.range(VTA_INSN_MEM_C_1, VTA_INSN_MEM_C_0);
  memop_pad_T x_pad_0 = insn.range(VTA_INSN_MEM_D_1, VTA_INSN_MEM_D_0);
  memop_pad_T x_pad_1 = insn.range(VTA_INSN_MEM_E_1, VTA_INSN_MEM_E_0);

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
    memcpy(
      const_cast<out_vec_T*>(&outputs[dram_idx*VTA_BATCH]),
      (const out_vec_T*) &out_mem[sram_idx][0],
      x_size * VTA_INP_ELEM_BYTES);
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
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile inp_vec_T *inputs,
  volatile wgt_vec_T *weights,
  volatile acc_vec_T *biases,
  volatile out_vec_T *outputs) {
#pragma HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE m_axi port = uops offset = slave bundle = uop_port
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = biases offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  // Instantiate temporary instruction queues (used for peeking)
  hls::stream<insn_T> tmp_load_queue;
  hls::stream<insn_T> tmp_gemm_queue;
  hls::stream<insn_T> tmp_store_queue;

  // Instatiate physical instruction queues
  hls::stream<insn_T> load_queue;
  hls::stream<insn_T> gemm_queue;
  hls::stream<insn_T> store_queue;

  // Dependence queues
  hls::stream<bool> l2g_dep_queue;
  hls::stream<bool> s2g_dep_queue;
  hls::stream<bool> g2l_dep_queue;
  hls::stream<bool> g2s_dep_queue;

  // Instantiate memories
  inp_vec_T inp_mem[VTA_INP_BUFF_DEPTH][VTA_BATCH];
  wgt_vec_T wgt_mem[VTA_WGT_BUFF_DEPTH][VTA_BLOCK_OUT];
  out_vec_T out_mem[VTA_ACC_BUFF_DEPTH][VTA_BATCH];

  // Push all instructions into the queues
  fetch(insn_count, insns, tmp_load_queue, tmp_gemm_queue, tmp_store_queue);

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
        tmp_load_queue.read(tmp_load);
        tmp_load_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_next_dependence = tmp_load[VTA_INSN_MEM_2];
      if ((pop_next_dependence && !g2l_dep_queue.empty()) ||
          !pop_next_dependence) {
        // Push the instruction in the load queue
        load_queue.write(tmp_load);
        tmp_load_popped = false;
        load(inputs, weights, load_queue, g2l_dep_queue, l2g_dep_queue, inp_mem, wgt_mem);
      } else {
        // Execution of load stage pending on completion of other stages, so break here...
        break;
      }
    }
    // Next execute as many gemm instructions as possible
    while (!tmp_gemm_queue.empty() || tmp_gemm_popped == true) {
      // Pop the gemm instruction
      if (!tmp_gemm_popped) {
        tmp_gemm_queue.read(tmp_gemv);
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
        compute(done, uops, biases, gemm_queue, l2g_dep_queue, s2g_dep_queue,
                g2l_dep_queue, g2s_dep_queue, inp_mem, wgt_mem, out_mem);
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
        tmp_store_queue.read(tmp_store);
        tmp_store_popped = true;
      }
      // Check dependences and invoke the load stage
      bool pop_prev_dependence = tmp_store[VTA_INSN_MEM_1];
      if ((pop_prev_dependence && !g2s_dep_queue.empty()) ||
          !pop_prev_dependence) {
        // Push the instruction in the load queue
        store_queue.write(tmp_store);
        tmp_store_popped = false;
        store(outputs, store_queue, g2s_dep_queue, s2g_dep_queue, out_mem);
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
  bool tmp_tok;
  int l2g_count = 0;
  int s2g_count = 0;
  int g2l_count = 0;
  int g2s_count = 0;
  while (l2g_dep_queue.read_nb(tmp_tok)) {
    l2g_count++;
  }
  while (s2g_dep_queue.read_nb(tmp_tok)) {
    s2g_count++;
  }
  while (g2l_dep_queue.read_nb(tmp_tok)) {
    g2l_count++;
  }
  while (g2s_dep_queue.read_nb(tmp_tok)) {
    g2s_count++;
  }

  assert(l2g_count == 0 && g2s_count == 0 && g2l_count == 0 && g2s_count == 0);
}
