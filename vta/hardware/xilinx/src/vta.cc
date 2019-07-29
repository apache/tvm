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
 * \file vta.cpp
 * \brief VTA HLS design.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vta.h"

template <typename DATA_T, int MAT_AXI_RATIO>
void reset_mem(
  memop_sram_T &sram_idx,
  memop_sram_T range,
  DATA_T mem[][MAT_AXI_RATIO]) {

  for (int i = 0; i < range; i ++) {
    for (int j = 0; j < MAT_AXI_RATIO; j ++) {
#pragma HLS UNROLL
      mem[sram_idx][j] = 0;
    }
    sram_idx ++;
  }
}

template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
void load_pad_2d(
  volatile DATA_T *src,
  DATA_T dst[][MAT_AXI_RATIO],
  memop_sram_T sram_idx,
  memop_dram_T dram_idx,
  memop_size_T y_size,
  memop_size_T x_size,
  memop_stride_T x_stride,
  memop_pad_T x_pad_0,
  memop_pad_T x_pad_1,
  memop_sram_T y_offset_0,
  memop_sram_T y_offset_1) {
#pragma HLS INLINE

  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_0, dst);
  for (int y = 0; y < y_size; y++) {
#pragma HLS PIPELINE
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_0, dst);
    memcpy(&dst[sram_idx][0],
           (const DATA_T*) &src[dram_idx * MAT_AXI_RATIO],
           x_size * ELEM_BYTES);
    sram_idx += x_size;
    dram_idx += x_stride;
    reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, x_pad_1, dst);
  }
  reset_mem<DATA_T, MAT_AXI_RATIO>(sram_idx, y_offset_1, dst);
}

template <typename DATA_T, int MAT_AXI_RATIO, int ELEM_BYTES>
void load_2d(
  volatile DATA_T *src,
  DATA_T dst[][MAT_AXI_RATIO],
  memop_sram_T sram_idx,
  memop_dram_T dram_idx,
  memop_size_T y_size,
  memop_size_T x_size,
  memop_stride_T x_stride) {
#pragma HLS INLINE

  for (int y = 0; y < y_size; y++) {
    memcpy(&dst[sram_idx][0],
           (const DATA_T*) &src[dram_idx * MAT_AXI_RATIO],
           x_size * ELEM_BYTES);
#pragma HLS RESOURCE variable = sram_idx core = Mul_LUT
    sram_idx += x_size;
    dram_idx += x_stride;
  }
}

template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
void read_tensor(
  IDX_T idx,
  WIDE_T src[][NARROW_W * Y_DIM * X_DIM / WIDE_W],
  NARROW_T dst[Y_DIM][X_DIM]) {
#pragma HLS INLINE

  // Read in weight tensor
  for (int p = 0; p < NARROW_W * Y_DIM * X_DIM / WIDE_W; p++) {
    WIDE_T packet = src[idx][p];
    for (int w = 0; w < (WIDE_W / NARROW_W); w++) {
      int x = (p * (WIDE_W / NARROW_W) + w) / X_DIM;
      int y = (p * (WIDE_W / NARROW_W) + w) % X_DIM;
      dst[x][y] = (NARROW_T) packet.range((w + 1) * NARROW_W - 1, w * NARROW_W);
    }
  }
}

template <typename WIDE_T, typename NARROW_T, typename IDX_T, int WIDE_W, int NARROW_W, int Y_DIM, int X_DIM>
void write_tensor(
  IDX_T idx,
  NARROW_T src[Y_DIM][X_DIM],
  WIDE_T dst[][NARROW_W * Y_DIM * X_DIM / WIDE_W]) {
#pragma HLS INLINE

  for (int p = 0; p < NARROW_W * Y_DIM * X_DIM / WIDE_W; p++) {
    WIDE_T packet = 0;
    for (int w = 0; w < (WIDE_W / NARROW_W); w++) {
      int x = (p * (WIDE_W / NARROW_W) + w) / X_DIM;
      int y = (p * (WIDE_W / NARROW_W) + w) % X_DIM;
      packet.range((w + 1) * NARROW_W - 1, w * NARROW_W) = src[x][y];
    }
    dst[idx][p] = packet;
  }
}

void fetch(
  uint32_t insn_count,
  volatile insn_T *insns,
  hls::stream<insn_T> &load_queue,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<insn_T> &store_queue) {
PRAGMA_HLS(HLS INTERFACE s_axilite port = insn_count bundle = CONTROL_BUS offset = VTA_FETCH_INSN_COUNT_OFFSET)
#pragma HLS INTERFACE m_axi port = insns offset = slave bundle = ins_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = gemm_queue
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS

  INSN_DECODE: for (int pc = 0; pc < insn_count; pc++) {
#pragma HLS PIPELINE
    // Read instruction fields
    insn_T raw_insn = insns[pc];
    VTAInsn insn;
    insn.generic = *((VTAGenericInsn *) &raw_insn);
    // Do some partial decoding
    opcode_T opcode = insn.generic.opcode;
    memop_id_T memory_type = insn.mem.memory_type;
    // Push to appropriate instruction queue
    if (opcode == VTA_OPCODE_STORE) {
      store_queue.write(raw_insn);
    } else if (opcode == VTA_OPCODE_LOAD) {
      if (memory_type == VTA_MEM_ID_INP || memory_type == VTA_MEM_ID_WGT) {
        load_queue.write(raw_insn);
      } else {
        gemm_queue.write(raw_insn);
      }
    } else {
      gemm_queue.write(raw_insn);
    }
  }
}

void load(
  volatile bus_T *inputs,
  volatile bus_T *weights,
  hls::stream<insn_T> &load_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &l2g_dep_queue,
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO]) {
#pragma HLS INTERFACE m_axi port = inputs offset = slave bundle = data_port
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = load_queue
#pragma HLS INTERFACE axis port = g2l_dep_queue
#pragma HLS INTERFACE axis port = l2g_dep_queue
#pragma HLS INTERFACE bram port = wgt_mem
#pragma HLS INTERFACE bram port = inp_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS RESOURCE variable = inp_mem core = RAM_1P
#pragma HLS RESOURCE variable = wgt_mem core = RAM_1P

  // Pop load instruction
  insn_T raw_insn = load_queue.read();
  // Cast to MemInsn
  insn_T raw_copy = raw_insn;
  VTAMemInsn insn = *((VTAMemInsn *) &raw_copy);

  // Pop dependence token if instructed
  if (insn.pop_next_dep) {
    g2l_dep_queue.read();
  }

  // Pre-processing
  memop_sram_T x_width = (insn.x_pad_0 + insn.x_size + insn.x_pad_1);
  memop_sram_T y_offset_0 = x_width * insn.y_pad_0;
#pragma HLS RESOURCE variable = y_offset_0 core = Mul_LUT latency = 4
  memop_sram_T y_offset_1 = x_width * insn.y_pad_1;
#pragma HLS RESOURCE variable = y_offset_1 core = Mul_LUT latency = 4

  if (insn.memory_type == VTA_MEM_ID_INP) {
    load_pad_2d<bus_T, INP_MAT_AXI_RATIO, VTA_INP_ELEM_BYTES>(
        inputs,
        inp_mem,
        insn.sram_base,
        insn.dram_base,
        insn.y_size,
        insn.x_size,
        insn.x_stride,
        insn.x_pad_0,
        insn.x_pad_1,
        y_offset_0,
        y_offset_1);
  } else if (insn.memory_type == VTA_MEM_ID_WGT) {
    load_2d<bus_T, WGT_MAT_AXI_RATIO, VTA_WGT_ELEM_BYTES>(
        weights,
        wgt_mem,
        insn.sram_base,
        insn.dram_base,
        insn.y_size,
        insn.x_size,
        insn.x_stride);
  }

  // Push dependence token if instructed
  if (insn.push_next_dep) {
    l2g_dep_queue.write(1);
  }
}

void gemm(
  insn_T insn_raw,
  uop_T uop_mem[VTA_UOP_BUFF_DEPTH],
  bus_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO],
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]) {
#pragma HLS INLINE

  VTAGemInsn insn = *((VTAGemInsn *) &insn_raw);

  // Loop offset
  acc_idx_T dst_offset_out = 0;
  inp_idx_T src_offset_out = 0;
  wgt_idx_T wgt_offset_out = 0;

  // Outer Loop
  EXE_OUT_LOOP: for (int it_out = 0; it_out < insn.iter_out; it_out++) {
    acc_idx_T dst_offset_in = dst_offset_out;
    inp_idx_T src_offset_in = src_offset_out;
    wgt_idx_T wgt_offset_in = wgt_offset_out;

    // Inner Loop
    EXE_IN_LOOP: for (int it_in = 0; it_in < insn.iter_in; it_in++) {

      // Iterate over micro op
      READ_GEMM_UOP: for (int upc = insn.uop_bgn; upc < insn.uop_end; upc++) {
#pragma HLS PIPELINE II = 1
        // Read micro-op fields
        uop_T uop = uop_mem[upc];

        // Decode indices
        acc_idx_T dst_idx =
            uop.range(VTA_UOP_GEM_0_1, VTA_UOP_GEM_0_0) + dst_offset_in;
        inp_idx_T src_idx =
            uop.range(VTA_UOP_GEM_1_1, VTA_UOP_GEM_1_0) + src_offset_in;
        wgt_idx_T wgt_idx =
            uop.range(VTA_UOP_GEM_2_1, VTA_UOP_GEM_2_0) + wgt_offset_in;

        // Read in weight tensor
        wgt_T w_tensor[VTA_BLOCK_OUT][VTA_BLOCK_IN];
        read_tensor<bus_T, wgt_T, wgt_idx_T, VTA_BUS_WIDTH, VTA_WGT_WIDTH, VTA_BLOCK_OUT, VTA_BLOCK_IN>(wgt_idx, wgt_mem, w_tensor);
        // Read in input tensor
        inp_T i_tensor[VTA_BATCH][VTA_BLOCK_IN];
        read_tensor<bus_T, inp_T, inp_idx_T, VTA_BUS_WIDTH, VTA_INP_WIDTH, VTA_BATCH, VTA_BLOCK_IN>(src_idx, inp_mem, i_tensor);
        // Read in accum tensor
        acc_T a_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, acc_mem, a_tensor);
        // Output tensor
        out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

        // Inner GEMM loop
        for (int b = 0; b < VTA_BATCH; b++) {
          for (int oc = 0; oc < VTA_BLOCK_OUT; oc++) {
            // Initialize the accumulator values
            acc_T accum = a_tensor[b][oc];
            // Dot product sum
            sum_T tmp = 0;
            // Inner matrix multiplication loop (input channel/feature)
            for (int ic = 0; ic < VTA_BLOCK_IN; ic++) {
              wgt_T w_elem = w_tensor[oc][ic];
              inp_T i_elem = i_tensor[b][ic];
              mul_T prod_dsp = i_elem * w_elem;
              tmp += (sum_T) prod_dsp;
            }
            // Update summation
            accum += (acc_T) tmp;
            // Write back result acc_mem
            a_tensor[b][oc] = insn.reset_reg ? (acc_T) 0 : accum;
            // And output vector
            o_tensor[b][oc] = (out_T) accum.range(VTA_OUT_WIDTH - 1, 0);
          }
        }

        // Write the results back into accumulator
        write_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, a_tensor, acc_mem);
        // Write the results back in the output buffer
        write_tensor<bus_T, out_T, acc_idx_T, VTA_BUS_WIDTH, VTA_OUT_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, o_tensor, out_mem);
      }
      // Update offsets
      dst_offset_in += insn.dst_factor_in;
      src_offset_in += insn.src_factor_in;
      wgt_offset_in += insn.wgt_factor_in;
    }
    // Update offsets
    dst_offset_out += insn.dst_factor_out;
    src_offset_out += insn.src_factor_out;
    wgt_offset_out += insn.wgt_factor_out;
  }
}

void alu(
  insn_T insn_raw,
  uop_T uop_mem[VTA_UOP_BUFF_DEPTH],
  bus_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO],
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]) {
#pragma HLS INLINE

  VTAAluInsn insn = *((VTAAluInsn *) &insn_raw);

  // Loop offset
  acc_idx_T dst_offset_out = 0;
  inp_idx_T src_offset_out = 0;

  // Outer Loop
  EXE_OUT_LOOP: for (int it_out = 0; it_out < insn.iter_out; it_out++) {
    acc_idx_T dst_offset_in = dst_offset_out;
    inp_idx_T src_offset_in = src_offset_out;

    // Inner Loop
    EXE_IN_LOOP: for (int it_in = 0; it_in < insn.iter_in; it_in++) {
      // Iterate over micro op
      READ_ALU_UOP: for (int upc = insn.uop_bgn; upc < insn.uop_end; upc++) {
#pragma HLS PIPELINE II = 2
        // Read micro-op fields
        uop_T uop = uop_mem[upc];

        // Decode
        acc_idx_T dst_idx =
            uop.range(VTA_UOP_ALU_0_1, VTA_UOP_ALU_0_0) + dst_offset_in;
        acc_idx_T src_idx =
            uop.range(VTA_UOP_ALU_1_1, VTA_UOP_ALU_1_0) + src_offset_in;

        // Read in src tensor
        acc_T src_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(src_idx, acc_mem, src_tensor);
        // Read in dst tensor
        acc_T dst_tensor[VTA_BATCH][VTA_BLOCK_OUT];
        read_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, acc_mem, dst_tensor);
        // Output tensor
        out_T o_tensor[VTA_BATCH][VTA_BLOCK_OUT];

        // Perform ALU op over matrix elements
        for (int i = 0; i < VTA_BATCH; i++) {
          for (int b = 0; b < VTA_BLOCK_OUT; b++) {
            // Read in operands
            acc_T src_0 = dst_tensor[i][b];
            acc_T src_1 = insn.use_imm ? (acc_T) insn.imm : src_tensor[i][b];
            aluop_shr_arg_T shft_by = src_1.range(VTA_SHR_ARG_BIT_WIDTH - 1, 0);
            aluop_mul_arg_T mul_by = src_1.range(VTA_MUL_ARG_BIT_WIDTH - 1, 0);
            if (insn.alu_opcode == VTA_ALU_OPCODE_MIN || insn.alu_opcode == VTA_ALU_OPCODE_MAX) {
              // Compute Min/Max
              acc_T mix_val = src_0 < src_1 ?
                  (insn.alu_opcode == VTA_ALU_OPCODE_MIN ? src_0 : src_1) :
                  (insn.alu_opcode == VTA_ALU_OPCODE_MIN ? src_1 : src_0);
              dst_tensor[i][b] = mix_val;
              o_tensor[i][b] = (out_T) mix_val.range(VTA_OUT_WIDTH - 1, 0);
            } else if (insn.alu_opcode == VTA_ALU_OPCODE_ADD) {
              // Compute Sum
              acc_T add_val =
                  src_0.range(VTA_ACC_WIDTH - 1, 0) + src_1.range(VTA_ACC_WIDTH - 1, 0);
              dst_tensor[i][b] = add_val;
              o_tensor[i][b] = (out_T) add_val.range(VTA_OUT_WIDTH - 1, 0);
            } else if (insn.alu_opcode == VTA_ALU_OPCODE_SHR) {
              // Compute Shift Right
              acc_T shr_val = src_0 >> shft_by;
              dst_tensor[i][b] = shr_val;
              o_tensor[i][b] = (out_T) shr_val.range(VTA_OUT_WIDTH - 1, 0);
            }
          }
        }

        // Write the results back into accumulator
        write_tensor<bus_T, acc_T, acc_idx_T, VTA_BUS_WIDTH, VTA_ACC_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, dst_tensor, acc_mem);
        // Write the results back in the output buffer
        write_tensor<bus_T, out_T, acc_idx_T, VTA_BUS_WIDTH, VTA_OUT_WIDTH, VTA_BATCH, VTA_BLOCK_OUT>(dst_idx, o_tensor, out_mem);
      }
      // Update offsets
      dst_offset_in += insn.dst_factor_in;
      src_offset_in += insn.src_factor_in;
    }
    // Update offsets
    dst_offset_out += insn.dst_factor_out;
    src_offset_out += insn.src_factor_out;
  }
}

void compute(
  volatile uint32_t &done,
  volatile uop_T *uops,
  volatile bus_T *biases,
  hls::stream<insn_T> &gemm_queue,
  hls::stream<bool> &l2g_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  hls::stream<bool> &g2l_dep_queue,
  hls::stream<bool> &g2s_dep_queue,
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO],
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO],
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]) {
PRAGMA_HLS(HLS INTERFACE s_axilite port = done bundle = CONTROL_BUS offset = VTA_COMPUTE_DONE_WR_OFFSET)
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
#pragma HLS RESOURCE variable = inp_mem core = RAM_1P
#pragma HLS RESOURCE variable = wgt_mem core = RAM_1P
#pragma HLS RESOURCE variable = out_mem core = RAM_1P

  // Micro-op storage
  static uop_T uop_mem[VTA_UOP_BUFF_DEPTH];

  // Accumulator storage
  static bus_T acc_mem[VTA_ACC_BUFF_DEPTH][ACC_MAT_AXI_RATIO];
#pragma HLS ARRAY_RESHAPE variable = acc_mem complete dim=2
// This is necessary to obtain II=1
#pragma HLS DEPENDENCE variable = acc_mem inter false

  // Pop GEMM instruction
  insn_T raw_insn = gemm_queue.read();
  // Cast to GenericInsn
  VTAInsn insn;
  insn_T raw_copy = raw_insn;
  insn.generic = *((VTAGenericInsn *) &raw_copy);

  // Pop dependence token if instructed
  if (insn.generic.pop_prev_dep) {
    l2g_dep_queue.read();
  }
  if (insn.generic.pop_next_dep) {
    s2g_dep_queue.read();
  }

  // Set done value
  done = 0;
  // Perform action based on opcode
  if (insn.generic.opcode == VTA_OPCODE_FINISH) {
    // Set done flag if we reach a FINISH instruction
    done = 1;
  } else if (insn.generic.opcode == VTA_OPCODE_LOAD) {
    // Initialize indices
    memop_sram_T sram_idx = insn.mem.sram_base;
    memop_dram_T dram_idx = insn.mem.dram_base;
    if (insn.mem.memory_type == VTA_MEM_ID_UOP) {
      // Perform data transfer
      memcpy(&uop_mem[sram_idx],
             (const uop_T*) &uops[dram_idx],
             insn.mem.x_size * sizeof(uop_T));
    } else if (insn.mem.memory_type == VTA_MEM_ID_ACC) {
      // Perform data transfer from DRAM
      load_2d<bus_T, ACC_MAT_AXI_RATIO, VTA_ACC_ELEM_BYTES>(
          biases,
          acc_mem,
          sram_idx,
          dram_idx,
          insn.mem.y_size,
          insn.mem.x_size,
          insn.mem.x_stride);
    }
  } else if (insn.generic.opcode == VTA_OPCODE_GEMM) {
    gemm(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
  } else if (insn.generic.opcode == VTA_OPCODE_ALU) {
    alu(raw_copy, uop_mem, acc_mem, inp_mem, wgt_mem, out_mem);
  }

  // Push dependence token if instructed
  if (insn.generic.push_prev_dep) {
    g2l_dep_queue.write(1);
  }
  if (insn.generic.push_next_dep) {
    g2s_dep_queue.write(1);
  }
}

void store(
  volatile bus_T *outputs,
  hls::stream<insn_T> &store_queue,
  hls::stream<bool> &g2s_dep_queue,
  hls::stream<bool> &s2g_dep_queue,
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO]) {
#pragma HLS INTERFACE m_axi port = outputs offset = slave bundle = data_port
#pragma HLS INTERFACE axis port = store_queue
#pragma HLS INTERFACE axis port = g2s_dep_queue
#pragma HLS INTERFACE axis port = s2g_dep_queue
#pragma HLS INTERFACE bram port = out_mem
#pragma HLS INTERFACE s_axilite port = return bundle = CONTROL_BUS
#pragma HLS RESOURCE variable = out_mem core = RAM_1P

  // Pop store instruction
  insn_T raw_insn = store_queue.read();
  // Cast to MemInsn
  insn_T raw_copy = raw_insn;
  VTAMemInsn insn = *((VTAMemInsn *) &raw_copy);

  // Pop dependence token if instructed
  if (insn.pop_prev_dep) {
    g2s_dep_queue.read();
  }

  // Initialize indices
  memop_sram_T sram_idx = insn.sram_base;
  memop_dram_T dram_idx = insn.dram_base;

  // Copy along y dimension
  for (int y = 0; y < insn.y_size; y++) {
#pragma HLS PIPELINE
    // Perform data transfer
    memcpy(
      const_cast<bus_T*>(&outputs[dram_idx * OUT_MAT_AXI_RATIO]),
      (const bus_T*) &out_mem[sram_idx][0],
      insn.x_size * VTA_OUT_ELEM_BYTES);
#pragma HLS RESOURCE variable = sram_idx core = Mul_LUT
    sram_idx += insn.x_size;
    dram_idx += insn.x_stride;
  }

  // Push dependence token if instructed
  if (insn.push_prev_dep) {
    s2g_dep_queue.write(1);
  }
}

void vta(
  uint32_t insn_count,
  volatile insn_T *insns,
  volatile uop_T *uops,
  volatile bus_T *inputs,
  volatile bus_T *weights,
  volatile bus_T *biases,
  volatile bus_T *outputs) {
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
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_load_queue)
  hls::stream<insn_T> tmp_gemm_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_gemm_queue)
  hls::stream<insn_T> tmp_store_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=tmp_store_queue)

  // Instatiate physical instruction queues
  hls::stream<insn_T> load_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=load_queue)
  hls::stream<insn_T> gemm_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=gemm_queue)
  hls::stream<insn_T> store_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=store_queue)

  // Dependence queues
  hls::stream<bool> l2g_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=l2g_dep_queue)
  hls::stream<bool> s2g_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=s2g_dep_queue)
  hls::stream<bool> g2l_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=g2l_dep_queue)
  hls::stream<bool> g2s_dep_queue;
  PRAGMA_HLS(HLS stream depth=STREAM_IN_DEPTH variable=g2s_dep_queue)

  // Instantiate memories
  bus_T inp_mem[VTA_INP_BUFF_DEPTH][INP_MAT_AXI_RATIO];
  bus_T wgt_mem[VTA_WGT_BUFF_DEPTH][WGT_MAT_AXI_RATIO];
  bus_T out_mem[VTA_ACC_BUFF_DEPTH][OUT_MAT_AXI_RATIO];

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
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_load);
      if ((insn.pop_next_dep && !g2l_dep_queue.empty()) ||
          !insn.pop_next_dep) {
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
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_gemv);
      if (
        (insn.pop_prev_dep && !l2g_dep_queue.empty() &&
         insn.pop_next_dep && !s2g_dep_queue.empty()) ||
        (!insn.pop_prev_dep && insn.pop_next_dep &&
         !s2g_dep_queue.empty()) ||
        (insn.pop_prev_dep && !l2g_dep_queue.empty() &&
        !insn.pop_next_dep) ||
        (!insn.pop_prev_dep && !insn.pop_next_dep)
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
      VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_store);

      if ((insn.pop_prev_dep && !g2s_dep_queue.empty()) ||
          !insn.pop_prev_dep) {
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
        VTAGenericInsn insn = *((VTAGenericInsn *) &tmp_gemv);
        if (l2g_dep_queue.empty() && insn.pop_prev_dep) {
          printf("waiting on l2g\n");
        }
        if (s2g_dep_queue.empty() && insn.pop_next_dep) {
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
