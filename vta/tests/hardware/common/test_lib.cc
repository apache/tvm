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
 *  Copyright (c) 2018 by Contributors
 * \file test_lib.cpp
 * \brief Test library for the VTA design simulation and driver tests.
 */

#include "test_lib.h"

#ifdef NO_SIM
#ifdef VTA_TARGET_PYNQ

uint64_t vta(
  uint32_t insn_count,
  VTAGenericInsn *insns,
  VTAUop *uops,
  inp_T *inputs,
  wgt_T *weights,
  acc_T *biases,
  inp_T *outputs) {
  // Performance counter variables
  uint64_t t_fpga;
  struct timespec start, stop;

  // Derive bitstream file
  char bitstream[128];
  char str_batch_size[4];
  char str_block_out_size[4];
  char str_block_in_size[4];
  char str_block_bit_width[4];
  snprintf(str_batch_size, sizeof(str_batch_size), "%d", VTA_BATCH);
  snprintf(str_block_out_size, sizeof(str_block_out_size), "%d", VTA_BLOCK_OUT);
  snprintf(str_block_in_size, sizeof(str_block_in_size), "%d", VTA_BLOCK_IN);
  snprintf(str_block_bit_width, sizeof(str_block_bit_width), "%d", VTA_WGT_WIDTH);
  snprintf(bitstream, sizeof(bitstream), "%s", "vta.bit");

  // Get VTA handles
  void* vta_fetch_handle = VTAMapRegister(VTA_FETCH_ADDR, VTA_RANGE);
  void* vta_load_handle = VTAMapRegister(VTA_LOAD_ADDR, VTA_RANGE);
  void* vta_compute_handle = VTAMapRegister(VTA_COMPUTE_ADDR, VTA_RANGE);
  void* vta_store_handle = VTAMapRegister(VTA_STORE_ADDR, VTA_RANGE);

  // Physical address pointers
  uint32_t insn_phy = insns ? VTAMemGetPhyAddr(insns) : 0;
  uint32_t uop_phy = uops ? VTAMemGetPhyAddr(uops) : 0;
  uint32_t input_phy = inputs ? VTAMemGetPhyAddr(inputs) : 0;
  uint32_t weight_phy = weights ? VTAMemGetPhyAddr(weights) : 0;
  uint32_t bias_phy = biases ? VTAMemGetPhyAddr(biases) : 0;
  uint32_t output_phy = outputs ? VTAMemGetPhyAddr(outputs) : 0;

#if VTA_DEBUG == 1
  printf("INFO - Starting FPGA!\n");
#endif

  clock_gettime(CLOCK_REALTIME, &start);

  // FETCH @ 0x10 : Data signal of insn_count_V
  VTAWriteMappedReg(vta_fetch_handle, 0x10, insn_count);
  // FETCH @ 0x18 : Data signal of insns_V
  if (insns) VTAWriteMappedReg(vta_fetch_handle, 0x18, insn_phy);
  // LOAD @ 0x10 : Data signal of inputs_V
  if (inputs) VTAWriteMappedReg(vta_load_handle, 0x10, input_phy);
  // LOAD @ 0x18 : Data signal of weight_V
  if (weights) VTAWriteMappedReg(vta_load_handle, 0x18, weight_phy);
  // COMPUTE @ 0x20 : Data signal of uops_V
  if (uops) VTAWriteMappedReg(vta_compute_handle, 0x20, uop_phy);
  // COMPUTE @ 0x28 : Data signal of biases_V
  if (biases) VTAWriteMappedReg(vta_compute_handle, 0x28, bias_phy);
  // STORE @ 0x10 : Data signal of outputs_V
  if (outputs) VTAWriteMappedReg(vta_store_handle, 0x10, output_phy);

  // VTA start
  VTAWriteMappedReg(vta_fetch_handle, 0x0, 0x1);
  VTAWriteMappedReg(vta_load_handle, 0x0, 0x81);
  VTAWriteMappedReg(vta_compute_handle, 0x0, 0x81);
  VTAWriteMappedReg(vta_store_handle, 0x0, 0x81);

  int flag = 0, t = 0;
  for (t = 0; t < 10000000; ++t) {
    flag = VTAReadMappedReg(vta_compute_handle, 0x18);
    if (flag & VTA_DONE) break;
  }

  if (t == 10000000) {
    printf("\tWARNING: VTA TIMEOUT!!!!\n");
#if VTA_DEBUG == 1
  } else {
    printf("INFO - FPGA Finished!\n");
#endif
  }

  clock_gettime(CLOCK_REALTIME, &stop);
  t_fpga = 1000000000ULL * (stop.tv_sec - start.tv_sec) + (stop.tv_nsec - start.tv_nsec);

  // Unmap VTA register
  VTAUnmapRegister(vta_fetch_handle, VTA_RANGE);
  VTAUnmapRegister(vta_load_handle, VTA_RANGE);
  VTAUnmapRegister(vta_compute_handle, VTA_RANGE);
  VTAUnmapRegister(vta_store_handle, VTA_RANGE);

  return t_fpga;
}

#endif  // VTA_TARGET_PYNQ
#endif  // NO_SIM

uint32_t globalSeed;

const char* getOpcodeString(int opcode, bool use_imm) {
  // Returns string name
  if (opcode == VTA_ALU_OPCODE_MIN) {
    if (use_imm) {
      return "min imm";
    } else {
      return "min";
    }
  } else if (opcode == VTA_ALU_OPCODE_MAX) {
    if (use_imm) {
      return "max imm";
    } else {
      return "max";
    }
  } else if (opcode == VTA_ALU_OPCODE_ADD) {
    if (use_imm) {
      return "add imm";
    } else {
      return "add";
    }
  } else if (opcode == VTA_ALU_OPCODE_SHR) {
    return "shr";
  }
  return "unknown op";
}

template <typename T, int T_WIDTH>
void packBuffer(T *dst, T **src, int y_size, int x_size, int y_block, int x_block) {
  int buffer_idx = 0;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        if (T_WIDTH < 8) {
          for (int l = 0; l < x_block; l += 8 / T_WIDTH) {
            dst[buffer_idx] = 0;
            for (int m = 0; m < 8 / T_WIDTH; m++) {
              dst[buffer_idx] |= (src[i * y_block + k][j * x_block + l + m] &
                ((1ULL << T_WIDTH) - 1)) << (m * T_WIDTH);
            }
            buffer_idx++;
          }
        } else {
          for (int l = 0; l < x_block; l++) {
            dst[buffer_idx++] = src[i * y_block + k][j * x_block + l];
          }
        }
      }
    }
  }
}

template <typename T, int T_WIDTH>
void unpackBuffer(T **dst, T *src, int y_size, int x_size, int y_block, int x_block) {
  int buffer_idx = 0;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        if (T_WIDTH < 8) {
          for (int l = 0; l < x_block; l += 8 / T_WIDTH) {
            for (int m = 0; m < 8 / T_WIDTH; m++) {
              dst[i * y_block + k][j * x_block + l + m] = (src[buffer_idx] >> (m * T_WIDTH))
                & ((1 << T_WIDTH) - 1);
            }
            buffer_idx++;
          }
        } else {
          for (int l = 0; l < x_block; l++) {
            dst[i * y_block + k][j * x_block + l] = src[buffer_idx++];
          }
        }
      }
    }
  }
}

template <typename T, int T_WIDTH>
T ** allocInit2dArray(int rows, int cols) {
  // Allocate
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  // Init
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      array[i][j] =
          static_cast<T>(rand_r(&globalSeed) % (1LL << (T_WIDTH - 1)) - (1LL << (T_WIDTH - 2)));
    }
  }
  return array;
}

template <typename T>
T ** alloc2dArray(int rows, int cols) {
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  return array;
}

template <typename T>
void free2dArray(T **array, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    free(array[i]);
  }
  free(array);
}

template <typename T>
T *** alloc3dArray(int rows, int cols, int depth) {
  T ***array = static_cast<T ***>(malloc(sizeof(T **) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T **>(malloc(sizeof(T *) * cols));
    for (int j = 0; j < cols; j++) {
      array[i][j] = static_cast<T*>(malloc(sizeof(T) * depth));
    }
  }
  return array;
}

template <typename T>
void free3dArray(T *** array, int rows, int cols, int depth) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

void * allocBuffer(size_t num_bytes) {
#ifdef NO_SIM
  return VTAMemAlloc(num_bytes, VTA_CACHED);
#else
  return malloc(num_bytes);
#endif
}

void freeBuffer(void * buffer) {
#ifdef NO_SIM
  return VTAMemFree(buffer);
#else
  return free(buffer);
#endif
}

VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
    int y_size, int x_size, int x_stride, int y_pad, int x_pad, int pop_prev_dep, int pop_next_dep,
    int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = y_size;
  insn.x_size = x_size;
  insn.x_stride = x_stride;
  insn.y_pad_0 = y_pad;
  insn.y_pad_1 = y_pad;
  insn.x_pad_0 = x_pad;
  insn.x_pad_1 = x_pad;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
    int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = 1;
  insn.x_size = size;
  insn.x_stride = size;
  insn.y_pad_0 = 0;
  insn.y_pad_1 = 0;
  insn.x_pad_0 = 0;
  insn.x_pad_1 = 0;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
    bool uop_compression, int pop_prev_dep, int pop_next_dep, int push_prev_dep,
    int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_GEMM;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.reset_reg = false;
  if (!uop_compression) {
    insn.uop_bgn = uop_offset;
    insn.uop_end = uop_offset + batch * in_feat * out_feat;
    insn.iter_out = 1;
    insn.iter_in = 1;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 0;
    insn.wgt_factor_out = 0;
    insn.dst_factor_in = 0;
    insn.src_factor_in = 0;
    insn.wgt_factor_in = 0;
  } else {
    insn.uop_bgn = uop_offset;
    insn.uop_end = uop_offset + batch;
    insn.iter_out = in_feat;
    insn.iter_in = out_feat;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 1;
    insn.wgt_factor_out = 1;
    insn.dst_factor_in = 1;
    insn.src_factor_in = 0;
    insn.wgt_factor_in = in_feat;
  }
  converter.gemm = insn;
  return converter.generic;
}

VTAGenericInsn getALUInsn(int opcode, int vector_size, bool use_imm, int imm, bool uop_compression,
    int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAAluInsn insn = {};
  insn.opcode = VTA_OPCODE_ALU;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.reset_reg = false;
  if (!uop_compression) {
    insn.uop_bgn = 0;
    insn.uop_end = vector_size;
    insn.iter_out = 1;
    insn.iter_in = 1;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 0;
    insn.dst_factor_in = 0;
    insn.src_factor_in = 0;
    insn.alu_opcode = opcode;
    insn.use_imm = use_imm;
    insn.imm = imm;
  } else {
    insn.uop_bgn = 0;
    insn.uop_end = 1;
    insn.iter_out = 1;
    insn.iter_in = vector_size;
    insn.dst_factor_out = 0;
    insn.src_factor_out = 0;
    insn.dst_factor_in = 1;
    insn.src_factor_in = 1;
    insn.alu_opcode = opcode;
    insn.use_imm = use_imm;
    insn.imm = imm;
  }
  converter.alu = insn;
  return converter.generic;
}

VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_FINISH;
  insn.pop_prev_dep = pop_prev;
  insn.pop_next_dep = pop_next;
  insn.push_prev_dep = 0;
  insn.push_next_dep = 0;
  insn.reset_reg = false;
  insn.uop_bgn = 0;
  insn.uop_end = 0;
  insn.iter_out = 0;
  insn.iter_in = 0;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}

VTAUop * getCopyUops(int y_size, int x_size, int uop_compression) {
  // Derive the total uop size
  int uop_size = (uop_compression) ? 1 : y_size * x_size;

  // Allocate buffer
#ifdef NO_SIM
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, VTA_CACHED));
#else
  VTAUop *uop_buf = static_cast<VTAUop *>(malloc(sizeof(VTAUop) * uop_size));
#endif

  if (!uop_compression) {
    int uop_idx = 0;
    for (int i = 0; i < y_size; i++) {
      for (int j = 0; j < x_size; j++) {
        uop_buf[uop_idx].dst_idx = i * x_size + j;
        uop_buf[uop_idx].src_idx = 0;
        uop_buf[uop_idx].wgt_idx = 0;
        uop_idx++;
      }
    }
  } else {
    uop_buf[0].dst_idx = 1;
    uop_buf[0].src_idx = 0;
    uop_buf[0].wgt_idx = 0;
  }

  return uop_buf;
}

VTAUop * getGEMMUops(int batch, int in_feat, int out_feat, bool uop_compression,
    bool multi_threaded) {
  // Derive the total uop size
  int uop_size = (uop_compression) ? batch : batch * in_feat * out_feat;
  if (multi_threaded) uop_size *= 2;

  // Allocate buffer
#ifdef NO_SIM
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, VTA_CACHED));
#else
  VTAUop *uop_buf = static_cast<VTAUop *>(malloc(sizeof(VTAUop) * uop_size));
#endif

  if (!uop_compression) {
    int uop_idx = 0;
    for (int i = 0; i < batch; i++) {
      for (int j = 0; j < in_feat; j++) {
        for (int k = 0; k < out_feat; k++) {
          uop_buf[uop_idx].dst_idx = i * out_feat + k;
          uop_buf[uop_idx].src_idx = i * in_feat + j;
          uop_buf[uop_idx].wgt_idx = k * in_feat + j;
          uop_idx++;
        }
      }
    }
  } else {
    for (int i = 0; i < batch; i++) {
      uop_buf[i].dst_idx = i * out_feat;
      uop_buf[i].src_idx = i * in_feat;
      uop_buf[i].wgt_idx = 0;
    }
  }

  if (multi_threaded) {
    if (!uop_compression) {
      int uop_idx = uop_size / 2;
      for (int i = 0; i < batch; i++) {
        for (int j = 0; j < in_feat; j++) {
          for (int k = 0; k < out_feat; k++) {
            uop_buf[uop_idx].dst_idx = i * out_feat + k;
            uop_buf[uop_idx].src_idx = batch * in_feat + i * in_feat + j;
            uop_buf[uop_idx].wgt_idx = out_feat * in_feat + k * in_feat + j;
            uop_idx++;
          }
        }
      }
    } else {
      for (int i = 0; i < batch; i++) {
        uop_buf[batch+i].dst_idx = i * out_feat;
        uop_buf[batch+i].src_idx = batch * in_feat + i * in_feat;
        uop_buf[batch+i].wgt_idx = out_feat * in_feat;
      }
    }
  }

  return uop_buf;
}

VTAUop * getMapALUUops(int vector_size, bool uop_compression) {
  // Derive the total uop size
  int uop_size = (uop_compression) ? 1 : vector_size;

  // Allocate buffer
#ifdef NO_SIM
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, VTA_CACHED));
#else
  VTAUop *uop_buf = static_cast<VTAUop *>(malloc(sizeof(VTAUop) * uop_size));
#endif

  if (!uop_compression) {
    for (int i = 0; i < vector_size; i++) {
      uop_buf[i].dst_idx = i;
      uop_buf[i].src_idx = vector_size + i;
    }
  } else {
    uop_buf[0].dst_idx = 0;
    uop_buf[0].src_idx = vector_size;
  }

  return uop_buf;
}

void printParameters() {
  // Some debugging code
  printf("Size of VTAInsn: %d\n", sizeof(VTAGenericInsn));
  printf("Size of VTAUop: %d\n", sizeof(VTAUop));
  printf("VTA_UOP_BUFF_DEPTH: %d\n", VTA_UOP_BUFF_DEPTH);
  printf("VTA_LOG_UOP_BUFF_DEPTH: %d\n", VTA_LOG_UOP_BUFF_DEPTH);
  printf("VTA_WGT_BUFF_DEPTH: %d\n", VTA_WGT_BUFF_DEPTH);
  printf("VTA_LOG_WGT_BUFF_DEPTH: %d\n", VTA_LOG_WGT_BUFF_DEPTH);
  printf("VTA_INP_BUFF_DEPTH: %d\n", VTA_INP_BUFF_DEPTH);
  printf("VTA_LOG_INP_BUFF_DEPTH: %d\n", VTA_LOG_INP_BUFF_DEPTH);
  printf("VTA_ACC_BUFF_DEPTH: %d\n", VTA_ACC_BUFF_DEPTH);
  printf("VTA_LOG_ACC_BUFF_DEPTH: %d\n", VTA_LOG_ACC_BUFF_DEPTH);
  printf("VTA_WGT_WORDS: %d\n", VTA_WGT_BUFF_DEPTH*VTA_BLOCK_IN*VTA_BLOCK_OUT);
  printf("VTA_INP_WORDS: %d\n", VTA_INP_BUFF_DEPTH*VTA_BLOCK_IN);
  printf("VTA_ACC_WORDS: %d\n", VTA_ACC_BUFF_DEPTH*VTA_BLOCK_OUT);
  printf("VTA_INS_ELEM_BYTES: %d\n", VTA_INS_ELEM_BYTES);
  printf("VTA_UOP_ELEM_BYTES: %d\n", VTA_UOP_ELEM_BYTES);
  printf("VTA_INP_ELEM_BYTES: %d\n", VTA_INP_ELEM_BYTES);
  printf("VTA_WGT_ELEM_BYTES: %d\n", VTA_WGT_ELEM_BYTES);
  printf("VTA_ACC_ELEM_BYTES: %d\n", VTA_ACC_ELEM_BYTES);
  printf("VTA_BLOCK_IN: %d\n", VTA_BLOCK_IN);
  printf("VTA_BLOCK_OUT: %d\n", VTA_BLOCK_OUT);
  printf("VTA_INSN_MEM_0 [%d-%d]\n", VTA_INSN_MEM_0_0, VTA_INSN_MEM_0_1);
  printf("VTA_INSN_MEM_1 [%d]\n", VTA_INSN_MEM_1);
  printf("VTA_INSN_MEM_2 [%d]\n", VTA_INSN_MEM_2);
  printf("VTA_INSN_MEM_3 [%d]\n", VTA_INSN_MEM_3);
  printf("VTA_INSN_MEM_4 [%d]\n", VTA_INSN_MEM_4);
  printf("VTA_INSN_MEM_5 [%d-%d]\n", VTA_INSN_MEM_5_0, VTA_INSN_MEM_5_1);
  printf("VTA_INSN_MEM_6 [%d-%d]\n", VTA_INSN_MEM_6_0, VTA_INSN_MEM_6_1);
  printf("VTA_INSN_MEM_7 [%d-%d]\n", VTA_INSN_MEM_7_0, VTA_INSN_MEM_7_1);
  printf("VTA_INSN_MEM_8 [%d-%d]\n", VTA_INSN_MEM_8_0, VTA_INSN_MEM_8_1);
  printf("VTA_INSN_MEM_9 [%d-%d]\n", VTA_INSN_MEM_9_0, VTA_INSN_MEM_9_1);
  printf("VTA_INSN_MEM_A [%d-%d]\n", VTA_INSN_MEM_A_0, VTA_INSN_MEM_A_1);
  printf("VTA_INSN_MEM_B [%d-%d]\n", VTA_INSN_MEM_B_0, VTA_INSN_MEM_B_1);
  printf("VTA_INSN_MEM_C [%d-%d]\n", VTA_INSN_MEM_C_0, VTA_INSN_MEM_C_1);
  printf("VTA_INSN_MEM_D [%d-%d]\n", VTA_INSN_MEM_D_0, VTA_INSN_MEM_D_1);
  printf("VTA_INSN_MEM_E [%d-%d]\n", VTA_INSN_MEM_E_0, VTA_INSN_MEM_E_1);
  printf("VTA_INSN_GEM_0 [%d-%d]\n", VTA_INSN_GEM_0_0, VTA_INSN_GEM_0_1);
  printf("VTA_INSN_GEM_1 [%d]\n", VTA_INSN_GEM_1);
  printf("VTA_INSN_GEM_2 [%d]\n", VTA_INSN_GEM_2);
  printf("VTA_INSN_GEM_3 [%d]\n", VTA_INSN_GEM_3);
  printf("VTA_INSN_GEM_4 [%d]\n", VTA_INSN_GEM_4);
  printf("VTA_INSN_GEM_5 [%d]\n", VTA_INSN_GEM_5);
  printf("VTA_INSN_GEM_6 [%d-%d]\n", VTA_INSN_GEM_6_0, VTA_INSN_GEM_6_1);
  printf("VTA_INSN_GEM_7 [%d-%d]\n", VTA_INSN_GEM_7_0, VTA_INSN_GEM_7_1);
  printf("VTA_INSN_GEM_8 [%d-%d]\n", VTA_INSN_GEM_8_0, VTA_INSN_GEM_8_1);
  printf("VTA_INSN_GEM_9 [%d-%d]\n", VTA_INSN_GEM_9_0, VTA_INSN_GEM_9_1);
  printf("VTA_INSN_GEM_A [%d-%d]\n", VTA_INSN_GEM_A_0, VTA_INSN_GEM_A_1);
  printf("VTA_INSN_GEM_B [%d-%d]\n", VTA_INSN_GEM_B_0, VTA_INSN_GEM_B_1);
  printf("VTA_INSN_GEM_C [%d-%d]\n", VTA_INSN_GEM_C_0, VTA_INSN_GEM_C_1);
  printf("VTA_INSN_GEM_D [%d-%d]\n", VTA_INSN_GEM_D_0, VTA_INSN_GEM_D_1);
  printf("VTA_INSN_GEM_E [%d-%d]\n", VTA_INSN_GEM_E_0, VTA_INSN_GEM_E_1);
  printf("VTA_INSN_GEM_F [%d-%d]\n", VTA_INSN_GEM_F_0, VTA_INSN_GEM_F_1);
  printf("VTA_INSN_ALU_E [%d-%d]\n", VTA_INSN_ALU_E_0, VTA_INSN_ALU_E_1);
  printf("VTA_INSN_ALU_F [%d]\n", VTA_INSN_ALU_F);
  printf("VTA_INSN_ALU_G [%d-%d]\n", VTA_INSN_ALU_G_0, VTA_INSN_ALU_G_1);
  printf("VTA_UOP_GEM_0 [%d-%d]\n", VTA_UOP_GEM_0_0, VTA_UOP_GEM_0_1);
  printf("VTA_UOP_GEM_1 [%d-%d]\n", VTA_UOP_GEM_1_0, VTA_UOP_GEM_1_1);
  printf("VTA_UOP_GEM_2 [%d-%d]\n", VTA_UOP_GEM_2_0, VTA_UOP_GEM_2_1);
  printf("VTA_UOP_ALU_0 [%d-%d]\n", VTA_UOP_ALU_0_0, VTA_UOP_ALU_0_1);
  printf("VTA_UOP_ALU_1 [%d-%d]\n", VTA_UOP_ALU_1_0, VTA_UOP_ALU_1_1);
}

void printInstruction(int num_insn, VTAGenericInsn *insns) {
  // Keep tabs on dependence queues
  int l2g_queue = 0;
  int g2l_queue = 0;
  int s2g_queue = 0;
  int g2s_queue = 0;
  // Converter
  union VTAInsn c;
  // Iterate over all instructions
  printf("DEBUG - There are %u instructions\n", num_insn);
  for (int i = 0; i < num_insn; i++) {
    // Fetch instruction and decode opcode
    c.generic = insns[i];
    printf("DEBUG - INSTRUCTION %u: ", i);
    if (c.mem.opcode == VTA_OPCODE_LOAD || c.mem.opcode == VTA_OPCODE_STORE) {
      // Print instruction field information
      if (c.mem.opcode == VTA_OPCODE_LOAD) {
        printf("LOAD ");
        if (c.mem.memory_type == VTA_MEM_ID_UOP) printf("UOP\n");
        if (c.mem.memory_type == VTA_MEM_ID_WGT) printf("WGT\n");
        if (c.mem.memory_type == VTA_MEM_ID_INP) printf("INP\n");
        if (c.mem.memory_type == VTA_MEM_ID_ACC) printf("ACC\n");
      }
      if (c.mem.opcode == VTA_OPCODE_STORE) {
        printf("STORE ACC\n");
      }
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\tDRAM: 0x%08x, SRAM:0x%04x\n",
             static_cast<int>(c.mem.dram_base),
             static_cast<int>(c.mem.sram_base));
      printf("\ty: size=%d, pad=[%d, %d]\n",
             static_cast<int>(c.mem.y_size),
             static_cast<int>(c.mem.y_pad_0),
             static_cast<int>(c.mem.y_pad_1));
      printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n",
             static_cast<int>(c.mem.x_size),
             static_cast<int>(c.mem.x_stride),
             static_cast<int>(c.mem.x_pad_0),
             static_cast<int>(c.mem.x_pad_1));
      if (c.mem.opcode == VTA_OPCODE_STORE) {
        if (c.mem.pop_prev_dep) g2s_queue--;
        if (c.mem.push_prev_dep) s2g_queue++;
      } else if (c.mem.opcode == VTA_OPCODE_LOAD &&
        (c.mem.memory_type == VTA_MEM_ID_INP || c.mem.memory_type == VTA_MEM_ID_WGT)) {
        if (c.mem.pop_next_dep) g2l_queue--;
        if (c.mem.push_next_dep) l2g_queue++;
      } else {
        if (c.mem.pop_prev_dep) l2g_queue--;
        if (c.mem.push_prev_dep) g2l_queue++;
        if (c.mem.pop_next_dep) s2g_queue--;
        if (c.mem.push_next_dep) g2s_queue++;
      }
    } else if (c.mem.opcode == VTA_OPCODE_GEMM) {
      // Print instruction field information
      printf("GEMM\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\trange (%d, %d)\n",
             static_cast<int>(c.gemm.uop_bgn),
             static_cast<int>(c.gemm.uop_end));
      printf("\treset_out: %d\n", static_cast<int>(c.gemm.reset_reg));
      printf("\touter loop - iter: %d, acc: %d, inp: %d, wgt: %d\n",
             static_cast<int>(c.gemm.iter_out),
             static_cast<int>(c.gemm.dst_factor_out),
             static_cast<int>(c.gemm.src_factor_out),
             static_cast<int>(c.gemm.wgt_factor_out));
      printf("\tinner loop - iter: %d, acc: %d, inp: %d, wgt: %d\n",
             static_cast<int>(c.gemm.iter_in),
             static_cast<int>(c.gemm.dst_factor_in),
             static_cast<int>(c.gemm.src_factor_in),
             static_cast<int>(c.gemm.wgt_factor_in));
      if (c.gemm.pop_prev_dep) l2g_queue--;
      if (c.gemm.push_prev_dep) g2l_queue++;
      if (c.gemm.pop_next_dep) s2g_queue--;
      if (c.gemm.push_next_dep) g2s_queue++;
    } else if (c.mem.opcode == VTA_OPCODE_FINISH) {
      printf("FINISH\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      if (c.gemm.pop_prev_dep) l2g_queue--;
      if (c.gemm.push_prev_dep) g2l_queue++;
      if (c.gemm.pop_next_dep) s2g_queue--;
      if (c.gemm.push_next_dep) g2s_queue++;
    } else if (c.mem.opcode == VTA_OPCODE_ALU) {
      // Print instruction field information
      printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
             static_cast<int>(c.mem.pop_prev_dep),
             static_cast<int>(c.mem.pop_next_dep),
             static_cast<int>(c.mem.push_prev_dep),
             static_cast<int>(c.mem.push_next_dep));
      printf("\treset_out: %d\n", static_cast<int>(c.alu.reset_reg));
      printf("\trange (%d, %d)\n",
             static_cast<int>(c.alu.uop_bgn),
             static_cast<int>(c.alu.uop_end));
      printf("\touter loop - iter: %d, dst: %d, src: %d\n",
             static_cast<int>(c.alu.iter_out),
             static_cast<int>(c.alu.dst_factor_out),
             static_cast<int>(c.alu.src_factor_out));
      printf("\tinner loop - iter: %d, dst: %d, src: %d\n",
             static_cast<int>(c.alu.iter_in),
             static_cast<int>(c.alu.dst_factor_in),
             static_cast<int>(c.alu.src_factor_in));
      if (c.alu.pop_prev_dep) l2g_queue--;
      if (c.alu.push_prev_dep) g2l_queue++;
      if (c.alu.pop_next_dep) s2g_queue--;
      if (c.alu.push_next_dep) g2s_queue++;
    }
  }
  printf("DEBUG - l2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
  printf("DEBUG - s2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
}

// Helper function: Print micro-ops status
void printMicroOp(int num_uop, VTAUop *uops) {
  // Iterate over all micro ops
  printf("DEBUG - There are %u micro-ops\n", num_uop);
  for (int i = 0; i < num_uop; i++) {
    // Read micro-op
    printf("DEBUG - UOP %u: ", i);
    printf("acc=%u, inp= %u, wgt=%u\n", uops[i].dst_idx, uops[i].src_idx, uops[i].wgt_idx);
  }
}

int alu_test(int opcode, bool use_imm, int batch, int vector_size, bool uop_compression) {
  // Some assertions
  assert(batch % VTA_BATCH == 0);
  assert(vector_size % VTA_BLOCK_OUT == 0);
  assert(!(opcode == VTA_ALU_OPCODE_SHR && !use_imm));
  printf("=====================================================================================\n");
  printf("INFO - ALU test of %s: batch=%d, vector_size=%d, uop_compression=%d\n",
    getOpcodeString(opcode, use_imm), batch, vector_size, uop_compression);

  // Instruction count
  int ins_size = 3 * batch / VTA_BATCH + 2;
  // Micro op count
  int uop_size = uop_compression ? 1 : vector_size / VTA_BLOCK_OUT;
  // Input/output elements in each transfer
  int tx_size = vector_size / VTA_BLOCK_OUT;
  // Number of input sets to be generated
  int input_sets = (use_imm) ? 1 : 2;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= VTA_UOP_BUFF_DEPTH);
  assert(tx_size * input_sets <= VTA_ACC_BUFF_DEPTH);

  // Immediate values
  acc_T *immediate = static_cast<acc_T *>(malloc(sizeof(acc_T) * batch / VTA_BATCH));
  for (int b = 0; b < batch / VTA_BATCH; b++) {
    if (opcode == VTA_ALU_OPCODE_MIN) {
      immediate[b] = static_cast<acc_T>(
          rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH / 2)) - (1LL << (VTA_INP_WIDTH / 2 - 1)));
    } else if (opcode == VTA_ALU_OPCODE_MAX) {
      immediate[b] = static_cast<acc_T>(
          rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH / 2)) - (1LL << (VTA_INP_WIDTH / 2 - 1)));
    } else if (opcode == VTA_ALU_OPCODE_ADD) {
      immediate[b] = static_cast<acc_T>(
          rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH / 2)) - (1LL << (VTA_INP_WIDTH / 2 - 1)));
    } else if (opcode == VTA_ALU_OPCODE_SHR) {
      immediate[b] = static_cast<acc_T>(
          rand_r(&globalSeed) % VTA_ACC_WIDTH - VTA_ACC_WIDTH/2);
    }
  }

  // Initialize instructions
  VTAGenericInsn *insn_buf =
      static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * ins_size));
  int insn_idx = 0;
  insn_buf[insn_idx++] =
      get1DLoadStoreInsn(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 0, 0, uop_size, 0, 0, 0, 0);
  for (int b = 0; b < batch; b += VTA_BATCH) {
    insn_buf[insn_idx++] = get2DLoadStoreInsn(
        VTA_OPCODE_LOAD,                   // opcode
        VTA_MEM_ID_ACC,                    // vector size
        0,                                 // sram offset
        b / VTA_BATCH * tx_size * input_sets,  // dram offset
        1,                                 // y size
        tx_size * input_sets,              // x size
        tx_size * input_sets,              // x stride
        0,                                 // y pad
        0,                                 // x pad
        0,                                 // pop prev dep
        b > 0,                             // pop next dep
        0,                                 // push prev dep
        0);                                // push next dep
    insn_buf[insn_idx++] = getALUInsn(
        opcode,                            // opcode
        tx_size,                           // vector size
        use_imm,                           // use imm
        immediate[b / VTA_BATCH],          // imm
        uop_compression,                   // uop compression
        0,                                 // pop prev dep
        0,                                 // pop next dep
        0,                                 // push prev dep
        1);                                // push next dep
    insn_buf[insn_idx++] = get2DLoadStoreInsn(
        VTA_OPCODE_STORE,                  // opcode
        VTA_MEM_ID_OUT,                    // vector size
        0,                                 // sram offset
        b / VTA_BATCH * tx_size,           // dram offset
        1,                                 // y size
        tx_size,                           // x size
        tx_size,                           // x stride
        0,                                 // y pad
        0,                                 // x pad
        1,                                 // pop prev dep
        0,                                 // pop next dep
        1,                                 // push prev dep
        0);                                // push next dep
  }
  // Finish
  insn_buf[insn_idx++] = getFinishInsn(0, 1);
  // Prepare the uop buffer
  VTAUop * uop_buf = getMapALUUops(tx_size, uop_compression);

#if VTA_DEBUG == 1
  printInstruction(ins_size, insn_buf);
  printMicroOp(uop_size, uop_buf);
#endif

  // Initialize the input/output data
  acc_T **inputs = alloc2dArray<acc_T>(batch, vector_size * input_sets);
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < vector_size * input_sets; j++) {
      if (opcode == VTA_ALU_OPCODE_MIN) {
        inputs[i][j] = static_cast<acc_T>(
            rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH - 1)) - (1LL << (VTA_INP_WIDTH - 2)));
      } else if (opcode == VTA_ALU_OPCODE_MAX) {
        inputs[i][j] = static_cast<acc_T>(
            rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH - 1)) - (1LL << (VTA_INP_WIDTH - 2)));
      } else if (opcode == VTA_ALU_OPCODE_ADD) {
        inputs[i][j] = static_cast<acc_T>(
            rand_r(&globalSeed) % (1LL << (VTA_INP_WIDTH - 1)) - (1LL << (VTA_INP_WIDTH - 2)));
      }
    }
  }

  // Compute reference output
  out_T **outputs_ref = alloc2dArray<out_T>(batch, vector_size);
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < vector_size; j++) {
      acc_T tmp = 0;
      if (opcode == VTA_ALU_OPCODE_MIN) {
        if (!use_imm) {
          tmp = inputs[i][j] < inputs[i][j + vector_size] ?
                    inputs[i][j] :
                    inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] < immediate[i / VTA_BATCH] ?
                    inputs[i][j] :
                    immediate[i / VTA_BATCH];
        }
      } else if (opcode == VTA_ALU_OPCODE_MAX) {
        if (!use_imm) {
          tmp = inputs[i][j] > inputs[i][j + vector_size] ?
                    inputs[i][j] :
                    inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] > immediate[i / VTA_BATCH] ?
                    inputs[i][j] :
                    immediate[i / VTA_BATCH];
        }
      } else if (opcode == VTA_ALU_OPCODE_ADD) {
        if (!use_imm) {
          tmp = inputs[i][j] + inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] + immediate[i / VTA_BATCH];
        }
      } else if (opcode == VTA_ALU_OPCODE_SHR) {
        if (immediate[i / VTA_BATCH] >= 0) {
          tmp = inputs[i][j] >> immediate[i / VTA_BATCH];
        } else {
          tmp = inputs[i][j] << (0 - immediate[i / VTA_BATCH]);
        }
      }
      // Set
      outputs_ref[i][j] = (out_T) tmp;
    }
  }

  // Pack input buffer
  acc_T *bias_buf =
      static_cast<acc_T *>(allocBuffer(VTA_ACC_ELEM_BYTES * batch * tx_size * input_sets));
  packBuffer<acc_T, VTA_ACC_WIDTH>(
      bias_buf, inputs, batch, vector_size * input_sets, VTA_BATCH, VTA_BLOCK_OUT);

  // Prepare output buffer
  out_T *output_buf =
      static_cast<out_T *>(allocBuffer(VTA_INP_ELEM_BYTES * batch * tx_size * input_sets));

#ifdef NO_SIM
  // Invoke the VTA
  uint64_t t_fpga = vta(ins_size, insn_buf, uop_buf, NULL, NULL, bias_buf, output_buf);
  // Report on timining
  printf("INFO - Synchronization time: %.3fms\n", static_cast<float>(t_fpga) / 1E6);
  printf("INFO - Throughput: %.3fGOps/s\n", static_cast<float>(vector_size * batch) / t_fpga);
#else
  // Invoke the VTA
  vta(ins_size,
      (volatile insn_T *) insn_buf,
      (volatile uop_T *) uop_buf,
      (volatile inp_vec_T *) NULL,
      (volatile wgt_vec_T *) NULL,
      (volatile acc_vec_T *) bias_buf,
      (volatile out_vec_T *) output_buf);
#endif

  // Unpack output buffer
  out_T **outputs = alloc2dArray<out_T>(batch, vector_size);
  unpackBuffer<out_T, VTA_OUT_WIDTH>(outputs,
                                     output_buf,
                                     batch,
                                     vector_size,
                                     VTA_BATCH,
                                     VTA_BLOCK_OUT);

  // Correctness checks
  int err = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < vector_size; j++) {
      if (outputs_ref[i][j] != outputs[i][j]) {
        err++;
#if VTA_DEBUG == 1
        printf("DEBUG - %d, %d: expected 0x%x but got 0x%x\n", i, j,
               static_cast<int>(outputs_ref[i][j]),
               static_cast<int>(outputs[i][j]));
#endif
      }
    }
  }

  // Free all allocated arrays
  free(immediate);
  free2dArray<acc_T>(inputs, batch, vector_size * input_sets);
  free2dArray<out_T>(outputs_ref, batch, vector_size);
  free2dArray<out_T>(outputs, batch, vector_size);
  freeBuffer(insn_buf);
  freeBuffer(uop_buf);
  freeBuffer(bias_buf);
  freeBuffer(output_buf);

  if (err == 0) {
    printf("INFO - ALU test successful!\n");
    return 0;
  } else {
    printf("INFO - ALU test failed, got %d errors!\n", err);
    return -1;
  }
}

int blocked_gemm_test(int batch, int channels, int block, bool uop_compression,
    int virtual_threads) {
  // Some assertions
  assert(block % VTA_BLOCK_IN == 0);
  assert(block % VTA_BLOCK_OUT == 0);
  assert(block % VTA_BATCH == 0);
  assert(channels % block == 0);
  assert(batch % block == 0);

  printf("=====================================================================================\n");
  printf("INFO - Blocked GEMM test: batch=%d, channels=%d, block=%d, uop_comp=%d, vt=%d\n",
         batch, channels, block, uop_compression, virtual_threads);

  // Input/output channels
  int in_feat = channels;
  int out_feat = channels;
  // Derive number of elements that need to be loaded/stored
  int ins_size = batch / block * out_feat / block * (2 + in_feat / block * 3) + 2;
  int uop_size = uop_compression ?
      block / VTA_BATCH * virtual_threads :
      block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT * virtual_threads;
  int inp_size = batch / VTA_BATCH * in_feat / VTA_BLOCK_IN;
  int wgt_size = in_feat / VTA_BLOCK_IN * out_feat / VTA_BLOCK_OUT;
  int out_size = batch / VTA_BATCH * out_feat / VTA_BLOCK_OUT;
  // Blocked buffer sizes (in terms of elements)
  int inp_block_size = block / VTA_BATCH * block / VTA_BLOCK_IN;
  int wgt_block_size = block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT;
  int out_block_size = block / VTA_BATCH * block / VTA_BLOCK_OUT;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= VTA_UOP_BUFF_DEPTH);
  assert(inp_block_size <= VTA_INP_BUFF_DEPTH);
  assert(wgt_block_size <= VTA_WGT_BUFF_DEPTH);
  assert(out_block_size <= VTA_ACC_BUFF_DEPTH);

  // Initialize instruction buffer
  VTAGenericInsn *insn_buf =
      static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * ins_size));
  int insn_idx = 0;

  // Load uops
  insn_buf[insn_idx++] = get1DLoadStoreInsn(VTA_OPCODE_LOAD,
                                            VTA_MEM_ID_UOP,
                                            0,
                                            0,
                                            uop_size,
                                            0,
                                            0,
                                            0,
                                            0);
  // Iterate over batch blocks
  for (int i = 0; i < batch; i += block) {
    // Iterate over output channel blocks
    for (int j = 0; j < out_feat; j += block) {
      // Load bias block (pop next if not first, push prev)
      insn_buf[insn_idx++] = get2DLoadStoreInsn(
          VTA_OPCODE_LOAD,                                    // opcode
          VTA_MEM_ID_ACC,                                     // type
          0,                                                  // sram offset
          (i / VTA_BATCH * out_feat + j) / VTA_BLOCK_OUT,     // dram offset
          block / VTA_BATCH,                                  // y size
          block / VTA_BLOCK_OUT,                              // x size
          out_feat / VTA_BLOCK_OUT,                           // x stride
          0,                                                  // y pad
          0,                                                  // x pad
          0,                                                  // pop prev dep
          (i > 0 || j > 0),                                   // pop next dep
          (virtual_threads == 1),                             // push prev dep
          0);                                                 // push next dep
      // Iterate over input channel blocks
      for (int k = 0; k < in_feat; k += block * virtual_threads) {
        for (int l = 0; l < block * virtual_threads; l += block) {
          // Derive dependence flags
          bool pop = (virtual_threads == 1) ?
              1 :
              (i > 0 || j > 0 || k > 0 || l > 0) && (k + l != block * virtual_threads - block);
          bool push_prev = (virtual_threads == 1) ?
              ((k + l) != in_feat - block) :
              ((k + l) != in_feat - virtual_threads * block) &&
              (
                  (k + l != in_feat - block) ||
                  (j != out_feat - block) ||
                  (i != batch - block));
          bool push_next = (k + l == in_feat - block);
          // Load weight block (pop next)
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_WGT,                                 // type
              l / VTA_BLOCK_IN * block / VTA_BLOCK_OUT,       // sram offset
              (j / VTA_BLOCK_OUT * in_feat + k + l) / VTA_BLOCK_IN,  // dram offset
              block / VTA_BLOCK_OUT,                          // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN,                         // x stride
              0,                                              // y pad
              0,                                              // x pad
              0,                                              // pop prev dep
              pop,                                            // pop next dep
              0,                                              // push prev dep
              0);                                             // push next dep
          // Load input block (push next)
          insn_buf[insn_idx++] = get2DLoadStoreInsn(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_INP,                                 // type
              l / VTA_BLOCK_IN * block / VTA_BATCH,           // sram offset
              (i / VTA_BATCH * in_feat + k + l) / VTA_BLOCK_IN,  // dram offset
              block / VTA_BATCH,                              // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN,                         // x stride
              0,                                              // y pad
              0,                                              // x pad
              0,                                              // pop prev dep
              0,                                              // pop next dep
              0,                                              // push prev dep
              1);                                             // push next dep
          // Perform GEMM (pop prev, push prev if not last, push next if last)
          insn_buf[insn_idx++] = getGEMMInsn(
              l / block * uop_size / virtual_threads,         // uop offset
              block / VTA_BATCH,                              // batch
              block / VTA_BLOCK_IN,                           // in_feat
              block / VTA_BLOCK_OUT,                          // out_feat
              uop_compression,                                // uop_compression
              1,                                              // pop_prev_dep
              0,                                              // pop_next_dep
              push_prev,                                      // push prev dep
              push_next);                                     // push_next_dep
        }
      }
      // Store output block (pop prev, push prev if not last)
      insn_buf[insn_idx++] = get2DLoadStoreInsn(
          VTA_OPCODE_STORE,                                   // opcode
          VTA_MEM_ID_OUT,                                     // type
          0,                                                  // sram offset
          (i / VTA_BATCH * out_feat + j) / VTA_BLOCK_OUT,     // dram offset
          block / VTA_BATCH,                                  // y size
          block / VTA_BLOCK_OUT,                              // x size
          out_feat / VTA_BLOCK_OUT,                           // x stride
          0,                                                  // y pad
          0,                                                  // x pad
          1,                                                  // pop prev dep
          0,                                                  // pop next dep
          1,                                                  // pop prev dep
          0);                                                 // push next dep
    }
  }
  // Finish
  insn_buf[insn_idx++] = getFinishInsn(0, 1);

  // Prepare the uop buffer
  VTAUop * uop_buf = getGEMMUops(
      block / VTA_BATCH,
      block / VTA_BLOCK_IN,
      block / VTA_BLOCK_OUT,
      uop_compression,
      virtual_threads > 1);

#if VTA_DEBUG == 1
  printInstruction(ins_size, insn_buf);
  printMicroOp(uop_size, uop_buf);
#endif

  // Initialize inputs
  inp_T **inputs = allocInit2dArray<inp_T, VTA_INP_WIDTH>(batch, in_feat);
  // Initialize weights
  wgt_T **weights = allocInit2dArray<wgt_T, VTA_WGT_WIDTH>(out_feat, in_feat);
  // Initialize biases
  acc_T **biases = allocInit2dArray<acc_T, VTA_ACC_WIDTH>(batch, out_feat);

  // Reference GEMM implementation
  out_T **outputs_ref = alloc2dArray<out_T>(batch, out_feat);
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < out_feat; j++) {
      acc_T sum = biases[i][j];
      for (int k = 0; k < in_feat; k++) {
        sum += (acc_T) (inputs[i][k] * weights[j][k]);
      }
      // Set
      outputs_ref[i][j] = (out_T) sum;
    }
  }

  // Prepare the input buffer
  inp_T *input_buf = static_cast<inp_T *>(allocBuffer(VTA_INP_ELEM_BYTES * inp_size));
  packBuffer<inp_T, VTA_INP_WIDTH>(input_buf,
                                   inputs,
                                   batch,
                                   in_feat,
                                   VTA_BATCH,
                                   VTA_BLOCK_IN);
  // Prepare the weight buffer
  wgt_T *weight_buf = static_cast<wgt_T *>(allocBuffer(VTA_WGT_ELEM_BYTES * wgt_size));
  packBuffer<wgt_T, VTA_WGT_WIDTH>(weight_buf,
                                   weights,
                                   out_feat,
                                   in_feat,
                                   VTA_BLOCK_OUT,
                                   VTA_BLOCK_IN);
  // Prepare the bias buffer
  acc_T *bias_buf = static_cast<acc_T *>(allocBuffer(VTA_ACC_ELEM_BYTES * out_size));
  packBuffer<acc_T, VTA_ACC_WIDTH>(bias_buf,
                                   biases,
                                   batch,
                                   out_feat,
                                   VTA_BATCH,
                                   VTA_BLOCK_OUT);
  // Prepare the output buffer
  out_T *output_buf = static_cast<out_T *>(allocBuffer(VTA_INP_ELEM_BYTES * out_size));

#ifdef NO_SIM
  // Invoke the VTA
  uint64_t t_fpga = vta(ins_size,
                        insn_buf,
                        uop_buf,
                        input_buf,
                        weight_buf,
                        bias_buf,
                        output_buf);
  // Report on timining
  printf("INFO - Synchronization time: %.3lfms\n", static_cast<float>(t_fpga) / 1E6);
  printf("INFO - Throughput: %.3lfGOPs/s\n",
         static_cast<float>(batch) * in_feat * out_feat * 2 / t_fpga);
#else
  // Invoke the VTA
  vta(ins_size,
      (volatile insn_T *) insn_buf,
      (volatile uop_T *) uop_buf,
      (volatile inp_vec_T *) input_buf,
      (volatile wgt_vec_T *) weight_buf,
      (volatile acc_vec_T *) bias_buf,
      (volatile out_vec_T *) output_buf);
#endif

  // Unpack output data
  out_T **outputs = alloc2dArray<out_T>(batch, out_feat);
  unpackBuffer<out_T, VTA_OUT_WIDTH>(outputs,
                                     output_buf,
                                     batch,
                                     out_feat,
                                     VTA_BATCH,
                                     VTA_BLOCK_OUT);

  // Correctness checks
  int err = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < out_feat; j++) {
      if (outputs_ref[i][j] != outputs[i][j]) {
        err++;
#if VTA_DEBUG == 1
        printf("DEBUG - %d, %d: expected 0x%x but got 0x%x\n", i, j,
               static_cast<int>(outputs_ref[i][j]),
               static_cast<int>(outputs[i][j]));
#endif
      }
    }
  }

  // Free all allocated arrays
  free2dArray<inp_T>(inputs, batch, in_feat);
  free2dArray<wgt_T>(weights, out_feat, in_feat);
  free2dArray<acc_T>(biases, batch, out_feat);
  free2dArray<out_T>(outputs_ref, batch, out_feat);
  free2dArray<out_T>(outputs, batch, out_feat);
  freeBuffer(insn_buf);
  freeBuffer(uop_buf);
  freeBuffer(input_buf);
  freeBuffer(weight_buf);
  freeBuffer(bias_buf);
  freeBuffer(output_buf);

  if (err == 0) {
    printf("INFO - Blocked GEMM test successful!\n");
    return 0;
  } else {
    printf("INFO - Blocked GEMM test failed, got %d errors!\n", err);
    return -1;
  }
}


int gemm_test(int batch, int in_channels, int out_channels, bool uop_compression) {
  // Some assertions
  assert(batch % VTA_BATCH == 0);
  assert(in_channels % VTA_BLOCK_IN == 0);
  assert(out_channels % VTA_BLOCK_OUT == 0);

  printf("=====================================================================================\n");
  printf("INFO - Blocked GEMM test: batch=%d, in_channels=%d, out_channels=%d, uop_comp=%d\n",
         batch, in_channels, out_channels, uop_compression);

  // Derive number of elements that need to be loaded/stored
  int ins_size = 7;
  int uop_size = uop_compression ?
      batch / VTA_BATCH :
      batch / VTA_BATCH * in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT;
  int inp_size = batch / VTA_BATCH * in_channels / VTA_BLOCK_IN;
  int wgt_size = in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT;
  int out_size = batch / VTA_BATCH * out_channels / VTA_BLOCK_OUT;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= VTA_UOP_BUFF_DEPTH);
  assert(inp_size <= VTA_INP_BUFF_DEPTH);
  assert(wgt_size <= VTA_WGT_BUFF_DEPTH);
  assert(out_size <= VTA_ACC_BUFF_DEPTH);

  // Initialize instruction buffer
  VTAGenericInsn *insn_buf =
      static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * ins_size));
  int insn_idx = 0;

  // Load uops
  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,
      VTA_MEM_ID_UOP,
      0,
      0,
      uop_size,
      0,
      0,
      0,
      0);
  // Load bias
  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_ACC,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      out_size,                                           // size
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      1,                                                  // push prev dep
      0);                                                 // push next dep
  // Load weight block (pop next)
  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_WGT,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      wgt_size,                                           // size
      0,                                                  // pop prev dep
      1,                                                  // pop next dep
      0,                                                  // push prev dep
      0);                                                 // push next dep
  // Load input block (push next)
  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_INP,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      inp_size,                                           // size
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      0,                                                  // push prev dep
      1);                                                 // push next dep
  // Perform GEMM (pop prev, push prev if not last, push next if last)
  insn_buf[insn_idx++] = getGEMMInsn(
      0,                                                  // uop offset
      batch / VTA_BATCH,                                  // batch
      in_channels / VTA_BLOCK_IN,                         // in_channels
      out_channels / VTA_BLOCK_OUT,                       // out_channels
      uop_compression,                                    // uop_compression
      1,                                                  // pop_prev_dep
      0,                                                  // pop_next_dep
      0,                                                  // push prev dep
      1);                                                 // push_next_dep
  // Store output block (pop prev, push prev if not last)
  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_STORE,                                   // opcode
      VTA_MEM_ID_OUT,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      out_size,                                           // size
      1,                                                  // pop prev dep
      0,                                                  // pop next dep
      1,                                                  // push prev dep
      0);                                                 // push next dep
  // Finish
  insn_buf[insn_idx++] = getFinishInsn(0, 1);

  // Prepare the uop buffer
  VTAUop * uop_buf = getGEMMUops(
      batch / VTA_BATCH,
      in_channels / VTA_BLOCK_IN,
      out_channels / VTA_BLOCK_OUT,
      uop_compression,
      0);

#if VTA_DEBUG == 1
  printInstruction(ins_size, insn_buf);
  printMicroOp(uop_size, uop_buf);
#endif

  // Initialize inputs
  inp_T **inputs = allocInit2dArray<inp_T, VTA_INP_WIDTH>(batch, in_channels);
  // Initialize weights
  wgt_T **weights = allocInit2dArray<wgt_T, VTA_WGT_WIDTH>(out_channels, in_channels);
  // Initialize biases
  acc_T **biases = allocInit2dArray<acc_T, VTA_ACC_WIDTH>(batch, out_channels);

  // Reference GEMM implementation
  out_T **outputs_ref = alloc2dArray<out_T>(batch, out_channels);
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < out_channels; j++) {
      acc_T sum = biases[i][j];
      for (int k = 0; k < in_channels; k++) {
        sum += (acc_T) (inputs[i][k] * weights[j][k]);
      }
      // Set
      outputs_ref[i][j] = (out_T) sum;
    }
  }

  // Prepare the input buffer
  inp_T *input_buf = static_cast<inp_T *>(allocBuffer(VTA_INP_ELEM_BYTES * inp_size));
  packBuffer<inp_T, VTA_INP_WIDTH>(input_buf,
                                   inputs,
                                   batch,
                                   in_channels,
                                   VTA_BATCH,
                                   VTA_BLOCK_IN);
  // Prepare the weight buffer
  wgt_T *weight_buf = static_cast<wgt_T *>(allocBuffer(VTA_WGT_ELEM_BYTES * wgt_size));
  packBuffer<wgt_T, VTA_WGT_WIDTH>(weight_buf,
                                   weights,
                                   out_channels,
                                   in_channels,
                                   VTA_BLOCK_OUT,
                                   VTA_BLOCK_IN);
  // Prepare the bias buffer
  acc_T *bias_buf = static_cast<acc_T *>(allocBuffer(VTA_ACC_ELEM_BYTES * out_size));
  packBuffer<acc_T, VTA_ACC_WIDTH>(bias_buf,
                                   biases,
                                   batch,
                                   out_channels,
                                   VTA_BATCH,
                                   VTA_BLOCK_OUT);
  // Prepare the output buffer
  out_T *output_buf = static_cast<out_T *>(allocBuffer(VTA_INP_ELEM_BYTES * out_size));

#ifdef NO_SIM
  // Invoke the VTA
  uint64_t t_fpga = vta(ins_size,
                        insn_buf,
                        uop_buf,
                        input_buf,
                        weight_buf,
                        bias_buf,
                        output_buf);
  // Report on timining
  printf("INFO - Synchronization time: %.3lfms\n", static_cast<float>(t_fpga) / 1E6);
  printf("INFO - Throughput: %.3lfGOPs/s\n",
         static_cast<float>(batch) * in_channels * out_channels * 2 / t_fpga);
#else
  // Invoke the VTA
  vta(ins_size,
      (volatile insn_T *) insn_buf,
      (volatile uop_T *) uop_buf,
      (volatile inp_vec_T *) input_buf,
      (volatile wgt_vec_T *) weight_buf,
      (volatile acc_vec_T *) bias_buf,
      (volatile out_vec_T *) output_buf);
#endif

  // Unpack output data
  out_T **outputs = alloc2dArray<out_T>(batch, out_channels);
  unpackBuffer<out_T, VTA_OUT_WIDTH>(outputs,
                                     output_buf,
                                     batch,
                                     out_channels,
                                     VTA_BATCH,
                                     VTA_BLOCK_OUT);

  // Correctness checks
  int err = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < out_channels; j++) {
      if (outputs_ref[i][j] != outputs[i][j]) {
        err++;
#if VTA_DEBUG == 1
        printf("DEBUG - %d, %d: expected 0x%x but got 0x%x\n", i, j,
               static_cast<int>(outputs_ref[i][j]),
               static_cast<int>(outputs[i][j]));
#endif
      }
    }
  }

  // Free all allocated arrays
  free2dArray<inp_T>(inputs, batch, in_channels);
  free2dArray<wgt_T>(weights, out_channels, in_channels);
  free2dArray<acc_T>(biases, batch, out_channels);
  free2dArray<out_T>(outputs_ref, batch, out_channels);
  free2dArray<out_T>(outputs, batch, out_channels);
  freeBuffer(insn_buf);
  freeBuffer(uop_buf);
  freeBuffer(input_buf);
  freeBuffer(weight_buf);
  freeBuffer(bias_buf);
  freeBuffer(output_buf);

  if (err == 0) {
    printf("INFO - Blocked GEMM test successful!\n");
    return 0;
  } else {
    printf("INFO - Blocked GEMM test failed, got %d errors!\n", err);
    return -1;
  }
}
