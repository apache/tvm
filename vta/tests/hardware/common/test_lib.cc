/*!
 *  Copyright (c) 2018 by Contributors
 * \file vta_test_lib.cpp
 * \brief Test library for the VTA design simulation and driver tests.
 */

#include "./test_lib.h"

const char* getOpcodeString(int opcode, bool use_imm) {
  // Returns string name
  if (opcode == ALU_OPCODE_MIN) {
    if (use_imm) {
      return "min imm";
    } else {
      return "min";
    }
  } else if (opcode == ALU_OPCODE_MAX) {
    if (use_imm) {
      return "max imm";
    } else {
      return "max";
    }
  } else if (opcode == ALU_OPCODE_ADD) {
    if (use_imm) {
      return "add imm";
    } else {
      return "add";
    }
  } else if (opcode == ALU_OPCODE_SUB) {
    if (use_imm) {
      return "sub imm";
    } else {
      return "sub";
    }
  } else if (opcode == ALU_OPCODE_MUL) {
    if (use_imm) {
      return "mul imm";
    } else {
      return "mul";
    }
  } else if (opcode == ALU_OPCODE_SHL) {
    return "shl";
  } else if (opcode == ALU_OPCODE_SHR) {
    return "shr";
  }
  return "unknown op";
}

template <typename T, int T_WIDTH>
void packBuffer(T *dst, T **src, int y_size, int x_size, int y_block, int x_block) {
  int buffer_idx = 0;
  for(int i = 0; i < y_size / y_block; i ++) {
    for(int j = 0; j < x_size / x_block; j ++) {
      for(int k = 0; k < y_block; k ++) {
        if (T_WIDTH < 8) {
          for (int l = 0; l < x_block; l += 8 / T_WIDTH) {
            dst[buffer_idx] = 0;
            for (int m = 0; m < 8 / T_WIDTH; m ++) {
              dst[buffer_idx] |= (src[i * y_block + k][j * x_block + l + m] &
                ((1ULL << T_WIDTH) - 1)) << (m * T_WIDTH);
            }
            buffer_idx ++;
          }
        } else {
          for (int l = 0; l < x_block; l ++) {
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
  for(int i = 0; i < y_size / y_block; i ++) {
    for(int j = 0; j < x_size / x_block; j ++) {
      for(int k = 0; k < y_block; k ++) {
        if (T_WIDTH < 8) {
          for (int l = 0; l < x_block; l += 8 / T_WIDTH) {
            for (int m = 0; m < 8 / T_WIDTH; m ++) {
              dst[i * y_block + k][j * x_block + l + m] = (src[buffer_idx] >> (m * T_WIDTH))
                & ((1 << T_WIDTH) - 1);
            }
            buffer_idx ++;
          }
        } else {
          for (int l = 0; l < x_block; l ++) {
            dst[i * y_block + k][j * x_block + l] = src[buffer_idx ++];
          }
        }
      }
    }
  }
}

template <typename T, int T_WIDTH>
T ** allocInit2dArray(int rows, int cols) {
  // Allocate
  T **array = (T **) malloc(sizeof(T *) * rows);
  for (int i = 0; i < rows; i ++) {
    array[i] = (T *) malloc(sizeof(T) * cols);
  }
  // Init
  for (int i = 0; i < rows; i ++) {
    for (int j = 0; j < cols; j ++) {
      array[i][j] = (T) (rand() % (1LL << (T_WIDTH - 1)) - (1LL << (T_WIDTH - 2)));
    }
  }
  return array;
}

template <typename T>
T ** alloc2dArray(int rows, int cols) {
  T **array = (T **) malloc(sizeof(T *) * rows);
  for (int i = 0; i < rows; i ++) {
    array[i] = (T *) malloc(sizeof(T) * cols);
  }
  return array;
}

template <typename T>
void free2dArray(T **array, int rows, int cols) {
  for (int i = 0; i < rows; i ++) {
    free(array[i]);
  }
  free(array);
}

template <typename T>
T *** alloc3dArray(int rows, int cols, int depth) {
  T ***array = (T ***) malloc(sizeof(T **) * rows);
  for (int i = 0; i < rows; i ++) {
    array[i] = (T **) malloc(sizeof(T *) * cols);
    for (int j = 0; j < cols; j ++) {
      array[i][j] = (T*) malloc(sizeof(T) * depth);
    }
  }
  return array;
}

template <typename T>
void free3dArray(T *** array, int rows, int cols, int depth) {
  for (int i = 0; i < rows; i ++) {
    for (int j = 0; j < cols; j ++) {
      free(array[i][j]);
    }
    free(array[i]);
  }
  free(array);
}

void * allocBuffer(size_t num_bytes) {
#ifdef NO_SIM
  return VTAMemAlloc(num_bytes, CACHED);
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

VTAGenericInsn reset2DInsn(int type, int sram_offset, int y_size, int x_size, int x_stride,
    int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = OPCODE_LOAD;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = 0;
  insn.y_size = 0;
  insn.x_size = x_size;
  insn.x_stride = x_stride;
  insn.y_pad_0 = y_size;
  insn.y_pad_1 = 0;
  insn.x_pad_0 = 0;
  insn.x_pad_1 = 0;
  converter.mem = insn;
  return converter.generic;
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
  // GEVM instruction initialization
  VTAGemInsn insn;
  insn.opcode = OPCODE_GEMM;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
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
  insn.opcode = OPCODE_ALU;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
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
  // GEVM instruction initialization
  VTAGemInsn insn;
  insn.opcode = OPCODE_FINISH;
  insn.pop_prev_dep = pop_prev;
  insn.pop_next_dep = pop_next;
  insn.push_prev_dep = 0;
  insn.push_next_dep = 0;
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
  VTAUop *uop_buf = (VTAUop *) VTAMemAlloc(sizeof(VTAUop) * uop_size, CACHED);
#else
  VTAUop *uop_buf = (VTAUop *) malloc(sizeof(VTAUop) * uop_size);
#endif

  if (!uop_compression) {
    int uop_idx = 0;
    for (int i = 0; i < y_size; i ++) {
      for (int j = 0; j < x_size; j ++) {
        uop_buf[uop_idx].reset_out = false;
        uop_buf[uop_idx].dst_idx = i * x_size + j;
        uop_buf[uop_idx].src_idx = 0;
        uop_buf[uop_idx].wgt_idx = 0;
        uop_idx++;
      }
    }
  } else {
    uop_buf[0].reset_out = false;
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
  VTAUop *uop_buf = (VTAUop *) VTAMemAlloc(sizeof(VTAUop) * uop_size, CACHED);
#else
  VTAUop *uop_buf = (VTAUop *) malloc(sizeof(VTAUop) * uop_size);
#endif

  if (!uop_compression) {
    int uop_idx = 0;
    for (int i = 0; i < batch; i ++) {
      for (int j = 0; j < in_feat; j ++) {
        for (int k = 0; k < out_feat; k ++) {
          uop_buf[uop_idx].reset_out = false;
          uop_buf[uop_idx].dst_idx = i * out_feat + k;
          uop_buf[uop_idx].src_idx = i * in_feat + j;
          uop_buf[uop_idx].wgt_idx = k * in_feat + j;
          uop_idx++;
        }
      }
    }
  } else {
    for (int i = 0; i < batch; i ++) {
      uop_buf[i].reset_out = false;
      uop_buf[i].dst_idx = i * out_feat;
      uop_buf[i].src_idx = i * in_feat;
      uop_buf[i].wgt_idx = 0;
    }
  }

  if (multi_threaded) {
    if (!uop_compression) {
      int uop_idx = uop_size / 2;
      for (int i = 0; i < batch; i ++) {
        for (int j = 0; j < in_feat; j ++) {
          for (int k = 0; k < out_feat; k ++) {
            uop_buf[uop_idx].reset_out = false;
            uop_buf[uop_idx].dst_idx = i * out_feat + k;
            uop_buf[uop_idx].src_idx = batch * in_feat + i * in_feat + j;
            uop_buf[uop_idx].wgt_idx = out_feat * in_feat + k * in_feat + j;
            uop_idx++;
          }
        }
      }
    } else {
      for (int i = 0; i < batch; i ++) {
        uop_buf[batch+i].reset_out = false;
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
  VTAUop *uop_buf = (VTAUop *) VTAMemAlloc(sizeof(VTAUop) * uop_size, CACHED);
#else
  VTAUop *uop_buf = (VTAUop *) malloc(sizeof(VTAUop) * uop_size);
#endif

  if (!uop_compression) {
    for (int i = 0; i < vector_size; i ++) {
      uop_buf[i].reset_out = 0;
      uop_buf[i].dst_idx = i;
      uop_buf[i].src_idx = vector_size + i;
    }
  } else {
    uop_buf[0].reset_out = 0;
    uop_buf[0].dst_idx = 0;
    uop_buf[0].src_idx = vector_size;
  }

  return uop_buf;
}

void printParameters() {
  // Some debugging code
  printf("Size of VTAInsn: %d\n", sizeof(VTAGenericInsn));
  printf("Size of VTAUop: %d\n", sizeof(VTAUop));
  printf("UOP_BUFF_DEPTH: %d\n", UOP_BUFF_DEPTH);
  printf("LOG_UOP_BUFF_DEPTH: %d\n", LOG_UOP_BUFF_DEPTH);
  printf("WGT_BUFF_DEPTH: %d\n", WGT_BUFF_DEPTH);
  printf("LOG_WGT_BUFF_DEPTH: %d\n", LOG_WGT_BUFF_DEPTH);
  printf("INP_BUFF_DEPTH: %d\n", INP_BUFF_DEPTH);
  printf("LOG_INP_BUFF_DEPTH: %d\n", LOG_INP_BUFF_DEPTH);
  printf("ACC_BUFF_DEPTH: %d\n", ACC_BUFF_DEPTH);
  printf("LOG_ACC_BUFF_DEPTH: %d\n", LOG_ACC_BUFF_DEPTH);
  printf("WGT_WORDS: %d\n", WGT_BUFF_DEPTH*BLOCK_IN*BLOCK_OUT);
  printf("INP_WORDS: %d\n", INP_BUFF_DEPTH*BLOCK_IN);
  printf("ACC_WORDS: %d\n", ACC_BUFF_DEPTH*BLOCK_OUT);
  printf("INS_ELEM_BYTES: %d\n", INS_ELEM_BYTES);
  printf("UOP_ELEM_BYTES: %d\n", UOP_ELEM_BYTES);
  printf("INP_ELEM_BYTES: %d\n", INP_ELEM_BYTES);
  printf("WGT_ELEM_BYTES: %d\n", WGT_ELEM_BYTES);
  printf("ACC_ELEM_BYTES: %d\n", ACC_ELEM_BYTES);
  printf("BLOCK_IN: %d\n", BLOCK_IN);
  printf("BLOCK_OUT: %d\n", BLOCK_OUT);
  printf("INSN_MEM_0 [%d-%d]\n", INSN_MEM_0_0, INSN_MEM_0_1);
  printf("INSN_MEM_1 [%d]\n", INSN_MEM_1);
  printf("INSN_MEM_2 [%d]\n", INSN_MEM_2);
  printf("INSN_MEM_3 [%d]\n", INSN_MEM_3);
  printf("INSN_MEM_4 [%d]\n", INSN_MEM_4);
  printf("INSN_MEM_5 [%d-%d]\n", INSN_MEM_5_0, INSN_MEM_5_1);
  printf("INSN_MEM_6 [%d-%d]\n", INSN_MEM_6_0, INSN_MEM_6_1);
  printf("INSN_MEM_7 [%d-%d]\n", INSN_MEM_7_0, INSN_MEM_7_1);
  printf("INSN_MEM_8 [%d-%d]\n", INSN_MEM_8_0, INSN_MEM_8_1);
  printf("INSN_MEM_9 [%d-%d]\n", INSN_MEM_9_0, INSN_MEM_9_1);
  printf("INSN_MEM_A [%d-%d]\n", INSN_MEM_A_0, INSN_MEM_A_1);
  printf("INSN_MEM_B [%d-%d]\n", INSN_MEM_B_0, INSN_MEM_B_1);
  printf("INSN_MEM_C [%d-%d]\n", INSN_MEM_C_0, INSN_MEM_C_1);
  printf("INSN_MEM_D [%d-%d]\n", INSN_MEM_D_0, INSN_MEM_D_1);
  printf("INSN_MEM_E [%d-%d]\n", INSN_MEM_E_0, INSN_MEM_E_1);
  printf("INSN_GEM_0 [%d-%d]\n", INSN_GEM_0_0, INSN_GEM_0_1);
  printf("INSN_GEM_1 [%d]\n", INSN_GEM_1);
  printf("INSN_GEM_2 [%d]\n", INSN_GEM_2);
  printf("INSN_GEM_3 [%d]\n", INSN_GEM_3);
  printf("INSN_GEM_4 [%d]\n", INSN_GEM_4);
  printf("INSN_GEM_5 [%d-%d]\n", INSN_GEM_5_0, INSN_GEM_5_1);
  printf("INSN_GEM_6 [%d-%d]\n", INSN_GEM_6_0, INSN_GEM_6_1);
  printf("INSN_GEM_7 [%d-%d]\n", INSN_GEM_7_0, INSN_GEM_7_1);
  printf("INSN_GEM_8 [%d-%d]\n", INSN_GEM_8_0, INSN_GEM_8_1);
  printf("INSN_GEM_9 [%d-%d]\n", INSN_GEM_9_0, INSN_GEM_9_1);
  printf("INSN_GEM_A [%d-%d]\n", INSN_GEM_A_0, INSN_GEM_A_1);
  printf("INSN_GEM_B [%d-%d]\n", INSN_GEM_B_0, INSN_GEM_B_1);
  printf("INSN_GEM_C [%d-%d]\n", INSN_GEM_C_0, INSN_GEM_C_1);
  printf("INSN_GEM_D [%d-%d]\n", INSN_GEM_D_0, INSN_GEM_D_1);
  printf("INSN_GEM_E [%d-%d]\n", INSN_GEM_E_0, INSN_GEM_E_1);
  printf("INSN_ALU_D [%d-%d]\n", INSN_ALU_D_0, INSN_ALU_D_1);
  printf("INSN_ALU_E [%d]\n", INSN_ALU_E);
  printf("INSN_ALU_F [%d-%d]\n", INSN_ALU_F_0, INSN_ALU_F_1);
  printf("UOP_GEM_0 [%d]\n", UOP_GEM_0);
  printf("UOP_GEM_1 [%d-%d]\n", UOP_GEM_1_0, UOP_GEM_1_1);
  printf("UOP_GEM_2 [%d-%d]\n", UOP_GEM_2_0, UOP_GEM_2_1);
  printf("UOP_GEM_3 [%d-%d]\n", UOP_GEM_3_0, UOP_GEM_3_1);
  printf("UOP_ALU_0 [%d]\n", UOP_ALU_0);
  printf("UOP_ALU_1 [%d-%d]\n", UOP_ALU_1_0, UOP_ALU_1_1);
  printf("UOP_ALU_2 [%d-%d]\n", UOP_ALU_2_0, UOP_ALU_2_1);
  printf("UOP_ALU_3 [%d-%d]\n", UOP_ALU_3_0, UOP_ALU_3_1);
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
  for (int i = 0; i < num_insn; i ++) {
    // Fetch instruction and decode opcode
    c.generic = insns[i];
    printf("DEBUG - INSTRUCTION %u: ", i);
    if (c.mem.opcode == OPCODE_LOAD || c.mem.opcode == OPCODE_STORE) {
      // Print instruction field information
      if (c.mem.opcode == OPCODE_LOAD) {
        printf("LOAD ");
        if (c.mem.memory_type == MEM_ID_UOP) printf("UOP\n");
        if (c.mem.memory_type == MEM_ID_WGT) printf("WGT\n");
        if (c.mem.memory_type == MEM_ID_INP) printf("INP\n");
        if (c.mem.memory_type == MEM_ID_ACC) printf("ACC\n");
      }
      if (c.mem.opcode == OPCODE_STORE) {
        printf("STORE ACC\n");
      }
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
        (int) c.mem.pop_prev_dep, (int) c.mem.pop_next_dep,
        (int) c.mem.push_prev_dep, (int) c.mem.push_next_dep);
      printf("\tDRAM: 0x%08x, SRAM:0x%04x\n", (int) c.mem.dram_base, (int) c.mem.sram_base);
      printf("\ty: size=%d, pad=[%d, %d]\n", (int) c.mem.y_size, (int) c.mem.y_pad_0,
        (int) c.mem.y_pad_1);
      printf("\tx: size=%d, stride=%d, pad=[%d, %d]\n", (int) c.mem.x_size, (int) c.mem.x_stride,
        (int) c.mem.x_pad_0, (int) c.mem.x_pad_1);
      if (c.mem.opcode == OPCODE_STORE) {
        if (c.mem.pop_prev_dep) g2s_queue --;
        if (c.mem.push_prev_dep) s2g_queue ++;
      } else if (c.mem.opcode == OPCODE_LOAD &&
        (c.mem.memory_type == MEM_ID_INP || c.mem.memory_type == MEM_ID_WGT)) {
        if (c.mem.pop_next_dep) g2l_queue --;
        if (c.mem.push_next_dep) l2g_queue ++;
      } else {
        if (c.mem.pop_prev_dep) l2g_queue --;
        if (c.mem.push_prev_dep) g2l_queue ++;
        if (c.mem.pop_next_dep) s2g_queue --;
        if (c.mem.push_next_dep) g2s_queue ++;
      }
    } else if (c.mem.opcode == OPCODE_GEMM) {
      // Print instruction field information
      printf("GEVM\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
        (int) c.mem.pop_prev_dep, (int) c.mem.pop_next_dep,
        (int) c.mem.push_prev_dep, (int) c.mem.push_next_dep);
      printf("\trange (%d, %d)\n", (int) c.gemm.uop_bgn, (int) c.gemm.uop_end);
      printf("\touter loop - iter: %d, acc: %d, inp: %d, wgt: %d\n", (int) c.gemm.iter_out,
        (int) c.gemm.dst_factor_out, (int) c.gemm.src_factor_out,
        (int) c.gemm.wgt_factor_out);
      printf("\tinner loop - iter: %d, acc: %d, inp: %d, wgt: %d\n", (int) c.gemm.iter_in,
        (int) c.gemm.dst_factor_in, (int) c.gemm.src_factor_in,
        (int) c.gemm.wgt_factor_in);
      if (c.gemm.pop_prev_dep) l2g_queue --;
      if (c.gemm.push_prev_dep) g2l_queue ++;
      if (c.gemm.pop_next_dep) s2g_queue --;
      if (c.gemm.push_next_dep) g2s_queue ++;
    } else if (c.mem.opcode == OPCODE_FINISH) {
      printf("FINISH\n");
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
        (int) c.mem.pop_prev_dep, (int) c.mem.pop_next_dep,
        (int) c.mem.push_prev_dep, (int) c.mem.push_next_dep);
      if (c.gemm.pop_prev_dep) l2g_queue --;
      if (c.gemm.push_prev_dep) g2l_queue ++;
      if (c.gemm.pop_next_dep) s2g_queue --;
      if (c.gemm.push_next_dep) g2s_queue ++;
    } else if (c.mem.opcode == OPCODE_ALU) {
      // Print instruction field information
      printf("ALU - %s\n", getOpcodeString(c.alu.alu_opcode, c.alu.use_imm));
      printf("\tdep - pop prev: %d, pop next: %d, push prev: %d, push next: %d\n",
        (int) c.mem.pop_prev_dep, (int) c.mem.pop_next_dep,
        (int) c.mem.push_prev_dep, (int) c.mem.push_next_dep);
      printf("\trange (%d, %d)\n", (int) c.alu.uop_bgn, (int) c.alu.uop_end);
      printf("\touter loop - iter: %d, dst: %d, src: %d\n", (int) c.alu.iter_out,
        (int) c.alu.dst_factor_out, (int) c.alu.src_factor_out);
      printf("\tinner loop - iter: %d, dst: %d, src: %d\n", (int) c.alu.iter_in,
        (int) c.alu.dst_factor_in, (int) c.alu.src_factor_in);
      if (c.alu.pop_prev_dep) l2g_queue --;
      if (c.alu.push_prev_dep) g2l_queue ++;
      if (c.alu.pop_next_dep) s2g_queue --;
      if (c.alu.push_next_dep) g2s_queue ++;
    }
  }
  printf("DEBUG - l2g_queue = %d, g2l_queue = %d\n", l2g_queue, g2l_queue);
  printf("DEBUG - s2g_queue = %d, g2s_queue = %d\n", s2g_queue, g2s_queue);
}

// Helper function: Print micro-ops status
void printMicroOp(int num_uop, VTAUop *uops) {
  // Iterate over all micro ops
  printf("DEBUG - There are %u micro-ops\n", num_uop);
  for (int i = 0; i < num_uop; i ++) {
    // Read micro-op
    printf("DEBUG - UOP %u: ", i);
    printf("rst_out=%u, acc=%u, inp= %u, wgt=%u\n", uops[i].reset_out, uops[i].dst_idx,
        uops[i].src_idx, uops[i].wgt_idx);

  }
}

int alu_test(int opcode, bool use_imm, int batch, int vector_size, bool uop_compression) {

  assert(batch % BATCH == 0);
  assert(vector_size % BLOCK_OUT == 0);
  assert(!(opcode == ALU_OPCODE_SHL && !use_imm));
  assert(!(opcode == ALU_OPCODE_SHR && !use_imm));

  printf("=====================================================================================\n");
  printf("INFO - ALU test of %s: batch=%d, vector_size=%d, uop_compression=%d\n",
    getOpcodeString(opcode, use_imm), batch, vector_size, uop_compression);

  // Instruction count
  int ins_size = 3 * batch / BATCH + 2;
  // Micro op count
  int uop_size = uop_compression ? 1 : vector_size / BLOCK_OUT;
  // Input/output elements in each transfer
  int tx_size = vector_size / BLOCK_OUT;
  // Number of input sets to be generated
  int input_sets = (use_imm) ? 1 : 2;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= UOP_BUFF_DEPTH);
  assert(tx_size * input_sets <= ACC_BUFF_DEPTH);

  // Immediate values
  acc_T *immediate = (acc_T *) malloc(sizeof(acc_T) * batch / BATCH);
  for (int b = 0; b < batch / BATCH; b ++) {
    if (opcode == ALU_OPCODE_MIN) {
      immediate[b] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
    } else if (opcode == ALU_OPCODE_MAX) {
      immediate[b] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
    } else if (opcode == ALU_OPCODE_ADD) {
      immediate[b] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
    } else if (opcode == ALU_OPCODE_SUB) {
      immediate[b] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
    } else if (opcode == ALU_OPCODE_MUL) {
      immediate[b] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
    } else if (opcode == ALU_OPCODE_SHL) {
      immediate[b] = (acc_T) (rand() % (INP_WIDTH + 1));
    } else if (opcode == ALU_OPCODE_SHR) {
      immediate[b] = (acc_T) (rand() % (INP_WIDTH + 1));
    }
  }

  // Initialize instructions
  VTAGenericInsn *insn_buf = (VTAGenericInsn *) allocBuffer(sizeof(VTAGenericInsn) * ins_size);
  int insn_idx = 0;
  insn_buf[insn_idx ++] = get1DLoadStoreInsn(OPCODE_LOAD, MEM_ID_UOP, 0, 0, uop_size, 0, 0, 0, 0);
  for (int b = 0; b < batch; b += BATCH) {
    insn_buf[insn_idx ++] = get2DLoadStoreInsn(
      OPCODE_LOAD,                      // opcode
      MEM_ID_ACC,                       // vector size
      0,                                // sram offset
      b / BATCH * tx_size * input_sets, // dram offset
      1,                                // y size
      tx_size * input_sets,             // x size
      tx_size * input_sets,             // x stride
      0,                                // y pad
      0,                                // x pad
      0,                                // pop prev dep
      b > 0,                            // pop next dep
      0,                                // push prev dep
      0);                               // push next dep
    insn_buf[insn_idx ++] = getALUInsn(
      opcode,                           // opcode
      tx_size,                          // vector size
      use_imm,                          // use imm
      immediate[b / BATCH],             // imm
      uop_compression,                  // uop compression
      0,                                // pop prev dep
      0,                                // pop next dep
      0,                                // push prev dep
      1);                               // push next dep
    insn_buf[insn_idx ++] = get2DLoadStoreInsn(
      OPCODE_STORE,                     // opcode
      MEM_ID_OUT,                       // vector size
      0,                                // sram offset
      b / BATCH * tx_size,              // dram offset
      1,                                // y size
      tx_size,                          // x size
      tx_size,                          // x stride
      0,                                // y pad
      0,                                // x pad
      1,                                // pop prev dep
      0,                                // pop next dep
      1,                                // push prev dep
      0);                               // push next dep
  }
  // Finish
  insn_buf[insn_idx ++] = getFinishInsn(0, 1);

  // Prepare the uop buffer
  VTAUop * uop_buf = getMapALUUops(tx_size, uop_compression);

#if DEBUG==1
  printInstruction(ins_size, insn_buf);
  printMicroOp(uop_size, uop_buf);
#endif

  // Initialize the input/output data
  acc_T **inputs = alloc2dArray<acc_T>(batch, vector_size * input_sets);
  for (int i = 0; i < batch; i ++) {
    for (int j = 0; j < vector_size * input_sets; j ++) {
      if (opcode == ALU_OPCODE_MIN) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      } else if (opcode == ALU_OPCODE_MAX) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      } else if (opcode == ALU_OPCODE_ADD) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      } else if (opcode == ALU_OPCODE_SUB) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      } else if (opcode == ALU_OPCODE_MUL) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH / 2)) - (1LL << (INP_WIDTH / 2 - 1)));
      } else if (opcode == ALU_OPCODE_SHL) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      } else if (opcode == ALU_OPCODE_SHR) {
        inputs[i][j] = (acc_T) (rand() % (1LL << (INP_WIDTH - 1)) - (1LL << (INP_WIDTH - 2)));
      }
    }
  }

  // Compute reference output
  out_T **outputs_ref = alloc2dArray<out_T>(batch, vector_size);
  for (int i = 0; i < batch; i ++) {
    for (int j = 0; j < vector_size; j ++) {
      acc_T tmp = 0;
      if (opcode == ALU_OPCODE_MIN) {
        if (!use_imm) {
          tmp = inputs[i][j] < inputs[i][j + vector_size] ? inputs[i][j] : inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] < immediate[i / BATCH] ? inputs[i][j] : immediate[i / BATCH];
        }
      } else if (opcode == ALU_OPCODE_MAX) {
        if (!use_imm) {
          tmp = inputs[i][j] > inputs[i][j + vector_size] ? inputs[i][j] : inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] > immediate[i / BATCH] ? inputs[i][j] : immediate[i / BATCH];
        }
      } else if (opcode == ALU_OPCODE_ADD) {
        if (!use_imm) {
          tmp = inputs[i][j] + inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] + immediate[i / BATCH];
        }
      } else if (opcode == ALU_OPCODE_SUB) {
        if (!use_imm) {
          tmp = inputs[i][j] - inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] - immediate[i / BATCH];
        }
      } else if (opcode == ALU_OPCODE_MUL) {
        if (!use_imm) {
          tmp = inputs[i][j] * inputs[i][j + vector_size];
        } else {
          tmp = inputs[i][j] * immediate[i / BATCH];
        }
      } else if (opcode == ALU_OPCODE_SHL) {
        tmp = inputs[i][j] << immediate[i / BATCH];
      } else if (opcode == ALU_OPCODE_SHR) {
        tmp = inputs[i][j] >> immediate[i / BATCH];
      }
      // Set
      outputs_ref[i][j] = (out_T) tmp;
    }
  }

  // Pack input buffer
  acc_T *bias_buf = (acc_T *) allocBuffer(ACC_ELEM_BYTES * batch * tx_size * input_sets);
  packBuffer<acc_T, ACC_WIDTH>(bias_buf, inputs, batch, vector_size * input_sets, BATCH, BLOCK_OUT);

  // Prepare output buffer
  out_T *output_buf = (out_T *) allocBuffer(INP_ELEM_BYTES * batch * tx_size * input_sets);

#ifdef NO_SIM
  // Invoke the VTA
  uint64_t t_fpga = vta(ins_size, insn_buf, uop_buf, NULL, NULL, bias_buf, output_buf);
  // Report on timining
  printf("INFO - Synchronization time: %.3lfms\n", (double) t_fpga / 1E6);
  printf("INFO - Throughput: %.3lfGOps/s\n", (double) vector_size * batch / t_fpga);
#else
  // Invoke the VTA
  vta(
    ins_size,
    (volatile insn_T *) insn_buf,
    (volatile uop_T *) uop_buf,
    (volatile inp_vec_T *) NULL,
    (volatile wgt_vec_T *) NULL,
    (volatile acc_vec_T *) bias_buf,
    (volatile inp_vec_T *) output_buf
  );
#endif

  // Unpack output buffer
  out_T **outputs = alloc2dArray<out_T>(batch, vector_size);
  unpackBuffer<out_T, OUT_WIDTH>(outputs, output_buf, batch, vector_size, BATCH, BLOCK_OUT);

  // Correctness checks
  int err = 0;
  for (int i = 0; i < batch; i ++) {
    for (int j = 0; j < vector_size; j ++) {
      if (outputs_ref[i][j] != outputs[i][j]) {
        err++;
#if DEBUG==1
        printf("DEBUG - %d, %d: expected 0x%x but got 0x%x\n", i, j, (int) outputs_ref[i][j],
            (int) outputs[i][j]);
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

  assert(block % BLOCK_IN == 0);
  assert(block % BLOCK_OUT == 0);
  assert(block % BATCH == 0);
  assert(channels % block == 0);
  assert(batch % block == 0);

  printf("=====================================================================================\n");
  printf("INFO - Blocked GEMM test: batch=%d, channels=%d, block=%d, uop_compression=%d, \
virtual_threads=%d\n",
    batch, channels, block, uop_compression, virtual_threads);

  // Input/output channels
  int in_feat = channels;
  int out_feat = channels;
  // Derive number of elements that need to be loaded/stored
  int ins_size = batch / block * out_feat / block * (2 + in_feat / block * 3) + 2;
  int uop_size = uop_compression ? block / BATCH * virtual_threads :
    block / BATCH * block / BLOCK_IN * block / BLOCK_OUT * virtual_threads;
  int inp_size = batch / BATCH * in_feat / BLOCK_IN;
  int wgt_size = in_feat / BLOCK_IN * out_feat / BLOCK_OUT;
  int out_size = batch / BATCH * out_feat / BLOCK_OUT;
  // Blocked buffer sizes (in terms of elements)
  int inp_block_size = block / BATCH * block / BLOCK_IN;
  int wgt_block_size = block / BLOCK_IN * block / BLOCK_OUT;
  int out_block_size = block / BATCH * block / BLOCK_OUT;
  // Make sure we don't exceed buffer bounds
  assert(uop_size <= UOP_BUFF_DEPTH);
  assert(inp_block_size <= INP_BUFF_DEPTH);
  assert(wgt_block_size <= WGT_BUFF_DEPTH);
  assert(out_block_size <= ACC_BUFF_DEPTH);

  // Initialize instruction buffer
  VTAGenericInsn *insn_buf = (VTAGenericInsn *) allocBuffer(sizeof(VTAGenericInsn) * ins_size);
  int insn_idx = 0;

  // Load uops
  insn_buf[insn_idx ++] = get1DLoadStoreInsn(OPCODE_LOAD, MEM_ID_UOP, 0, 0, uop_size, 0, 0, 0, 0);
  // Iterate over batch blocks
  for (int i = 0; i < batch; i += block) {
    // Iterate over output channel blocks
    for (int j = 0; j < out_feat; j += block) {
      // Load bias block (pop next if not first, push prev)
      insn_buf[insn_idx ++] = get2DLoadStoreInsn(
        OPCODE_LOAD,                                        // opcode
        MEM_ID_ACC,                                         // type
        0,                                                  // sram offset
        (i / BATCH * out_feat + j) / BLOCK_OUT,             // dram offset
        block / BATCH,                                      // y size
        block / BLOCK_OUT,                                  // x size
        out_feat / BLOCK_OUT,                               // x stride
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
              (i != batch - block)
            );
          bool push_next = (k + l == in_feat - block);
          // Load weight block (pop next)
          insn_buf[insn_idx ++] = get2DLoadStoreInsn(
            OPCODE_LOAD,                                    // opcode
            MEM_ID_WGT,                                     // type
            l / BLOCK_IN * block / BLOCK_OUT,               // sram offset
            (j / BLOCK_OUT * in_feat + k + l) / BLOCK_IN,   // dram offset
            block / BLOCK_OUT,                              // y size
            block / BLOCK_IN,                               // x size
            in_feat / BLOCK_IN,                             // x stride
            0,                                              // y pad
            0,                                              // x pad
            0,                                              // pop prev dep
            pop,                                            // pop next dep
            0,                                              // push prev dep
            0);                                             // push next dep
          // Load input block (push next)
          insn_buf[insn_idx ++] = get2DLoadStoreInsn(
            OPCODE_LOAD,                                    // opcode
            MEM_ID_INP,                                     // type
            l / BLOCK_IN * block / BATCH,                   // sram offset
            (i / BATCH * in_feat + k + l) / BLOCK_IN,       // dram offset
            block / BATCH,                                  // y size
            block / BLOCK_IN,                               // x size
            in_feat / BLOCK_IN,                             // x stride
            0,                                              // y pad
            0,                                              // x pad
            0,                                              // pop prev dep
            0,                                              // pop next dep
            0,                                              // push prev dep
            1);                                             // push next dep
          // Perform GEMM (pop prev, push prev if not last, push next if last)
          insn_buf[insn_idx ++] = getGEMMInsn(
            l / block * uop_size / virtual_threads,         // uop offset
            block / BATCH,                                  // batch
            block / BLOCK_IN,                               // in_feat
            block / BLOCK_OUT,                              // out_feat
            uop_compression,                                // uop_compression
            1,                                              // pop_prev_dep
            0,                                              // pop_next_dep
            push_prev,                                      // push prev dep
            push_next);                                     // push_next_dep
        }
      }
      // Store output block (pop prev, push prev if not last)
      insn_buf[insn_idx ++] = get2DLoadStoreInsn(
        OPCODE_STORE,                                       // opcode
        MEM_ID_OUT,                                         // type
        0,                                                  // sram offset
        (i / BATCH * out_feat + j) / BLOCK_OUT,             // dram offset
        block / BATCH,                                      // y size
        block / BLOCK_OUT,                                  // x size
        out_feat / BLOCK_OUT,                               // x stride
        0,                                                  // y pad
        0,                                                  // x pad
        1,                                                  // pop prev dep
        0,                                                  // pop next dep
        1,                                                  // pop prev dep
        0);                                                 // push next dep
    }
  }
  // Finish
  insn_buf[insn_idx ++] = getFinishInsn(0, 1);

  // Prepare the uop buffer
  VTAUop * uop_buf = getGEMMUops(block / BATCH, block / BLOCK_IN, block / BLOCK_OUT, uop_compression,
    virtual_threads > 1);

#if DEBUG==1
  printInstruction(ins_size, insn_buf);
  printMicroOp(uop_size, uop_buf);
#endif

  // Initialize inputs
  inp_T **inputs = allocInit2dArray<inp_T, INP_WIDTH>(batch, in_feat);
  // Initialize weights
  wgt_T **weights = allocInit2dArray<wgt_T, WGT_WIDTH>(out_feat, in_feat);
  // Initialize biases
  acc_T **biases = allocInit2dArray<acc_T, ACC_WIDTH>(batch, out_feat);

  // Reference GEMM implementation
  out_T **outputs_ref = alloc2dArray<out_T>(batch, out_feat);
  for (int i = 0; i < batch; i ++) {
    for (int j = 0; j < out_feat; j ++) {
      acc_T sum = biases[i][j];
      for (int k = 0; k < in_feat; k ++) {
        sum += (acc_T) (inputs[i][k] * weights[j][k]);
      }
      // Set
      outputs_ref[i][j] = (out_T) sum;
    }
  }

  // Prepare the input buffer
  inp_T *input_buf = (inp_T *) allocBuffer(INP_ELEM_BYTES * inp_size);
  packBuffer<inp_T, INP_WIDTH>(input_buf, inputs, batch, in_feat, BATCH, BLOCK_IN);
  // Prepare the weight buffer
  wgt_T *weight_buf = (wgt_T *) allocBuffer(WGT_ELEM_BYTES * wgt_size);
  packBuffer<wgt_T, WGT_WIDTH>(weight_buf, weights, out_feat, in_feat, BLOCK_OUT, BLOCK_IN);
  // Prepare the bias buffer
  acc_T *bias_buf = (acc_T *) allocBuffer(ACC_ELEM_BYTES * out_size);
  packBuffer<acc_T, ACC_WIDTH>(bias_buf, biases, batch, out_feat, BATCH, BLOCK_OUT);
  // Prepare the output buffer
  out_T *output_buf = (out_T *) allocBuffer(INP_ELEM_BYTES * out_size);

#ifdef NO_SIM
  // Invoke the VTA
  uint64_t t_fpga = vta(ins_size, insn_buf, uop_buf, input_buf, weight_buf, bias_buf, output_buf);
  // Report on timining
  printf("INFO - Synchronization time: %.3lfms\n", (double) t_fpga / 1E6);
  printf("INFO - Throughput: %.3lfGOPs/s\n", (double) batch * in_feat * out_feat * 2 / t_fpga);
#else
  // Invoke the VTA
  vta(
    ins_size,
    (volatile insn_T *) insn_buf,
    (volatile uop_T *) uop_buf,
    (volatile inp_vec_T *) input_buf,
    (volatile wgt_vec_T *) weight_buf,
    (volatile acc_vec_T *) bias_buf,
    (volatile inp_vec_T *) output_buf
  );
#endif

  // Unpack output data
  out_T **outputs = alloc2dArray<out_T>(batch, out_feat);
  unpackBuffer<out_T, OUT_WIDTH>(outputs, output_buf, batch, out_feat, BATCH, BLOCK_OUT);

  // Correctness checks
  int err = 0;
  for (int i = 0; i < batch; i ++) {
    for (int j = 0; j < out_feat; j ++) {
      if (outputs_ref[i][j] != outputs[i][j]) {
        err++;
#if DEBUG==1
        printf("DEBUG - %d, %d: expected 0x%x but got 0x%x\n", i, j, (int) outputs_ref[i][j],
            (int) outputs[i][j]);
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
  freeBuffer((void *) insn_buf);
  freeBuffer((void *) uop_buf);
  freeBuffer((void *) input_buf);
  freeBuffer((void *) weight_buf);
  freeBuffer((void *) bias_buf);
  freeBuffer((void *) output_buf);

  if (err == 0) {
    printf("INFO - Blocked GEMM test successful!\n");
    return 0;
  } else {
    printf("INFO - Blocked GEMM test failed, got %d errors!\n", err);
    return -1;
  }

}
