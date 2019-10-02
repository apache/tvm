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
 * \file metal_test.cpp
 * \brief Bare-metal test to test driver and VTA design.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vta/driver.h>
#ifdef VTA_TARGET_PYNQ
#  include "../../../src/pynq/pynq_driver.h"
#endif  // VTA_TARGET_PYNQ
#include "../common/test_lib.h"

#define VTA_BASE_ADDR 0x43c00000
#define VTA_ENABLE_CACHE 1

int main() {

  int vec_size = 36;;
  int insn_count = 7;
  int insn_idx = 0;
  int block_size = 16;
  int i;

  VTAGenericInsn *insn_buf =
      static_cast<VTAGenericInsn *>(VTAMemAlloc(sizeof(VTAGenericInsn) * insn_count, VTA_ENABLE_CACHE));

  uint32_t *acc_buf =
      static_cast<uint32_t *>(VTAMemAlloc(sizeof(uint32_t) * block_size * vec_size, VTA_ENABLE_CACHE));

  VTAUop *uop_buf =
      static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop), VTA_ENABLE_CACHE));

  uint8_t *out_buf =
      static_cast<uint8_t *>(VTAMemAlloc(sizeof(uint8_t) * block_size * vec_size, VTA_ENABLE_CACHE));

  for (i = 0; i < vec_size; i++) {
    uop_buf[i].dst_idx = i;
    uop_buf[i].src_idx = i;
  }

  for (i = 0; i < (block_size * vec_size); i++) {
    acc_buf[i] = i;
    out_buf[i] = 0;
  }

  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_ACC,                                     // type
      0,                                                  // sram offset
      VTAMemGetPhyAddr(acc_buf),                          // dram offset
      vec_size,                                           // size
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      0,                                                  // push prev dep
      0);                                                 // push next dep

  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_UOP,                                     // type
      0,                                                  // sram offset
      VTAMemGetPhyAddr(uop_buf),                          // dram offset
      1,                                                  // size
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      0,                                                  // push prev dep
      0);                                                 // push next dep

  insn_buf[insn_idx++] = getALUInsn(
      VTA_ALU_OPCODE_SHR,                                 // opcode
      vec_size,                                           // vector size
      true,                                               // use imm
      0,                                                  // imm
      true,                                               // uop compression
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      0,                                                  // push prev dep
      1);                                                 // push next dep

  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_STORE,                                   // opcode
      VTA_MEM_ID_OUT,                                     // type
      0,                                                  // sram offset
      VTAMemGetPhyAddr(out_buf),                          // dram offset
      vec_size,                                           // size
      1,                                                  // pop prev dep
      0,                                                  // pop next dep
      1,                                                  // push prev dep
      0);                                                 // push next dep

  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_INP,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      0,                                                  // size
      0,                                                  // pop prev dep
      0,                                                  // pop next dep
      0,                                                  // push prev dep
      1);                                                 // push next dep

  insn_buf[insn_idx++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,                                    // opcode
      VTA_MEM_ID_UOP,                                     // type
      0,                                                  // sram offset
      0,                                                  // dram offset
      0,                                                  // size
      1,                                                  // pop prev dep
      1,                                                  // pop next dep
      0,                                                  // push prev dep
      0);                                                 // push next dep

  insn_buf[insn_idx++] = getFinishInsn(0, 0);
  printInstruction(insn_count, insn_buf);

  for (i = 0; i < insn_count; i++) {
    printf("i:%x inst:%016llx%016llx\n", i, insn_buf[i].word_1, insn_buf[i].word_0);
  }

  VTAFlushCache(VTAMemGetPhyAddr(insn_buf), sizeof(VTAGenericInsn) * insn_count);
  VTAFlushCache(VTAMemGetPhyAddr(uop_buf), sizeof(VTAUop) * vec_size);
  VTAFlushCache(VTAMemGetPhyAddr(acc_buf), sizeof(uint32_t) * block_size * vec_size);

  VTADeviceHandle vta_handle = VTADeviceAlloc();

  int vta_timeout = 0;
  uint32_t wait_cycles = 10000000;
  vta_timeout = VTADeviceRun(vta_handle, VTAMemGetPhyAddr(insn_buf), insn_count, wait_cycles);

  if (vta_timeout) {
    printf("VTA timeout\n");
  }

  VTAInvalidateCache(VTAMemGetPhyAddr(out_buf), sizeof(uint8_t) * block_size * vec_size);

  for (i = 0; i < (block_size * vec_size); i++) {
    printf("i:%x acc:%08x out:%02x\n", i, acc_buf[i], out_buf[i]);
  }

  VTADeviceFree(vta_handle);
  VTAMemFree(insn_buf);
  VTAMemFree(uop_buf);
  VTAMemFree(acc_buf);
  VTAMemFree(out_buf);

  return 0;
}
