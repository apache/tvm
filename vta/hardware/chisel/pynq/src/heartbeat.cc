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
#define VTA_INS_COUNT 1
#define VTA_INST_BYTES VTA_INS_COUNT*8

void gen_finish(uint64_t* insn) {
  insn[0] = 3;
  insn[1] = 0;
}

int main(void) {

  void* insn;

  insn = VTAMemAlloc(VTA_INST_BYTES, 0);

  gen_finish(reinterpret_cast<uint64_t*>(insn));

  void* vta_handle = VTAMapRegister(VTA_BASE_ADDR, 0x100);

  VTAWriteMappedReg(vta_handle, 0x04, 0x0);
  VTAWriteMappedReg(vta_handle, 0x08, VTA_INS_COUNT);
  VTAWriteMappedReg(vta_handle, 0x0c, VTAMemGetPhyAddr(insn));
  VTAWriteMappedReg(vta_handle, 0x00, 1);

  int flag = 0, t = 0, cycles = 0;
  for (t = 0; t < 10000000; ++t) {
    flag = VTAReadMappedReg(vta_handle, 0x0);
    if (flag & 2) break;
  }

  cycles = VTAReadMappedReg(vta_handle, 0x4);

  VTAUnmapRegister(vta_handle, VTA_RANGE);

  printf("cycles:%d\n", cycles);

  VTAMemFree(insn);

  return 0;
}
