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

int main(void) {
#if VTA_DEBUG == 1
  printParameters();
#endif

  int status = 0;

  // Run ALU test (vector-scalar operators)
  status |= alu_test(VTA_ALU_OPCODE_MAX, true, 16, 128, true);
  status |= alu_test(VTA_ALU_OPCODE_MAX, true, 16, 128, false);
  status |= alu_test(VTA_ALU_OPCODE_ADD, true, 16, 128, true);
  status |= alu_test(VTA_ALU_OPCODE_ADD, true, 16, 128, false);
  status |= alu_test(VTA_ALU_OPCODE_SHR, true, 16, 128, true);
  status |= alu_test(VTA_ALU_OPCODE_SHR, true, 16, 128, false);

  // Run ALU test (vector-vector operators)
  status |= alu_test(VTA_ALU_OPCODE_MAX, false, 16, 128, true);
  status |= alu_test(VTA_ALU_OPCODE_MAX, false, 16, 128, false);
  status |= alu_test(VTA_ALU_OPCODE_ADD, false, 16, 128, true);
  status |= alu_test(VTA_ALU_OPCODE_ADD, false, 16, 128, false);

  // Run blocked GEMM test
  status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 2);
  status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 2);
  status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, true, 1);
  status |= blocked_gemm_test(256, 256, VTA_BLOCK_OUT*4, false, 1);

  if (status == 0) {
    printf("\nINFO - Unit tests successful!\n");
  } else {
    printf("\nINTO - Unit tests failed!\n");
  }

  return status;
}
