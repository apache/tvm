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
#include "../../../src/pynq/pynq_driver.h"
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
