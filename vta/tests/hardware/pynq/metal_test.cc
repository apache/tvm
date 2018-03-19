/*!
 *  Copyright (c) 2018 by Contributors
 * \file driver_test.cpp
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

// VTA invocation (present the same abstraction as in the simulation tests)
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

#if VTA_DEBUG == 1
  printf("INFO - Programming FPGA: %s!\n", bitstream);
#endif

  // Program VTA
  VTAProgram(bitstream);
  // Get VTA handles
  VTAHandle vta_fetch_handle = VTAMapRegister(VTA_FETCH_ADDR, VTA_RANGE);
  VTAHandle vta_load_handle = VTAMapRegister(VTA_LOAD_ADDR, VTA_RANGE);
  VTAHandle vta_compute_handle = VTAMapRegister(VTA_COMPUTE_ADDR, VTA_RANGE);
  VTAHandle vta_store_handle = VTAMapRegister(VTA_STORE_ADDR, VTA_RANGE);

  // Physical address pointers
  uint32_t insn_phy = insns ? cma_get_phy_addr(insns) : 0;
  uint32_t uop_phy = uops ? cma_get_phy_addr(uops) : 0;
  uint32_t input_phy = inputs ? cma_get_phy_addr(inputs) : 0;
  uint32_t weight_phy = weights ? cma_get_phy_addr(weights) : 0;
  uint32_t bias_phy = biases ? cma_get_phy_addr(biases) : 0;
  uint32_t output_phy = outputs ? cma_get_phy_addr(outputs) : 0;

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
